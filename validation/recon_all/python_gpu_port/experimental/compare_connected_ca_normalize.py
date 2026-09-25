"""Compare connected GCA-normalized volumes with the archived official subject."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np


def compare(candidate: Path, reference: Path) -> dict:
    first, second = nib.load(str(candidate)), nib.load(str(reference))
    if first.shape != second.shape:
        raise ValueError(f"volume shapes differ: {first.shape}, {second.shape}")
    counts = []
    errors = []
    frames = range(first.shape[3]) if len(first.shape) == 4 else (None,)
    for frame in frames:
        a = np.asarray(first.dataobj[..., frame] if frame is not None else first.dataobj)
        b = np.asarray(second.dataobj[..., frame] if frame is not None else second.dataobj)
        delta = a.astype(np.float64) - b.astype(np.float64)
        counts.append(int(np.count_nonzero(delta)))
        errors.append(float(np.max(np.abs(delta))))
    with gzip.open(candidate, "rb") as stream:
        candidate_header = stream.read(284)
    with gzip.open(reference, "rb") as stream:
        reference_header = stream.read(284)
    return {
        "candidate_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
        "reference_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
        "shape": [int(v) for v in first.shape], "dtype_equal": first.get_data_dtype() == second.get_data_dtype(),
        "voxel_mismatches_by_frame": counts,
        "max_abs_difference_by_frame": errors,
        "affine_max_abs_mm": float(np.max(np.abs(first.affine - second.affine))),
        "mgh_header_equal": candidate_header == reference_header,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate_mri", type=Path)
    parser.add_argument("official_mri", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    rows = {name: compare(args.candidate_mri / name, args.official_mri / name)
            for name in ("norm.mgz", "ctrl_pts.mgz")}
    nu_difference = np.asarray(nib.load(str(args.candidate_mri / "nu.mgz")).dataobj) != \
                    np.asarray(nib.load(str(args.official_mri / "nu.mgz")).dataobj)
    norm_difference = np.asarray(nib.load(str(args.candidate_mri / "norm.mgz")).dataobj) != \
                      np.asarray(nib.load(str(args.official_mri / "norm.mgz")).dataobj)
    rows["norm_difference_location"] = {
        "nu_differing_voxels": int(np.count_nonzero(nu_difference)),
        "norm_differing_voxels_within_nu_difference": int(np.count_nonzero(norm_difference & nu_difference)),
        "norm_differing_voxels_outside_nu_difference": int(np.count_nonzero(norm_difference & ~nu_difference)),
    }
    args.report.write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
