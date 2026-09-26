"""Audit fresh official SynthSeg posterior replay against the archived subject."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_orig", type=Path)
    parser.add_argument("fresh_dir", type=Path)
    parser.add_argument("archived_subject", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    fresh_seg = args.fresh_dir / "synthseg.rca.mgz"
    archived_seg = args.archived_subject / "mri/synthseg.rca.mgz"
    fresh_csv = args.fresh_dir / "synthseg.vol.csv"
    archived_csv = args.archived_subject / "stats/synthseg.vol.csv"
    fresh = nib.load(str(fresh_seg))
    archived = nib.load(str(archived_seg))
    fresh_array = np.asanyarray(fresh.dataobj)
    archived_array = np.asanyarray(archived.dataobj)
    report = {
        "scope": "fresh unmodified FreeSurfer 8.2 mri_synthseg --cpu --post versus archived official subject",
        "input_sha256": sha256(args.input_orig),
        "fresh_seg_sha256": sha256(fresh_seg),
        "fresh_csv_sha256": sha256(fresh_csv),
        "fresh_posterior_sha256": sha256(args.fresh_dir / "posterior.mgz"),
        "official_exit_status": int((args.fresh_dir / "exit").read_text().strip()),
        "fresh_vs_archived_seg_shape_equal": fresh_array.shape == archived_array.shape,
        "fresh_vs_archived_seg_dtype_equal": fresh_array.dtype == archived_array.dtype,
        "fresh_vs_archived_seg_voxel_mismatches": int(np.count_nonzero(fresh_array != archived_array)),
        "fresh_vs_archived_seg_affine_max_abs_error": float(np.max(np.abs(fresh.affine - archived.affine))),
        "fresh_vs_archived_seg_mgh_header_equal": fresh.header.binaryblock == archived.header.binaryblock,
        "fresh_vs_archived_csv_byte_equal": fresh_csv.read_bytes() == archived_csv.read_bytes(),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
