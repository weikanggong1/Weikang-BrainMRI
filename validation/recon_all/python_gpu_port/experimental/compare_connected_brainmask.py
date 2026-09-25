"""Compare the connected T1-to-brainmask outputs with fixed references."""

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
    left, right = np.asarray(first.dataobj), np.asarray(second.dataobj)
    diff = left.astype(np.int16) - right.astype(np.int16)
    a, b = gzip.decompress(candidate.read_bytes()), gzip.decompress(reference.read_bytes())
    data_end = 284 + left.size * left.dtype.itemsize
    return {
        "candidate_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
        "reference_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
        "shape_equal": left.shape == right.shape,
        "dtype_equal": left.dtype == right.dtype,
        "voxel_mismatches": int(np.count_nonzero(diff)),
        "max_abs_intensity_difference": int(np.max(np.abs(diff))),
        "affine_max_abs_mm": float(np.max(np.abs(first.affine - second.affine))),
        "mgh_header_equal": a[:284] == b[:284],
        "mgh_header_payload_equal": a[:data_end] == b[:data_end],
        "decompressed_bytes_equal": a == b,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--native-mask", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    rows = {}
    for name in ("nu.mgz", "T1.mgz", "brainmask.mgz"):
        candidate = args.candidate / "mri" / name
        rows[name] = {"archived_official": compare(candidate, args.official / "mri" / name)}
        previous = args.previous / "mri" / name
        if previous.is_file():
            rows[name]["previous_python"] = compare(candidate, previous)
    if args.native_mask is not None:
        rows["brainmask.mgz"]["fresh_native_same_input"] = compare(
            args.candidate / "mri/brainmask.mgz", args.native_mask)
    candidate_t1 = np.asarray(nib.load(str(args.candidate / "mri/T1.mgz")).dataobj)
    reference_t1 = np.asarray(nib.load(str(args.official / "mri/T1.mgz")).dataobj)
    candidate_strip = np.asarray(nib.load(str(args.candidate / "mri/synthstrip.mgz")).dataobj)
    reference_strip = np.asarray(nib.load(str(args.official / "mri/synthstrip.mgz")).dataobj)
    candidate_mask = np.asarray(nib.load(str(args.candidate / "mri/brainmask.mgz")).dataobj)
    reference_mask = np.asarray(nib.load(str(args.official / "mri/brainmask.mgz")).dataobj)
    t1_delta = candidate_t1.astype(np.int16) - reference_t1.astype(np.int16)
    mask_delta = candidate_mask.astype(np.int16) - reference_mask.astype(np.int16)
    rows["brainmask.mgz"]["archived_difference_attribution"] = {
        "synthstrip_voxel_mismatches": int(np.count_nonzero(candidate_strip != reference_strip)),
        "T1_difference_voxels_inside_synthstrip": int(np.count_nonzero((t1_delta != 0) & (candidate_strip != 0))),
        "brainmask_difference_voxels_unexplained_by_masked_T1_difference": int(np.count_nonzero(
            mask_delta - np.where(candidate_strip != 0, t1_delta, 0))),
    }
    args.report.write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
