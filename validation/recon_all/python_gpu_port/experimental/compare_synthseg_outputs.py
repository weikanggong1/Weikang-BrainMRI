"""Compare two fixed-subject SynthSeg segmentation and volume outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference_seg", type=Path)
    parser.add_argument("reference_csv", type=Path)
    parser.add_argument("candidate_seg", type=Path)
    parser.add_argument("candidate_csv", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    first, second = nib.load(str(args.reference_seg)), nib.load(str(args.candidate_seg))
    a, b = np.asarray(first.dataobj), np.asarray(second.dataobj)
    if a.shape != b.shape:
        raise ValueError("segmentation grids differ")
    aa, bb = a.astype(np.int32), b.astype(np.int32)
    n_a, n_b = np.bincount(aa.ravel()), np.bincount(bb.ravel())
    n_both = np.bincount(aa[aa == bb])
    labels = []
    for label in np.union1d(np.unique(aa), np.unique(bb)):
        label = int(label)
        count_a = int(n_a[label]) if label < len(n_a) else 0
        count_b = int(n_b[label]) if label < len(n_b) else 0
        shared = int(n_both[label]) if label < len(n_both) else 0
        labels.append({"label": label, "reference_voxels": count_a,
                       "candidate_voxels": count_b,
                       "dice": 2 * shared / (count_a + count_b)})
    with args.reference_csv.open(newline="") as stream:
        reference_rows = list(csv.reader(stream))
    with args.candidate_csv.open(newline="") as stream:
        candidate_rows = list(csv.reader(stream))
    if len(reference_rows) != 2 or len(candidate_rows) != 2 or reference_rows[0] != candidate_rows[0]:
        raise ValueError("SynthSeg volume table layout differs")
    volumes = [{"structure": name, "reference_mm3": float(left),
                "candidate_mm3": float(right), "absolute_difference_mm3": abs(float(left) - float(right))}
               for name, left, right in zip(reference_rows[0][1:], reference_rows[1][1:],
                                            candidate_rows[1][1:])]
    report = {
        "reference_seg_sha256": hashlib.sha256(args.reference_seg.read_bytes()).hexdigest(),
        "candidate_seg_sha256": hashlib.sha256(args.candidate_seg.read_bytes()).hexdigest(),
        "reference_csv_sha256": hashlib.sha256(args.reference_csv.read_bytes()).hexdigest(),
        "candidate_csv_sha256": hashlib.sha256(args.candidate_csv.read_bytes()).hexdigest(),
        "shape": [int(v) for v in a.shape],
        "dtype_equal": a.dtype == b.dtype,
        "affine_max_abs_mm": float(np.max(np.abs(first.affine - second.affine))),
        "segmentation_voxel_mismatches": int(np.count_nonzero(a != b)),
        "min_foreground_dice": min(row["dice"] for row in labels if row["label"] != 0),
        "label_metrics": labels,
        "volume_csv_exact": reference_rows == candidate_rows,
        "volume_max_abs_difference_mm3": max(row["absolute_difference_mm3"] for row in volumes),
        "volume_differences": volumes,
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in (
        "dtype_equal", "segmentation_voxel_mismatches", "min_foreground_dice",
        "volume_csv_exact", "volume_max_abs_difference_mm3")}, indent=2))


if __name__ == "__main__":
    main()
