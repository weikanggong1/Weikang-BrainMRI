"""Replay 33-class SynthSeg from a Python-conformed T1 against native run."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import time

import nibabel as nib
import numpy as np

from fnit.synthseg_parc import SynthSeg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_orig", type=Path)
    parser.add_argument("official_seg", type=Path)
    parser.add_argument("official_volumes", type=Path)
    parser.add_argument("weights", type=Path)
    parser.add_argument("lut", type=Path)
    parser.add_argument("output_seg", type=Path)
    parser.add_argument("output_volumes", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    if not args.device.startswith("cuda"):
        parser.error("the connected comparator requires CUDA")
    started = time.perf_counter()
    result = SynthSeg(weights=args.weights, device=args.device, threads=4)(
        args.input_orig, keep_geometry=True, color_lut=args.lut)
    result.segmentation.save(str(args.output_seg))
    result.write_volumes_csv(args.input_orig, args.output_volumes)
    elapsed = time.perf_counter() - started

    native, candidate = nib.load(str(args.official_seg)), nib.load(str(args.output_seg))
    a, b = np.asarray(native.dataobj), np.asarray(candidate.dataobj)
    raw_a = gzip.decompress(args.official_seg.read_bytes())
    raw_b = gzip.decompress(args.output_seg.read_bytes())
    payload_end = 284 + a.size * a.dtype.itemsize
    with args.official_volumes.open(newline="") as handle:
        native_rows = list(csv.reader(handle))
    with args.output_volumes.open(newline="") as handle:
        candidate_rows = list(csv.reader(handle))
    volume_differences = [
        {"structure": name, "official_mm3": float(left),
         "candidate_mm3": float(right), "absolute_difference_mm3": abs(float(left) - float(right))}
        for name, left, right in zip(native_rows[0][1:], native_rows[1][1:],
                                     candidate_rows[1][1:])
    ] if len(native_rows) == len(candidate_rows) == 2 and native_rows[0] == candidate_rows[0] else []
    label_metrics = []
    if a.shape == b.shape:
        labels_a, labels_b = a.astype(np.int32), b.astype(np.int32)
        official_count = np.bincount(labels_a.ravel())
        candidate_count = np.bincount(labels_b.ravel())
        intersection = np.bincount(labels_a[labels_a == labels_b])
        for label in np.union1d(np.unique(labels_a), np.unique(labels_b)):
            label = int(label)
            n_official = int(official_count[label]) if label < len(official_count) else 0
            n_candidate = int(candidate_count[label]) if label < len(candidate_count) else 0
            n_both = int(intersection[label]) if label < len(intersection) else 0
            label_metrics.append({"label": label, "official_voxels": n_official,
                                  "candidate_voxels": n_candidate,
                                  "dice": 2 * n_both / (n_official + n_candidate)})
    report = {
        "scope": "connected Python T1 input through 33-class SynthSeg; CUDA inference",
        "input_sha256": hashlib.sha256(args.input_orig.read_bytes()).hexdigest(),
        "lut_sha256": hashlib.sha256(args.lut.read_bytes()).hexdigest(),
        "reference_sha256": hashlib.sha256(args.official_seg.read_bytes()).hexdigest(),
        "candidate_sha256": hashlib.sha256(args.output_seg.read_bytes()).hexdigest(),
        "device": args.device,
        "seconds": elapsed,
        "near_tie_voxels": result.near_tie_voxels,
        "shape_equal": a.shape == b.shape,
        "dtype_equal": a.dtype == b.dtype,
        "voxel_mismatch_count": int(np.count_nonzero(a != b)) if a.shape == b.shape else None,
        "affine_max_abs_error": float(np.max(np.abs(native.affine - candidate.affine))),
        "mgh_header_equal": raw_a[:284] == raw_b[:284],
        "header_and_voxel_payload_equal": raw_a[:payload_end] == raw_b[:payload_end],
        "volume_csv_equal": native_rows == candidate_rows,
        "volume_csv_reference_rows": len(native_rows),
        "volume_csv_candidate_rows": len(candidate_rows),
        "volume_max_abs_difference_mm3": max((row["absolute_difference_mm3"]
                                                for row in volume_differences), default=None),
        "volume_differences": volume_differences,
        "label_metrics": label_metrics,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not (report["shape_equal"] and report["dtype_equal"] and
            report["voxel_mismatch_count"] == 0 and report["mgh_header_equal"] and
            report["volume_csv_equal"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
