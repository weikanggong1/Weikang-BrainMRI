"""Replay EntoWM from the connected Python nu.mgz against the official subject."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import time

import nibabel as nib
import numpy as np
import torch

from fnit.recon_all.sclimbic import mri_entowm_seg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_nu", type=Path)
    parser.add_argument("official_entowm", type=Path)
    parser.add_argument("weights", type=Path)
    parser.add_argument("output_entowm", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--official-stats", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.device == "cpu":
        torch.backends.mkldnn.enabled = False
    started = time.perf_counter()
    stats_path = args.output_entowm.with_suffix(".stats") if args.official_stats else None
    mri_entowm_seg(args.input_nu, args.output_entowm, args.weights,
                   device=args.device, stats_path=stats_path)
    elapsed = time.perf_counter() - started
    official, candidate = (nib.load(str(path)) for path in
                           (args.official_entowm, args.output_entowm))
    a, b = np.asarray(official.dataobj), np.asarray(candidate.dataobj)
    if a.shape != b.shape:
        raise ValueError("EntoWM grids differ")
    labels = []
    for label in np.union1d(np.unique(a), np.unique(b)):
        reference_count = int(np.count_nonzero(a == label))
        candidate_count = int(np.count_nonzero(b == label))
        shared = int(np.count_nonzero((a == label) & (b == label)))
        labels.append({"label": int(label), "official_voxels": reference_count,
                       "candidate_voxels": candidate_count,
                       "dice": 2 * shared / (reference_count + candidate_count)})
    raw_a, raw_b = (gzip.decompress(path.read_bytes()) for path in
                    (args.official_entowm, args.output_entowm))
    report = {"scope": "Python nu to EntoWM isolated stage; archived official comparator",
              "device": args.device, "seconds": elapsed,
              "voxel_mismatches": int(np.count_nonzero(a != b)),
              "maximum_label_difference": int(np.max(np.abs(a.astype(np.int32) - b.astype(np.int32)))),
              "minimum_foreground_dice": min(row["dice"] for row in labels if row["label"] != 0),
              "dtype_equal": a.dtype == b.dtype,
              "affine_max_abs_mm": float(np.max(np.abs(official.affine - candidate.affine))),
              "mgh_header_equal": raw_a[:284] == raw_b[:284],
              "header_and_voxel_payload_equal":
                  raw_a[:284 + a.size * a.dtype.itemsize] ==
                  raw_b[:284 + b.size * b.dtype.itemsize],
              "labels": labels}
    if args.official_stats:
        def rows(path: Path) -> dict[int, tuple[int, float]]:
            return {int(parts[1]): (int(parts[2]), float(parts[3]))
                    for line in path.read_text().splitlines()
                    if line and not line.startswith("#")
                    for parts in [line.split()]}
        reference, actual = rows(args.official_stats), rows(stats_path)
        if reference.keys() != actual.keys():
            raise ValueError("EntoWM stats structure IDs differ")
        report["stats_numeric"] = [
            {"label": label, "official_voxels": reference[label][0],
             "candidate_voxels": actual[label][0],
             "official_volume_mm3": reference[label][1],
             "candidate_volume_mm3": actual[label][1],
             "absolute_volume_difference_mm3":
                 abs(reference[label][1] - actual[label][1])}
            for label in reference]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in (
        "device", "seconds", "voxel_mismatches", "minimum_foreground_dice",
        "dtype_equal", "mgh_header_equal")}, indent=2))


if __name__ == "__main__":
    main()
