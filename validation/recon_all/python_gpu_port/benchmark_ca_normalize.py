"""Real-data, file-level validation of the fixed Python GCA normalization."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from time import perf_counter

import nibabel as nib
import numpy as np

from fnit.recon_all.ca_normalize_python import run_ca_normalize


def compare_mgh(candidate: Path, reference: Path) -> dict:
    image = nib.load(str(reference))
    frame_bytes = int(np.prod(image.shape[:3])) * image.get_data_dtype().itemsize
    count = int(np.prod(image.shape)) * image.get_data_dtype().itemsize
    voxel_differences = [0] * (image.shape[3] if len(image.shape) == 4 else 1)
    maximum = 0.
    with gzip.open(candidate, "rb") as left, gzip.open(reference, "rb") as right:
        same_header = left.read(284) == right.read(284)
        for offset in range(0, count, 1 << 20):
            length = min(1 << 20, count - offset)
            a, b = left.read(length), right.read(length)
            if len(a) != length or len(b) != length:
                raise ValueError("Truncated MGH payload")
            if a == b:
                continue
            av = np.frombuffer(a, dtype=image.get_data_dtype())
            bv = np.frombuffer(b, dtype=image.get_data_dtype())
            voxel_differences[offset // frame_bytes] += int(np.count_nonzero(av != bv))
            maximum = max(maximum, float(np.max(np.abs(av.astype(np.float64) - bv))))
        candidate_footer, reference_footer = left.read(), right.read()
    return {"shape": [int(dimension) for dimension in image.shape],
            "header_284_equal": same_header,
            "voxel_differences_per_frame": voxel_differences,
            "maximum_absolute_voxel_difference": maximum,
            "decompressed_mgh_equal": same_header and not any(voxel_differences) and
            candidate_footer == reference_footer,
            "candidate_footer_bytes": len(candidate_footer),
            "reference_footer_bytes": len(reference_footer)}


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("nu", "mask", "gca", "lta", "reference_norm", "reference_ctrl", "output_dir"):
        parser.add_argument(f"--{name.replace('_', '-')}", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    norm = args.output_dir / "norm.mgz"
    ctrl = args.output_dir / "ctrl_pts.mgz"
    start = perf_counter()
    stages = run_ca_normalize(args.nu, args.mask, args.gca, args.lta, norm, ctrl)
    report = {"wall_seconds": perf_counter() - start, "stage": stages,
              "norm": compare_mgh(norm, args.reference_norm),
              "ctrl_pts": compare_mgh(ctrl, args.reference_ctrl)}
    output = args.output_dir / "ca_normalize_comparison.json"
    rendered = json.dumps(report, indent=2)
    output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
