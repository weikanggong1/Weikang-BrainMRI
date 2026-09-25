"""Check fixed inverse-GCAM Voronoi x field against a captured native pass."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import nibabel as nib
import numpy as np

from fnit.recon_all.ca_register_inverse_fill import soap_bubble_float, voronoi_fill


def comparison(actual: np.ndarray, expected: np.ndarray) -> dict[str, float | int | list[int] | None]:
    difference = np.abs(actual - expected)
    different = int(np.count_nonzero(difference))
    return {
        "elements": int(difference.size),
        "different": different,
        "max_abs": float(difference.max()),
        "p99_abs": float(np.quantile(difference, 0.99)),
        "first_difference": (
            list(map(int, np.unravel_index(np.argmax(difference != 0), difference.shape)))
            if different else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-x-sum", type=Path, required=True)
    parser.add_argument("--native-counts", type=Path, required=True)
    parser.add_argument("--native-control", type=Path, required=True)
    parser.add_argument("--native-voronoi", type=Path, required=True)
    parser.add_argument("--native-final", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    sums = np.asarray(nib.load(args.native_x_sum).dataobj, dtype=np.float32)
    counts = np.asarray(nib.load(args.native_counts).dataobj, dtype=np.float32)
    control = np.asarray(nib.load(args.native_control).dataobj, dtype=np.uint8).astype(bool)
    values = np.zeros_like(sums)
    np.divide(sums, counts, out=values, where=control)
    expected = np.asarray(nib.load(args.native_voronoi).dataobj, dtype=np.float32)
    started = time.perf_counter()
    actual, iterations = voronoi_fill(values, control)
    elapsed = time.perf_counter() - started
    report = {
        "voronoi": {"seconds_including_jit": elapsed, "iterations": iterations, **comparison(actual, expected)}
    }
    if args.native_final:
        expected_final = np.asarray(nib.load(args.native_final).dataobj, dtype=np.float32)
        started = time.perf_counter()
        smoothed, smoothing_iterations = soap_bubble_float(actual, control)
        report["soap_bubble"] = {
            "seconds_including_jit": time.perf_counter() - started,
            "iterations": smoothing_iterations,
            **comparison(smoothed, expected_final),
        }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(args.out.read_text())
    if report["voronoi"]["different"] or (args.native_final and report["soap_bubble"]["different"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
