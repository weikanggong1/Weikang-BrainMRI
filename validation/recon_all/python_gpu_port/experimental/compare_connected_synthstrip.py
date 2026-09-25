"""Replay SynthStrip from a Python-conformed T1 and compare the native run."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import time

import nibabel as nib
import numpy as np

from fnit.synthstrip import SynthStrip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_orig", type=Path)
    parser.add_argument("official_synthstrip", type=Path)
    parser.add_argument("weights", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    started = time.perf_counter()
    result = SynthStrip(weights=args.weights, device=args.device, threads=4)(args.input_orig)
    result.image.save(str(args.output))
    elapsed = time.perf_counter() - started

    native = nib.load(str(args.official_synthstrip))
    candidate = nib.load(str(args.output))
    a, b = np.asarray(native.dataobj), np.asarray(candidate.dataobj)
    raw_a = gzip.decompress(args.official_synthstrip.read_bytes())
    raw_b = gzip.decompress(args.output.read_bytes())
    payload_end = 284 + a.size * a.dtype.itemsize
    report = {
        "scope": "connected Python T1 input through SynthStrip; CPU inference",
        "input_sha256": hashlib.sha256(args.input_orig.read_bytes()).hexdigest(),
        "reference_sha256": hashlib.sha256(args.official_synthstrip.read_bytes()).hexdigest(),
        "candidate_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "device": args.device,
        "seconds": elapsed,
        "shape_equal": a.shape == b.shape,
        "dtype_equal": a.dtype == b.dtype,
        "voxel_mismatch_count": int(np.count_nonzero(a != b)) if a.shape == b.shape else None,
        "max_abs_voxel_error": float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
        if a.shape == b.shape else None,
        "affine_max_abs_error": float(np.max(np.abs(native.affine - candidate.affine))),
        "mgh_header_equal": raw_a[:284] == raw_b[:284],
        "header_and_voxel_payload_equal": raw_a[:payload_end] == raw_b[:payload_end],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not (report["shape_equal"] and report["dtype_equal"] and
            report["voxel_mismatch_count"] == 0 and report["mgh_header_equal"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
