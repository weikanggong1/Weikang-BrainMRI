"""Run the connected T1 import/conform chain against one native subject."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from fnit.recon_all.input_chain import run_input_chain


def _compare(reference: Path, candidate: Path) -> dict:
    native = nib.load(str(reference))
    python = nib.load(str(candidate))
    a, b = np.asarray(native.dataobj), np.asarray(python.dataobj)
    raw_a = gzip.decompress(reference.read_bytes())
    raw_b = gzip.decompress(candidate.read_bytes())
    payload_end = 284 + a.size * a.dtype.itemsize
    return {
        "reference_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
        "candidate_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
        "shape_equal": a.shape == b.shape,
        "dtype_equal": a.dtype == b.dtype,
        "voxel_mismatch_count": int(np.count_nonzero(a != b)) if a.shape == b.shape else None,
        "affine_max_abs_error": float(np.max(np.abs(native.affine - python.affine))),
        "mgh_header_equal": raw_a[:284] == raw_b[:284],
        "header_and_voxel_payload_equal": raw_a[:payload_end] == raw_b[:payload_end],
        "footer_equal": raw_a[payload_end:] == raw_b[payload_end:],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("t1", type=Path)
    parser.add_argument("reference_subject", type=Path)
    parser.add_argument("candidate_subject", type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    run = run_input_chain(args.t1, args.candidate_subject, device=args.device)
    comparisons = {
        relative: _compare(args.reference_subject / relative, args.candidate_subject / relative)
        for relative in ("mri/orig/001.mgz", "mri/rawavg.mgz", "mri/orig.mgz")
    }
    passed = all(x["shape_equal"] and x["dtype_equal"] and
                 x["voxel_mismatch_count"] == 0 and x["mgh_header_equal"] and
                 x["header_and_voxel_payload_equal"] for x in comparisons.values())
    result = {"scope": "connected T1 input through orig.mgz, CPU",
              "run": run, "comparisons": comparisons, "passed": passed}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"passed": passed, "timings": run,
                      "voxel_mismatches": {k: v["voxel_mismatch_count"]
                                           for k, v in comparisons.items()}}, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
