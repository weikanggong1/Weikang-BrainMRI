"""Compare first pial signed gradient averaging with pinned source."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np


STATE = np.dtype([("floats", "<f4", 9), ("flags", "<i4", 3)])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--surface", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--require-exact", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(args.module))
    from fnit.recon_all.place_surface_gradient_average import average_signed_gradients

    _, faces = nib.freesurfer.read_geometry(args.surface)
    before = np.fromfile(args.probe / "lh.gradient.surface_repulsion", dtype=STATE)
    after = np.fromfile(args.probe / "lh.gradient.pre_normal_spring", dtype=STATE)
    start = time.perf_counter()
    first = average_signed_gradients(before["floats"][:, 6:9], faces, before["flags"][:, 0], 1)
    second = average_signed_gradients(first, faces, before["flags"][:, 0], 1)
    checked = {}
    for name, candidate, expected in (
        ("pass_1", first, np.fromfile(args.probe / "lh.average.1", dtype="<f4").reshape(-1, 3)),
        ("pass_2", second, np.fromfile(args.probe / "lh.average.2", dtype="<f4").reshape(-1, 3)),
        ("pre_normal_spring", second, after["floats"][:, 6:9]),
    ):
        difference = np.abs(candidate.astype(np.float64) - expected.astype(np.float64))
        mismatch = np.argwhere(candidate != expected)
        checked[name] = {
            "exact_elements": int(np.count_nonzero(candidate == expected)),
            "total_elements": int(expected.size),
            "max_abs_error": float(difference.max(initial=0)),
            "first_mismatch": mismatch[0].tolist() if len(mismatch) else None,
        }
    report = {
        "reference": "copied pinned FreeSurfer 8.2 first pial step source probe",
        "comparisons": checked,
        "seconds_including_jit": time.perf_counter() - start,
    }
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if args.require_exact and any(item["exact_elements"] != item["total_elements"] for item in checked.values()):
        raise SystemExit("signed gradient average differs from pinned source")


if __name__ == "__main__":
    main()
