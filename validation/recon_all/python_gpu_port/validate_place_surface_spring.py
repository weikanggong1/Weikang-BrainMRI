"""Compare first-step spring gradients with a copied-source pial probe."""

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
    from fnit.recon_all.place_surface_spring import spring_gradient

    _, faces = nib.freesurfer.read_geometry(args.surface)
    stages = {
        name: np.fromfile(args.probe / f"lh.gradient.{name}", dtype=STATE)
        for name in ("pre_normal_spring", "normal_spring", "pre_tangential_spring", "tangential_spring")
    }
    report = {"reference": "copied pinned FreeSurfer 8.2 source first pial optimizer step", "comparisons": {}}
    for before, after, direction in (
        ("pre_normal_spring", "normal_spring", "normal"),
        ("pre_tangential_spring", "tangential_spring", "tangent"),
    ):
        state = stages[before]
        start = time.perf_counter()
        delta = spring_gradient(
            state["floats"][:, :3], state["floats"][:, 3:6], faces,
            state["flags"][:, 0], weight=0.3, direction=direction,
            border=state["flags"][:, 1], negative=state["flags"][:, 2],
        )
        candidate = np.float32(state["floats"][:, 6:9] + delta)
        expected = stages[after]["floats"][:, 6:9]
        difference = np.abs(candidate.astype(np.float64) - expected.astype(np.float64))
        report["comparisons"][direction] = {
            "vertices": len(candidate),
            "exact_elements": int(np.count_nonzero(candidate == expected)),
            "total_elements": int(expected.size),
            "max_abs_error": float(difference.max(initial=0)),
            "seconds_including_jit": time.perf_counter() - start,
        }
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if args.require_exact and any(
        item["exact_elements"] != item["total_elements"]
        for item in report["comparisons"].values()
    ):
        raise SystemExit("spring gradient differs from pinned source")


if __name__ == "__main__":
    main()
