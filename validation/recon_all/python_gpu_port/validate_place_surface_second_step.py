"""Compare the second LH pial collision-limited step with pinned source."""

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
    parser.add_argument("--first-step", type=Path, required=True)
    parser.add_argument("--previous-first-step", type=Path, required=True)
    parser.add_argument("--second-step", type=Path, required=True)
    parser.add_argument("--gradient", type=Path, required=True)
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--output-step", type=Path)
    parser.add_argument("--require-exact", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(args.module))
    from fnit.recon_all.place_surface_collision import asynchronous_first_step
    from fnit.recon_all.place_surface_step import unconstrained_step

    first, faces = nib.freesurfer.read_geometry(args.first_step)
    previous, previous_faces = nib.freesurfer.read_geometry(args.previous_first_step)
    source, second_faces = nib.freesurfer.read_geometry(args.second_step)
    first, previous, source = (item.astype(np.float32) for item in (first, previous, source))
    gradient_state = np.fromfile(args.gradient, dtype=STATE)
    ripped = gradient_state["flags"][:, 0]
    if len(gradient_state) != len(first):
        raise SystemExit("second gradient has different vertex count")
    proposal = unconstrained_step(first, gradient_state["floats"][:, 6:9], ripped)
    start = time.perf_counter()
    actual, order = asynchronous_first_step(first, faces, proposal, ripped)
    seconds = time.perf_counter() - start
    exact = np.all(actual == source, axis=1)
    distance = np.linalg.norm(actual.astype(np.float64) - source.astype(np.float64), axis=1)
    report = {
        "reference": "pinned FreeSurfer 8.2 copied-source second LH pial mesh snapshot",
        "previous_first_step_exact": bool(np.array_equal(first, previous)),
        "gradient_start_exact": bool(np.array_equal(first, gradient_state["floats"][:, :3])),
        "ordered_faces_exact": bool(np.array_equal(faces, previous_faces) and np.array_equal(faces, second_faces)),
        "exact_vertices": int(np.count_nonzero(exact)),
        "total_vertices": len(source),
        "max_distance_mm": float(distance.max(initial=0)),
        "first_mismatched_vertices": np.flatnonzero(~exact)[:20].tolist(),
        "source_collision_blocked_vertices": int(np.count_nonzero(
            np.any(proposal != first, axis=1) & np.all(source == first, axis=1)
        )),
        "processed_vertices": len(order),
        "python_step_seconds_excluding_io": seconds,
    }
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if args.require_exact and not (
        report["previous_first_step_exact"]
        and report["gradient_start_exact"]
        and report["ordered_faces_exact"]
        and np.all(exact)
    ):
        raise SystemExit("second pial mesh step differs from pinned source")
    if args.output_step:
        np.save(args.output_step, actual)


if __name__ == "__main__":
    main()
