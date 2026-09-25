"""Profile one already validated LH pial collision step with fixed inputs."""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import os
import pstats
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np


STATE = np.dtype([("floats", "<f4", 9), ("flags", "<i4", 3)])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--surface", type=Path, required=True)
    parser.add_argument("--gradient", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(args.module))
    from fnit.recon_all.place_surface_collision import asynchronous_first_step
    from fnit.recon_all.place_surface_step import unconstrained_step

    previous = np.load(args.previous).astype(np.float32, copy=False)
    _, faces = nib.freesurfer.read_geometry(args.surface)
    source, source_faces = nib.freesurfer.read_geometry(args.source)
    source = source.astype(np.float32)
    state = np.fromfile(args.gradient, dtype=STATE)
    gradient = state["floats"][:, 6:9]
    ripped = state["flags"][:, 0]
    if not np.array_equal(previous, state["floats"][:, :3]):
        raise SystemExit("saved Python second step differs from third gradient start")
    if not np.array_equal(faces, source_faces):
        raise SystemExit("ordered faces differ")
    proposal = unconstrained_step(previous, gradient, ripped)
    report = {
        "reference": "independent Python LH pial second step to source third step",
        "load_before": os.getloadavg(),
        "fast": args.fast,
        "repeats": [],
    }
    for repeat in range(args.repeats):
        profiler = cProfile.Profile()
        started = time.perf_counter()
        profiler.enable()
        actual, order = asynchronous_first_step(previous, faces, proposal, ripped, fast=args.fast)
        profiler.disable()
        seconds = time.perf_counter() - started
        output = io.StringIO()
        pstats.Stats(profiler, stream=output).sort_stats("cumulative").print_stats(25)
        exact = np.all(actual == source, axis=1)
        distance = np.linalg.norm(actual.astype(np.float64) - source.astype(np.float64), axis=1)
        entry = {
            "repeat": repeat,
            "wall_seconds": seconds,
            "load_after": os.getloadavg(),
            "ordered_coordinates_exact": bool(np.all(exact)),
            "exact_vertices": int(exact.sum()),
            "total_vertices": len(exact),
            "max_distance_mm": float(distance.max(initial=0)),
            "first_mismatched_vertices": np.flatnonzero(~exact)[:10].tolist(),
            "processed_vertices": len(order),
            "profile_top_cumulative": output.getvalue(),
        }
        report["repeats"].append(entry)
        if not entry["ordered_coordinates_exact"]:
            break
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if len(report["repeats"]) != args.repeats or not all(
        entry["ordered_coordinates_exact"] for entry in report["repeats"]
    ):
        raise SystemExit("profiled collision step differs from pinned source")


if __name__ == "__main__":
    main()
