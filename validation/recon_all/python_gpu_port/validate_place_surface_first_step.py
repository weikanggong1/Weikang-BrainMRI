"""Validate the first collision-limited LH pial step against copied source and installed RAM diagnostics."""

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
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--source-step", type=Path, required=True)
    parser.add_argument("--installed-step", type=Path, required=True)
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--require-exact-source", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(args.module))
    from fnit.recon_all.place_surface_collision import asynchronous_first_step
    from fnit.recon_all.place_surface_step import unconstrained_step

    state = np.fromfile(args.probe / "lh.gradient.tangential_spring", dtype=STATE)
    initial = state["floats"][:, :3]
    gradient = state["floats"][:, 6:9]
    ripped = state["flags"][:, 0]
    source, source_faces = nib.freesurfer.read_geometry(args.source_step)
    installed, installed_faces = nib.freesurfer.read_geometry(args.installed_step)
    source = source.astype(np.float32)
    installed = installed.astype(np.float32)
    proposal = unconstrained_step(initial, gradient, ripped)
    start = time.perf_counter()
    actual, order = asynchronous_first_step(initial, source_faces, proposal, ripped)
    seconds = time.perf_counter() - start
    source_exact = np.all(actual == source, axis=1)
    installed_exact = np.all(actual == installed, axis=1)
    installed_mm = np.linalg.norm(actual.astype(np.float64) - installed.astype(np.float64), axis=1)
    source_mismatch = np.flatnonzero(~source_exact)
    report = {
        "references": {
            "source": "copied pinned FreeSurfer 8.2 first-step source probe",
            "installed": "installed FreeSurfer 8.2 binary with RAM-only one-iteration GDB diagnostic",
        },
        "source": {
            "exact_vertices": int(source_exact.sum()),
            "total_vertices": len(source),
            "ordered_faces_exact": bool(np.array_equal(source_faces, installed_faces)),
            "max_vertex_distance_mm": float(np.linalg.norm(actual.astype(np.float64) - source.astype(np.float64), axis=1).max(initial=0)),
            "first_mismatched_vertices": source_mismatch[:20].tolist(),
            "collision_blocked_vertices": int(np.count_nonzero(
                np.any(proposal != initial, axis=1) & np.all(source == initial, axis=1)
            )),
        },
        "installed": {
            "exact_vertices": int(installed_exact.sum()),
            "total_vertices": len(installed),
            "p99_vertex_distance_mm": float(np.percentile(installed_mm, 99)),
            "max_vertex_distance_mm": float(installed_mm.max(initial=0)),
        },
        "processed_vertices": len(order),
        "seconds_including_first_jit_excluding_io": seconds,
    }
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if args.require_exact_source and not np.all(source_exact):
        raise SystemExit("Python first optimizer step differs from copied pinned source")


if __name__ == "__main__":
    main()
