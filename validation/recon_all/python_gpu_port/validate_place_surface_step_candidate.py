"""Measure collision-free first pial step against a pinned-source snapshot."""

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
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.module))
    from fnit.recon_all.place_surface_step import unconstrained_step

    before = np.fromfile(args.probe / "lh.gradient.tangential_spring", dtype=STATE)
    expected, expected_faces = nib.freesurfer.read_geometry(args.source_step)
    expected = expected.astype(np.float32)
    start = time.perf_counter()
    candidate = unconstrained_step(
        before["floats"][:, :3], before["floats"][:, 6:9], before["flags"][:, 0],
        dt=0.5, max_mm=0.3,
    )
    discrepancy = np.linalg.norm(candidate.astype(np.float64) - expected.astype(np.float64), axis=1)
    exact_vertex = np.all(candidate == expected, axis=1)
    initial = before["floats"][:, :3]
    mismatched = np.flatnonzero(~exact_vertex)
    source_unmoved = np.all(expected == initial, axis=1)
    report = {
        "reference": "copied pinned FreeSurfer 8.2 first pial step source probe",
        "candidate": "gradient times dt, clipped to 0.3 mm, with no collision test",
        "vertices": len(candidate),
        "ordered_faces": len(expected_faces),
        "exact_vertices": int(np.count_nonzero(exact_vertex)),
        "exact_coordinate_components": int(np.count_nonzero(candidate == expected)),
        "max_vertex_distance_mm": float(discrepancy.max(initial=0)),
        "p99_vertex_distance_mm": float(np.percentile(discrepancy, 99)),
        "above_1e_5_mm": int(np.count_nonzero(discrepancy > 1e-5)),
        "mismatched_source_unmoved": int(np.count_nonzero(source_unmoved[mismatched])),
        "mismatched_source_moved": int(np.count_nonzero(~source_unmoved[mismatched])),
        "first_mismatched_vertices": mismatched[:20].tolist(),
        "seconds_including_jit": time.perf_counter() - start,
    }
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
