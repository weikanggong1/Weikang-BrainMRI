"""Compare ordered native pial mesh coordinates at a frozen checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--installed-ram", type=Path, required=True)
    parser.add_argument("--source-snapshot", type=Path, required=True)
    parser.add_argument("--installed-ram-log", type=Path, required=True)
    parser.add_argument("--source-log", type=Path, required=True)
    parser.add_argument("--source-step", type=int, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    installed, installed_faces = nib.freesurfer.read_geometry(args.installed_ram)
    source, source_faces = nib.freesurfer.read_geometry(args.source_snapshot)
    if installed.shape != source.shape:
        raise ValueError("vertex count or shape mismatch")
    source_log = args.source_log.read_text()
    installed_log = args.installed_ram_log.read_text()
    if "INSTALLED_RAM_AFTER_FIRST_OUTER" not in installed_log:
        raise ValueError("RAM run did not stop at the first outer pass")
    if f"INSTALLED_RAM_NITER={args.source_step}" not in installed_log and args.source_step != 26:
        raise ValueError("RAM diagnostic did not use expected iteration limit")
    exact = np.all(installed == source, axis=1)
    distance = np.linalg.norm(installed.astype(np.float64) - source.astype(np.float64), axis=1)
    first = int(np.flatnonzero(~exact)[0]) if not np.all(exact) else None
    installed_lines = [line for line in installed_log.splitlines() if line.startswith(f"{args.source_step:03d}: dt:")]
    source_lines = [line for line in source_log.splitlines() if line.startswith(f"{args.source_step:03d}: dt:")]
    report = {
        "reference_class": "installed FreeSurfer 8.2.0-1 RAM first-outer-pass snapshot versus copied-source instrumented snapshot; normal CLI final surface is separate",
        "source_step": args.source_step,
        "installed_ram_mesh": str(args.installed_ram),
        "source_snapshot_mesh": str(args.source_snapshot),
        "ordered_faces_equal": bool(np.array_equal(installed_faces, source_faces)),
        "ordered_face_count": int(len(source_faces)),
        "vertex_count": int(len(source)),
        "exact_vertices": int(exact.sum()),
        "first_different_vertex": first,
        "first_different_installed_xyz": installed[first].tolist() if first is not None else None,
        "first_different_source_xyz": source[first].tolist() if first is not None else None,
        "first_difference_mm": float(distance[first]) if first is not None else 0.0,
        "median_mm": float(np.median(distance)),
        "p99_mm": float(np.percentile(distance, 99)),
        "max_mm": float(distance.max(initial=0)),
        "largest_distance_vertices": [
            {
                "vertex": int(vertex), "distance_mm": float(distance[vertex]),
                "installed_xyz": installed[vertex].tolist(),
                "source_xyz": source[vertex].tolist(),
            }
            for vertex in np.argsort(distance)[-10:][::-1]
        ],
        "vertices_over_tolerance_mm": {
            f"{threshold:g}": int(np.count_nonzero(distance > threshold))
            for threshold in (1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
        },
        "installed_objective_line": installed_lines[-1] if installed_lines else "",
        "source_objective_line": source_lines[-1] if source_lines else "",
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
