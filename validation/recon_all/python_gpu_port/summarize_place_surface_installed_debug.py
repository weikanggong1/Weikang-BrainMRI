"""Compare installed-binary vertex debug gradients with frozen pial RAM meshes."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import nibabel as nib
import numpy as np


VERTICES = ((53114, 13), (55276, 13), (81282, 14))
GRADIENT = re.compile(r"dxyz=\[([^]]+)\]")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    report = {
        "reference": "installed FreeSurfer 8.2.0-1 LH pial first outer pass, n_averages=16, sigma=2, dt=0.5 at steps 13 and 14",
        "method": "GDB limits first outer iteration count and skips later outer passes/final cleanup; --debug-vertex prints pre-move aggregate gradient; RAM meshes are compared independently",
        "gradient_precision": "native %g text output, approximately six significant decimal digits, not exact float32 bits",
        "gradient_lines_per_vertex": "final N lines in each run contain exactly N pre-move gradients for accepted step N",
        "vertices": [],
    }
    for vertex, step in VERTICES:
        branch = json.loads((root / f"place_surface_official_step{step}_branch_report.json").read_text())
        source = next(row for row in branch["vertices"] if row["vertex"] == vertex)
        debug_dir = root / f"probe_installed_debug_v{vertex}_step{step}"
        debug = json.loads((debug_dir / "installed_ram_report.json").read_text())
        baseline_dir = root / f"probe_installed_ram_official_step{step}"
        baseline = json.loads((baseline_dir / "installed_ram_report.json").read_text())
        lines = [line for line in debug["debug_gradient_lines"]
                 if line.startswith(f"vno={vertex}  xyz=[") and "dxyz=[" in line]
        if len(lines) != step:
            raise AssertionError((vertex, step, "gradient lines", len(lines)))
        match = GRADIENT.search(lines[-1])
        if match is None:
            raise AssertionError(lines[-1])
        installed_gradient = np.array([float(value) for value in match.group(1).split(",")])
        source_gradient = np.array(source["source_gradient"])
        debug_mesh = debug_dir / "surf/lh.pial.ram"
        baseline_mesh = baseline_dir / "surf/lh.pial.ram"
        debug_xyz, debug_faces = nib.freesurfer.read_geometry(debug_mesh)
        baseline_xyz, baseline_faces = nib.freesurfer.read_geometry(baseline_mesh)
        same_mesh = bool(np.array_equal(debug_xyz, baseline_xyz) and np.array_equal(debug_faces, baseline_faces))
        same_trajectory = bool(debug["accepted_lines"] == baseline["accepted_lines"])
        if not same_mesh or not same_trajectory:
            raise AssertionError((vertex, step, "debug option changed installed checkpoint"))
        installed_delta = np.array(source["installed_accepted_delta_mm"])
        implied_gradient = installed_delta / 0.5
        report["vertices"].append({
            "vertex": vertex,
            "step": step,
            "installed_debug_gradient_line": lines[-1],
            "installed_debug_gradient": installed_gradient.tolist(),
            "source_probe_gradient": source_gradient.tolist(),
            "installed_minus_source_gradient": (installed_gradient - source_gradient).tolist(),
            "installed_accepted_delta_mm": installed_delta.tolist(),
            "installed_motion_gradient_estimate_from_delta": implied_gradient.tolist(),
            "installed_debug_vs_motion_gradient_max_abs": float(np.max(np.abs(installed_gradient - implied_gradient))),
            "source_candidate_delta_mm": (np.array(source["source_clipped_endpoint"]) - np.array(source["source_start_xyz"])).tolist(),
            "source_prediction_matches_native_checkpoint": source["source_prediction_matches_native_checkpoint"],
            "source_first_triangle_collision": source["source_first_triangle_collision"],
            "source_moved": source["source_moved"],
            "installed_moved": source["installed_moved"],
            "pre_step_source_installed_distance_mm": source["pre_step_source_installed_distance_mm"],
            "post_step_source_installed_distance_mm": source["post_step_source_installed_distance_mm"],
            "debug_mesh_matches_installed_RAM_baseline_all_vertices_and_faces": same_mesh,
            "debug_accepted_objective_lines_match_baseline": same_trajectory,
            "debug_mesh_sha256": digest(debug_mesh),
            "baseline_mesh_sha256": digest(baseline_mesh),
            "debug_capture_report": str(debug_dir / "installed_ram_report.json"),
            "source_branch_report": str(root / f"place_surface_official_step{step}_branch_report.json"),
        })
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"report": str(args.report), "vertices": report["vertices"]}, indent=2))


if __name__ == "__main__":
    main()
