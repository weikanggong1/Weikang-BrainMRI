"""Consolidate the bounded installed-versus-source LH pial first-difference evidence."""

from __future__ import annotations

import argparse
import json
import re
import struct
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

STATE = np.dtype([("floats", "<f4", 9), ("flags", "<i4", 3)])


def decode_double(line: str) -> float:
    match = re.search(r"v2_int64 = \{(0x[0-9a-f]+)", line)
    if match is None:
        raise ValueError(line)
    return struct.unpack("<d", struct.pack("<Q", int(match.group(1), 16)))[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.module))
    from fnit.recon_all.place_surface_step import unconstrained_step_with_offsets

    root = args.root
    ring = json.loads((root / "place_surface_early_step_16ring_report.json").read_text())
    branch = json.loads((root / "place_surface_installed_debug_first_branch_report.json").read_text())
    force = json.loads((root / "place_surface_installed_step14_force_terms_report.json").read_text())
    exact = json.loads((root / "probe_installed_exact_v80852_step1/exact_gradient_report.json").read_text())
    if exact["capture_count"] != 1:
        raise ValueError("expected one native exact-gradient capture")
    lines = exact["capture_lines"][0]
    if lines[0] != "EXACT_GRADIENT_VNO=80852":
        raise ValueError(lines)
    stack = re.search(r":\s*(0x[0-9a-f]+)$", lines[3])
    if stack is None:
        raise ValueError(lines[3])
    installed_gradient = np.array([
        decode_double(lines[1]), decode_double(lines[2]),
        struct.unpack("<d", struct.pack("<Q", int(stack.group(1), 16)))[0],
    ], dtype=np.float32)
    prefix = root / "probe_official_pass0_run/lh.gradient.step01"
    clear = np.fromfile(str(prefix) + ".clear", dtype=STATE)
    final = np.fromfile(str(prefix) + ".tangential_spring", dtype=STATE)
    after = np.fromfile(str(prefix) + ".after_collision", dtype=STATE)
    vertex = 80852
    proposal, offsets = unconstrained_step_with_offsets(
        clear["floats"][:, :3], final["floats"][:, 6:9],
        clear["flags"][:, 0].astype(bool), dt=0.5,
    )
    installed, faces = nib.freesurfer.read_geometry(
        root / "probe_installed_ram_official_step1_exact/surf/lh.pial.ram",
    )
    installed_debug, debug_faces = nib.freesurfer.read_geometry(exact["mesh"])
    source, source_faces = nib.freesurfer.read_geometry(
        root / "probe_official_pass0_run/surf/lh.probe0001",
    )
    if not np.array_equal(faces, source_faces):
        raise ValueError("source/installed ordered faces differ")
    if not np.array_equal(installed, installed_debug) or not np.array_equal(faces, debug_faces):
        raise ValueError("exact-gradient GDB capture changed installed output mesh")
    if not np.array_equal(proposal[vertex], after["floats"][vertex, :3]):
        raise ValueError("source candidate was modified by collision")
    if not np.array_equal(after["floats"][vertex, :3], source[vertex]):
        raise ValueError("source checkpoint differs from saved mesh")
    if not np.array_equal(installed_gradient, final["floats"][vertex, 6:9]):
        raise ValueError("native exact first-step gradient differs from source")

    trace = force["signed_average_trace"]
    first_pass = trace["passes"][0]
    native_debug_log = (
        root / "probe_installed_debug_v81282_step14/installed_ram.log"
    ).read_text().splitlines()
    selected = max(index for index, line in enumerate(native_debug_log)
                   if line.startswith("vno=81282  xyz=["))
    prior_step = max(index for index in range(selected)
                     if native_debug_log[index].startswith("013: dt:"))
    before = native_debug_log[prior_step:selected]
    native_force_lines = [line for line in before if
                          line.startswith("v 81282 intensity term:")
                          or line.startswith("before averaging dot =")
                          or line.startswith("v 81282 curvature term:")]
    if len(native_force_lines) != 3:
        raise ValueError(native_force_lines)
    installed_step14 = next(row for row in branch["vertices"]
                            if row["vertex"] == 81282 and row["step"] == 14)
    ring_step1 = next(row for row in ring["checkpoints"] if row["step"] == 1)
    ring_step13 = next(row for row in ring["checkpoints"] if row["step"] == 13)
    report = {
        "reference": "same frozen LH pial input; installed FreeSurfer 8.2.0-1 RAM checkpoints versus GCC11.2 copied-source diagnostics; first pass n_averages=16, sigma=2, dt=0.5",
        "reference_classes_separate": True,
        "selected_first_step_vertex": vertex,
        "selected_vertex_ring_distance_from_step14_81282": 10,
        "step1_81282_center_difference_mm": ring_step1["center_distance_mm"],
        "step1_16_ring_vertices": ring["support_vertices"],
        "step1_16_ring_differing_vertices": ring["support_vertices"] - ring_step1["support_exact_vertices"],
        "step1_16_ring_max_difference_mm": ring_step1["support_max_mm"],
        "step13_16_ring_max_difference_mm": ring_step13["support_max_mm"],
        "step13_16_ring_vertices_above_1e-4_mm": ring_step13["support_above_1e-4_mm"],
        "step1_selected_installed_gradient_bits_equal_copied_source": True,
        "step1_selected_installed_gradient": installed_gradient.tolist(),
        "step1_selected_source_gradient": final["floats"][vertex, 6:9].tolist(),
        "step1_selected_source_candidate_equals_source_after_collision": True,
        "step1_selected_source_candidate_xyz": proposal[vertex].tolist(),
        "step1_selected_installed_accepted_xyz": installed[vertex].tolist(),
        "step1_selected_installed_minus_source_xyz_mm": (
            installed[vertex].astype(np.float64) - source[vertex].astype(np.float64)
        ).tolist(),
        "step1_selected_installed_debug_mesh_matches_RAM_baseline": True,
        "step1_unresolved_operator": "exact matched pre-move gradient; 1-ULP y difference occurs in clipped displacement, close-neighbor projection, collision/acceptance, or coordinate update; native intermediate odxyz not captured",
        "step14_vertex": 81282,
        "step14_pre_step_center_difference_mm": installed_step14["pre_step_source_installed_distance_mm"],
        "step14_native_force_lines_rounded": native_force_lines,
        "step14_python_installed_mesh_aggregate_minus_native_text_max_abs": max(
            abs(value) for value in force["python_on_installed_mesh_minus_installed_native_aggregate_gradient"]
        ),
        "step14_first_signed_average_gate_difference_pass": trace["first_target_inclusion_difference_pass"],
        "step14_first_signed_average_output_y_sign_difference_pass": trace["first_target_output_y_sign_difference_pass"],
        "step14_first_pass_source_output_y": first_pass["source_output_y"],
        "step14_first_pass_installed_output_y": first_pass["installed_output_y"],
        "step14_first_gate_difference": first_pass["neighbor_contributions"][0],
        "step14_first_positive_installed_only_neighbor": first_pass["neighbor_contributions"][3],
        "step14_last_pass_source_output_y": trace["passes"][-1]["source_output_y"],
        "step14_last_pass_installed_output_y": trace["passes"][-1]["installed_output_y"],
        "installed_cropped_counters_captured": False,
        "installed_exact_force_terms_captured": False,
        "acceptance": "installed-binary final surface parity not achieved by copied source or independent Python; native first-step intermediate displacement operator unresolved",
        "source_evidence": {
            "early_16_ring": str(root / "place_surface_early_step_16ring_report.json"),
            "exact_installed_gradient": str(root / "probe_installed_exact_v80852_step1/exact_gradient_report.json"),
            "installed_debug_branch": str(root / "place_surface_installed_debug_first_branch_report.json"),
            "force_terms_and_16_passes": str(root / "place_surface_installed_step14_force_terms_report.json"),
        },
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items()
                      if key not in ("step14_first_gate_difference", "step14_first_positive_installed_only_neighbor", "source_evidence")}, indent=2))


if __name__ == "__main__":
    main()
