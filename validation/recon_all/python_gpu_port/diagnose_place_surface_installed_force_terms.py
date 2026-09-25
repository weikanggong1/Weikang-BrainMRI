"""Replay source-order pial force terms on an installed pre-step RAM mesh."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

STATE = np.dtype([("floats", "<f4", 9), ("flags", "<i4", 3)])
STAGES = ("intensity", "surface_repulsion", "pre_normal_spring", "normal_spring", "curvature", "tangential_spring")


def vertex_average_trace(values, vertex, neighbors, valid, ripped):
    center = values[vertex]
    summed = center.astype(np.float64)
    count = 1
    rows = []
    for rank, neighbor in enumerate(neighbors[vertex]):
        if not valid[vertex, rank]:
            break
        neighbor = int(neighbor)
        if ripped[neighbor]:
            rows.append({"rank": rank, "vertex": neighbor, "ripped": True})
            continue
        other = values[neighbor]
        dot = np.float32(np.float32(center[0] * other[0] + center[1] * other[1]) + center[2] * other[2])
        included = bool(dot >= 0)
        if included:
            summed += other.astype(np.float64)
            count += 1
        rows.append({
            "rank": rank, "vertex": neighbor, "input_y": float(other[1]),
            "dot": float(dot), "included": included,
            "partial_sum_y": float(summed[1]), "count": count,
        })
    return np.float32(summed / count), rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=Path, required=True)
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--installed-previous", type=Path, required=True)
    parser.add_argument("--native-debug-report", type=Path, required=True)
    parser.add_argument("--vertex", type=int, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--trace-averaging", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(args.module))
    from fnit.recon_all.place_surface_border import compute_border_values_first_pass
    from fnit.recon_all.place_surface_curvature import quadratic_curvature, tangent_basis, two_ring_neighbors
    from fnit.recon_all.place_surface_geometry import surface_ras_to_voxel
    from fnit.recon_all.place_surface_gradient_average import average_signed_gradients
    from fnit.recon_all.place_surface_intensity import intensity_gradient
    from fnit.recon_all.place_surface_normals import initial_vertex_normals
    from fnit.recon_all.place_surface_repulsion import original_vertex_normals, surface_repulsion_gradient, vertex_buckets
    from fnit.recon_all.place_surface_rip import rip_outside_label
    from fnit.recon_all.place_surface_smoothing import average_marked_values, _ordered_neighbors
    from fnit.recon_all.place_surface_spring import spring_gradient
    from fnit.recon_all.place_surface_volume import prepare_placement_volume

    subject = args.subject
    brain = nib.load(subject / "mri/brain.finalsurfs.mgz")
    wm = np.asarray(nib.load(subject / "mri/wm.mgz").dataobj)
    segmentation = np.asarray(nib.load(subject / "mri/aseg.presurf.mgz").dataobj)
    original, faces, metadata = nib.freesurfer.read_geometry(subject / "surf/lh.white", read_metadata=True)
    original = original.astype(np.float32)
    current, installed_faces = nib.freesurfer.read_geometry(args.installed_previous)
    current = current.astype(np.float32)
    if not np.array_equal(faces, installed_faces):
        raise ValueError("installed ordered faces differ")
    ripped = np.asarray(rip_outside_label(
        len(original), nib.freesurfer.read_label(subject / "label/lh.cortex+hipamyg.label"),
    ), dtype=np.bool_)
    stats = dict(
        line.split()[:2] for line in (subject / "surf/autodet.gw.stats.lh.dat").read_text().splitlines()
        if len(line.split()) >= 2
    )
    volume, bright = prepare_placement_volume(
        np.asarray(brain.dataobj), wm, surface="pial", mid_gray=float(stats["MID_GRAY"]),
    )
    placement = volume.copy()
    placement[bright == 130] = 0
    affine = surface_ras_to_voxel(brain.header, metadata)
    thresholds = np.array([
        float(stats[f"pial_{name}"])
        for name in ("inside_hi", "border_hi", "border_low", "outside_low", "outside_hi")
    ])
    initial_normals = initial_vertex_normals(original, faces)
    border = compute_border_values_first_pass(
        volume, segmentation, original, initial_normals, original, ripped,
        np.full(len(original), -1.0, dtype=np.float32), affine, thresholds,
        hemisphere="lh", surface="pial",
    )
    values = average_marked_values(border[0], border[4], ripped, faces, 5)
    fixed_normals = original_vertex_normals(original, faces)
    neighbor_indices, neighbor_valid, _ = _ordered_neighbors(faces, len(original))
    ordered = (neighbor_indices, neighbor_valid)
    two_offsets, two_candidates = two_ring_neighbors(faces, len(original), ordered_neighbors=ordered)
    prefix = args.source_run / f"lh.gradient.step{args.step:02d}"
    cropped = np.fromfile(str(prefix) + ".clear.cropped", dtype="<i4")

    normals = initial_vertex_normals(current, faces)
    intensity = intensity_gradient(
        placement, current, normals, ripped, values, border[5], affine,
        brain.header.get_zooms()[:3], weight=0.2, sigma_global=2.0,
    )
    offsets, candidates = vertex_buckets(current, original, ripped)
    repulsion = surface_repulsion_gradient(
        current, normals, original, fixed_normals, ripped, offsets, candidates,
        weight=5.0, cropped=cropped,
    )
    with_repulsion = np.float32(intensity + repulsion)
    averaged = average_signed_gradients(with_repulsion, faces, ripped, 16, ordered_neighbors=ordered)
    normal = spring_gradient(current, normals, faces, ripped, weight=0.3, direction="normal", ordered_neighbors=ordered)
    with_normal = np.float32(averaged + normal)
    basis = tangent_basis(normals)
    scalar = quadratic_curvature(current, normals, basis, ripped, two_offsets, two_candidates)
    with_curvature = np.float32(with_normal + np.float32(scalar[:, None] * normals))
    tangent = spring_gradient(current, normals, faces, ripped, weight=0.3, direction="tangent", ordered_neighbors=ordered)
    final = np.float32(with_curvature + tangent)

    vertex = args.vertex
    calculated = (intensity, with_repulsion, averaged, with_normal, with_curvature, final)
    source = {
        stage: np.memmap(Path(str(prefix) + "." + stage), dtype=STATE, mode="r", shape=(len(original),))["floats"][vertex, 6:9].tolist()
        for stage in STAGES
    }
    debug = json.loads(args.native_debug_report.read_text())
    installed = next(row for row in debug["vertices"] if row["vertex"] == vertex and row["step"] == args.step)
    average_trace = None
    if args.trace_averaging:
        source_preaverage = np.memmap(
            Path(str(prefix) + ".surface_repulsion"), dtype=STATE, mode="r", shape=(len(original),),
        )["floats"][:, 6:9].copy()
        source_current = source_preaverage
        installed_current = with_repulsion
        passes = []
        for iteration in range(1, 17):
            source_output, source_rows = vertex_average_trace(
                source_current, vertex, neighbor_indices, neighbor_valid, ripped,
            )
            installed_output, installed_rows = vertex_average_trace(
                installed_current, vertex, neighbor_indices, neighbor_valid, ripped,
            )
            next_source = average_signed_gradients(
                source_current, faces, ripped, 1, ordered_neighbors=ordered,
            )
            next_installed = average_signed_gradients(
                installed_current, faces, ripped, 1, ordered_neighbors=ordered,
            )
            if not np.array_equal(source_output, next_source[vertex]) or not np.array_equal(installed_output, next_installed[vertex]):
                raise AssertionError((iteration, "source-order trace differs from Numba average"))
            passes.append({
                "pass": iteration,
                "source_input_y": float(source_current[vertex, 1]),
                "installed_input_y": float(installed_current[vertex, 1]),
                "source_output_y": float(next_source[vertex, 1]),
                "installed_output_y": float(next_installed[vertex, 1]),
                "neighbor_contributions": [
                    {"vertex": a["vertex"], "rank": a["rank"], "source": a, "installed": b}
                    for a, b in zip(source_rows, installed_rows)
                ],
            })
            source_current, installed_current = next_source, next_installed
        source_expected = np.memmap(
            Path(str(prefix) + ".pre_normal_spring"), dtype=STATE, mode="r", shape=(len(original),),
        )["floats"][:, 6:9]
        average_trace = {
            "source_final_matches_native_probe_all_components": bool(np.array_equal(source_current, source_expected)),
            "installed_final_matches_direct_16_pass_python_all_components": bool(np.array_equal(installed_current, averaged)),
            "first_target_inclusion_difference_pass": next((
                row["pass"] for row in passes
                if any(a["source"].get("included") != a["installed"].get("included")
                       for a in row["neighbor_contributions"])
            ), None),
            "first_target_output_y_sign_difference_pass": next((
                row["pass"] for row in passes
                if np.signbit(row["source_output_y"]) != np.signbit(row["installed_output_y"])
            ), None),
            "passes": passes,
        }

    report = {
        "reference": "source-order Python force terms on installed-binary RAM pre-step geometry; copied-source cropped counters substituted because installed counters unavailable",
        "step": args.step,
        "vertex": vertex,
        "installed_pre_step_mesh": str(args.installed_previous),
        "source_cropped_counter_at_vertex": int(cropped[vertex]),
        "installed_cropped_counter_available": False,
        "source_probe_pre_step_normal": np.memmap(Path(str(prefix) + ".clear"), dtype=STATE, mode="r", shape=(len(original),))["floats"][vertex, 3:6].tolist(),
        "python_installed_mesh_pre_step_normal": normals[vertex].tolist(),
        "installed_native_debug_pre_step_normal_text": installed["installed_debug_gradient_line"].split("nxyz=[", 1)[1].split("]", 1)[0],
        "source_probe_stage_gradients": source,
        "python_on_installed_mesh_stage_gradients": {stage: vector[vertex].tolist() for stage, vector in zip(STAGES, calculated)},
        "python_on_installed_mesh_component_increments": {
            "intensity": intensity[vertex].tolist(),
            "surface_repulsion": repulsion[vertex].tolist(),
            "signed_average_minus_preaverage": (averaged[vertex].astype(np.float64) - with_repulsion[vertex]).tolist(),
            "normal_spring": normal[vertex].tolist(),
            "curvature": np.float32(scalar[vertex] * normals[vertex]).tolist(),
            "tangential_spring": tangent[vertex].tolist(),
        },
        "installed_native_debug_aggregate_gradient_text": installed["installed_debug_gradient"],
        "signed_average_trace": average_trace,
        "python_on_installed_mesh_minus_installed_native_aggregate_gradient": (final[vertex].astype(np.float64) - np.array(installed["installed_debug_gradient"])).tolist(),
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
