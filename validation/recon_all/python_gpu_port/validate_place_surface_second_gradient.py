"""Compare second LH pial optimizer gradient against pinned source snapshots."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np


STATE = np.dtype([("floats", "<f4", 9), ("flags", "<i4", 3)])


def _compare(actual: np.ndarray, expected: np.ndarray) -> dict:
    delta = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    return {
        "exact_elements": int(np.count_nonzero(actual == expected)),
        "total_elements": int(expected.size),
        "max_abs_error": float(delta.max(initial=0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--first-step", type=Path, required=True)
    parser.add_argument("--first-gradient", type=Path, required=True)
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--require-exact", action="store_true")
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
    from fnit.recon_all.place_surface_smoothing import average_marked_values
    from fnit.recon_all.place_surface_spring import spring_gradient
    from fnit.recon_all.place_surface_step import unconstrained_step
    from fnit.recon_all.place_surface_volume import prepare_placement_volume

    subject = args.subject
    brain = nib.load(subject / "mri/brain.finalsurfs.mgz")
    wm = np.asarray(nib.load(subject / "mri/wm.mgz").dataobj)
    segmentation = np.asarray(nib.load(subject / "mri/aseg.presurf.mgz").dataobj)
    original, faces, metadata = nib.freesurfer.read_geometry(subject / "surf/lh.white", read_metadata=True)
    original = original.astype(np.float32)
    current, step_faces = nib.freesurfer.read_geometry(args.first_step)
    current = current.astype(np.float32)
    if not np.array_equal(faces, step_faces):
        raise SystemExit("ordered faces changed after first source step")
    ripped = rip_outside_label(len(original), nib.freesurfer.read_label(subject / "label/lh.cortex+hipamyg.label"))
    stats = dict(
        line.split()[:2]
        for line in (subject / "surf/autodet.gw.stats.lh.dat").read_text().splitlines()
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
    normals = initial_vertex_normals(current, faces)
    intensity = intensity_gradient(
        placement, current, normals, ripped, values, border[5], affine,
        brain.header.get_zooms()[:3], weight=0.2, sigma_global=2.0,
    )
    first_gradient = np.fromfile(args.first_gradient, dtype=STATE)["floats"][:, 6:9]
    proposal = unconstrained_step(original, first_gradient, ripped)
    cropped = np.any(proposal != original, axis=1) & np.all(current == original, axis=1)
    original_normals = original_vertex_normals(original, faces)
    offsets, candidates = vertex_buckets(current, original, ripped)
    repulsion = surface_repulsion_gradient(
        current, normals, original, original_normals, ripped, offsets, candidates,
        weight=5.0, cropped=cropped,
    )
    with_repulsion = np.float32(intensity + repulsion)
    averaged = average_signed_gradients(with_repulsion, faces, ripped, 2)
    normal = spring_gradient(current, normals, faces, ripped, weight=0.3, direction="normal")
    with_normal = np.float32(averaged + normal)
    basis = tangent_basis(normals)
    two_offsets, two_candidates = two_ring_neighbors(faces, len(current))
    scalar = quadratic_curvature(current, normals, basis, ripped, two_offsets, two_candidates)
    with_curvature = np.float32(with_normal + np.float32(scalar[:, None] * normals))
    tangent = spring_gradient(current, normals, faces, ripped, weight=0.3, direction="tangent")
    final = np.float32(with_curvature + tangent)

    clear = np.fromfile(args.probe / "lh.gradient.clear", dtype=STATE)
    ripped_mask = np.asarray(ripped, dtype=np.bool_)
    checked = {
        "start_xyz": _compare(current, clear["floats"][:, :3]),
        "normals_active": _compare(normals[~ripped_mask], clear["floats"][~ripped_mask, 3:6]),
        "normals_ripped_diagnostic_only": _compare(normals[ripped_mask], clear["floats"][ripped_mask, 3:6]),
        "ripped": _compare(ripped, clear["flags"][:, 0]),
    }
    for stage, value in (
        ("intensity", intensity),
        ("surface_repulsion", with_repulsion),
        ("pre_normal_spring", averaged),
        ("normal_spring", with_normal),
        ("curvature", with_curvature),
        ("tangential_spring", final),
    ):
        source = np.fromfile(args.probe / f"lh.gradient.{stage}", dtype=STATE)
        checked[stage] = _compare(value, source["floats"][:, 6:9])
    report = {
        "reference": "pinned FreeSurfer 8.2 copied-source second LH pial gradient step",
        "inputs": "original MRI, WM, aseg, LH white, label, frozen autodet stats; source-validated first-step gradient and mesh",
        "collision_cropped_vertices_from_first_step": int(np.count_nonzero(cropped)),
        "comparisons": checked,
    }
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    required = {name: value for name, value in checked.items() if name != "normals_ripped_diagnostic_only"}
    if args.require_exact and any(x["exact_elements"] != x["total_elements"] for x in required.values()):
        raise SystemExit("second pial gradient differs from pinned source")


if __name__ == "__main__":
    main()
