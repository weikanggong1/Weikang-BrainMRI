"""Replay two installed pial steps from original MRI and white surface."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np


STATE = np.dtype([("floats", "<f4", 9), ("flags", "<i4", 3)])
STAGES = ("intensity", "surface_repulsion", "pre_normal_spring", "normal_spring", "curvature", "tangential_spring")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(actual: np.ndarray, reference: np.ndarray) -> dict:
    a, b = np.asarray(actual, dtype=np.float32), np.asarray(reference, dtype=np.float32)
    if a.shape != b.shape:
        raise ValueError("shape differs")
    same = a.view(np.uint32) == b.view(np.uint32)
    bad = ~np.all(same, axis=1)
    return {
        "exact_components": int(np.count_nonzero(same)),
        "total_components": int(a.size),
        "exact_vertices": int(np.count_nonzero(~bad)),
        "max_abs_error": float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max(initial=0)),
        "first_different_vertices": np.flatnonzero(bad)[:12].tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--hemisphere", choices=("lh", "rh"), required=True)
    parser.add_argument("--installed-binary", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--require-exact", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(args.module))
    from fnit.recon_all.place_surface_border import compute_border_values_first_pass
    from fnit.recon_all.place_surface_collision import asynchronous_first_step
    from fnit.recon_all.place_surface_curvature import quadratic_curvature, tangent_basis, two_ring_neighbors
    from fnit.recon_all.place_surface_geometry import surface_ras_to_voxel
    from fnit.recon_all.place_surface_gradient_average import average_signed_gradients
    from fnit.recon_all.place_surface_intensity import intensity_gradient
    from fnit.recon_all.place_surface_normals import initial_vertex_normals
    from fnit.recon_all.place_surface_repulsion import original_vertex_normals, surface_repulsion_gradient, vertex_buckets
    from fnit.recon_all.place_surface_rip import rip_outside_label
    from fnit.recon_all.place_surface_smoothing import average_marked_values, _ordered_neighbors
    from fnit.recon_all.place_surface_spring import spring_gradient
    from fnit.recon_all.place_surface_step import unconstrained_step_with_offsets
    from fnit.recon_all.place_surface_volume import prepare_placement_volume

    hemi = args.hemisphere
    probe = args.root / "probe_official_pass0_run"
    ram_dirs = (
        "probe_installed_ram_official_step1_exact" if hemi == "lh" else "probe_installed_rh_ram_official_step1_exact",
        "probe_installed_ram_official_step2_exact" if hemi == "lh" else "probe_installed_rh_ram_official_step2_exact",
    )
    ram_paths = [args.root / directory / "surf" / f"{hemi}.pial.ram" for directory in ram_dirs]
    white_path = args.subject / "surf" / f"{hemi}.white"
    label_path = args.subject / "label" / f"{hemi}.cortex+hipamyg.label"
    stats_path = args.subject / "surf" / f"autodet.gw.stats.{hemi}.dat"
    original, faces, metadata = nib.freesurfer.read_geometry(white_path, read_metadata=True)
    original = original.astype(np.float32)
    native = []
    for path in ram_paths:
        xyz, native_faces = nib.freesurfer.read_geometry(path)
        if not np.array_equal(faces, native_faces):
            raise ValueError("installed ordered faces differ from input")
        native.append(xyz.astype(np.float32))
    ripped = np.asarray(rip_outside_label(
        len(original), nib.freesurfer.read_label(label_path),
    ), dtype=np.bool_)
    brain_path = args.subject / "mri/brain.finalsurfs.mgz"
    wm_path = args.subject / "mri/wm.mgz"
    aseg_path = args.subject / "mri/aseg.presurf.mgz"
    brain = nib.load(brain_path)
    wm = np.asarray(nib.load(wm_path).dataobj)
    aseg = np.asarray(nib.load(aseg_path).dataobj)
    stats = dict(line.split()[:2] for line in stats_path.read_text().splitlines()
                 if len(line.split()) >= 2)
    volume, bright = prepare_placement_volume(
        np.asarray(brain.dataobj), wm, surface="pial", mid_gray=float(stats["MID_GRAY"]),
    )
    placement = volume.copy()
    placement[bright == 130] = 0
    affine = surface_ras_to_voxel(brain.header, metadata)
    thresholds = np.array([float(stats[f"pial_{name}"]) for name in
                           ("inside_hi", "border_hi", "border_low", "outside_low", "outside_hi")])
    initial_normals = initial_vertex_normals(original, faces)
    border = compute_border_values_first_pass(
        volume, aseg, original, initial_normals, original, ripped,
        np.full(len(original), -1.0, dtype=np.float32), affine, thresholds,
        hemisphere=hemi, surface="pial",
    )
    values = average_marked_values(border[0], border[4], ripped, faces, 5)
    neighbor_indices, neighbor_valid, _ = _ordered_neighbors(faces, len(original))
    ordered = (neighbor_indices, neighbor_valid)
    two_offsets, two_candidates = two_ring_neighbors(faces, len(original), ordered_neighbors=ordered)
    fixed_normals = original_vertex_normals(original, faces)

    def gradient(current: np.ndarray, cropped: np.ndarray) -> tuple[np.ndarray, ...]:
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
        normal = spring_gradient(current, normals, faces, ripped, weight=0.3,
                                 direction="normal", ordered_neighbors=ordered)
        with_normal = np.float32(averaged + normal)
        basis = tangent_basis(normals)
        scalar = quadratic_curvature(current, normals, basis, ripped, two_offsets, two_candidates)
        with_curvature = np.float32(with_normal + np.float32(scalar[:, None] * normals))
        tangent = spring_gradient(current, normals, faces, ripped, weight=0.3,
                                  direction="tangent", ordered_neighbors=ordered)
        final = np.float32(with_curvature + tangent)
        return intensity, with_repulsion, averaged, with_normal, with_curvature, final

    report = {
        "reference": f"installed FreeSurfer 8.2.0-1 {hemi} pial RAM steps 1 and 2",
        "inference_inputs": "original brain.finalsurfs, WM, aseg, white surface, label, autodet stats; no installed RAM or copied-source gradient as inference input",
        "limits": "First two accepted iterations only; no later optimizer, final surface, or downstream vertex-metric claim.",
        "sha256": {str(path): sha256(path) for path in
                   (args.installed_binary, white_path, label_path, stats_path,
                    brain_path, wm_path, aseg_path, *ram_paths)},
        "ordered_faces_exact": True,
        "steps": [],
    }
    current = original.copy()
    cropped = np.zeros(len(original), dtype=np.int32)
    for index in (1, 2):
        stages = gradient(current, cropped)
        proposal, displacement = unconstrained_step_with_offsets(current, stages[-1], ripped)
        accepted, order = asynchronous_first_step(
            current, faces, proposal, ripped, fast=True, offsets=displacement,
            ordered_neighbors=ordered,
        )
        blocked = np.any(proposal != current, axis=1) & np.all(accepted == current, axis=1)
        cropped = np.where(ripped, cropped, np.where(blocked, cropped + 1, 0)).astype(np.int32)
        entry = {
            "step": index,
            "dt": 0.5,
            "processed_vertices": len(order),
            "cropped_after_step": int(np.count_nonzero(cropped)),
            "candidate_vs_installed": compare(proposal, native[index - 1]),
            "accepted_vs_installed": compare(accepted, native[index - 1]),
        }
        source_paths = [probe / f"{hemi}.gradient.step{index:02d}.{name}" for name in STAGES]
        if all(path.exists() for path in source_paths):
            entry["stage_vs_copied_source_diagnostic_only"] = {
                name: compare(value, np.fromfile(path, STATE)["floats"][:, 6:9])
                for name, value, path in zip(STAGES, stages, source_paths)
            }
        report["steps"].append(entry)
        current = accepted
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"hemisphere": hemi, "steps": [
        {"step": entry["step"], "accepted_vs_installed": entry["accepted_vs_installed"]}
        for entry in report["steps"]
    ]}, indent=2))
    if args.require_exact and any(
        step["accepted_vs_installed"]["exact_components"] != step["accepted_vs_installed"]["total_components"]
        for step in report["steps"]
    ):
        raise SystemExit("installed pial coordinate bits differ")


if __name__ == "__main__":
    main()
