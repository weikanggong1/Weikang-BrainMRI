"""Regenerate the first LH pial optimizer gradient from frozen command inputs."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np


STATE = np.dtype([("floats", "<f4", 9), ("flags", "<i4", 3)])


def compare(actual: np.ndarray, expected: np.ndarray) -> dict:
    error = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    return {
        "exact_elements": int(np.count_nonzero(actual == expected)),
        "total_elements": int(expected.size),
        "max_abs_error": float(error.max(initial=0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
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
    from fnit.recon_all.place_surface_volume import prepare_placement_volume

    start = time.perf_counter()
    subject = args.subject
    brain = nib.load(subject / "mri/brain.finalsurfs.mgz")
    wm = np.asarray(nib.load(subject / "mri/wm.mgz").dataobj)
    segmentation = np.asarray(nib.load(subject / "mri/aseg.presurf.mgz").dataobj)
    xyz, faces, metadata = nib.freesurfer.read_geometry(subject / "surf/lh.white", read_metadata=True)
    xyz = xyz.astype(np.float32)
    normals = initial_vertex_normals(xyz, faces)
    ripped = rip_outside_label(len(xyz), nib.freesurfer.read_label(subject / "label/lh.cortex+hipamyg.label"))
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
    border = compute_border_values_first_pass(
        volume, segmentation, xyz, normals, xyz, ripped,
        np.full(len(xyz), -1.0, dtype=np.float32), affine, thresholds,
        hemisphere="lh", surface="pial",
    )
    values = average_marked_values(border[0], border[4], ripped, faces, 5)
    intensity = intensity_gradient(
        placement, xyz, normals, ripped, values, border[5], affine,
        brain.header.get_zooms()[:3], weight=0.2, sigma_global=2.0,
    )
    original_normals = original_vertex_normals(xyz, faces)
    offsets, candidates = vertex_buckets(xyz, xyz, ripped)
    repulsion = surface_repulsion_gradient(
        xyz, normals, xyz, original_normals, ripped, offsets, candidates, weight=5.0,
    )
    with_repulsion = np.float32(intensity + repulsion)
    averaged = average_signed_gradients(with_repulsion, faces, ripped, 2)
    normal = spring_gradient(xyz, normals, faces, ripped, weight=0.3, direction="normal")
    with_normal = np.float32(averaged + normal)
    basis = tangent_basis(normals)
    two_offsets, two_candidates = two_ring_neighbors(faces, len(xyz))
    scalar = quadratic_curvature(xyz, normals, basis, ripped, two_offsets, two_candidates)
    with_curvature = np.float32(with_normal + np.float32(scalar[:, None] * normals))
    tangent = spring_gradient(xyz, normals, faces, ripped, weight=0.3, direction="tangent")
    final = np.float32(with_curvature + tangent)
    computation_seconds = time.perf_counter() - start

    checkpoints = {
        "intensity": intensity,
        "surface_repulsion": with_repulsion,
        "pre_normal_spring": averaged,
        "normal_spring": with_normal,
        "curvature": with_curvature,
        "tangential_spring": final,
    }
    checked = {}
    for name, candidate in checkpoints.items():
        expected = np.fromfile(args.probe / f"lh.gradient.{name}", dtype=STATE)
        checked[name] = compare(candidate, expected["floats"][:, 6:9])
    report = {
        "reference": "copied pinned FreeSurfer 8.2 source probe, fixed first LH pial optimizer step",
        "inputs": "original fs_sub01 MRI, WM, aseg, LH white, label, frozen autodet stats",
        "comparisons": checked,
        "seconds_including_io_and_jit_before_comparison": computation_seconds,
    }
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if args.require_exact and any(item["exact_elements"] != item["total_elements"] for item in checked.values()):
        raise SystemExit("first pial gradient differs from pinned source")


if __name__ == "__main__":
    main()
