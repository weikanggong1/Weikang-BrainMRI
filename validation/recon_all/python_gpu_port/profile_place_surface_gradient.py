"""Time fixed third LH pial gradient terms with exact source checkpoint checks."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np


STATE = np.dtype([("floats", "<f4", 9), ("flags", "<i4", 3)])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=Path, required=True)
    parser.add_argument("--second-python", type=Path, required=True)
    parser.add_argument("--first-source", type=Path, required=True)
    parser.add_argument("--second-gradient", type=Path, required=True)
    parser.add_argument("--third-gradient", type=Path, required=True)
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--reuse-neighbors", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(args.module))
    from fnit.recon_all.place_surface_border import compute_border_values_first_pass
    from fnit.recon_all.place_surface_curvature import quadratic_curvature, tangent_basis, two_ring_neighbors
    from fnit.recon_all.place_surface_geometry import surface_ras_to_voxel
    from fnit.recon_all.place_surface_gradient_average import average_signed_gradients
    from fnit.recon_all.place_surface_intensity import intensity_gradient
    from fnit.recon_all.place_surface_normals import initial_vertex_normals
    from fnit.recon_all.place_surface_objective import intensity_error, tangential_spring_energy, surface_total_area
    from fnit.recon_all.place_surface_repulsion import original_vertex_normals, surface_repulsion_gradient, vertex_buckets
    from fnit.recon_all.place_surface_rip import rip_outside_label
    from fnit.recon_all.place_surface_smoothing import average_marked_values, _ordered_neighbors
    from fnit.recon_all.place_surface_spring import spring_gradient
    from fnit.recon_all.place_surface_step import unconstrained_step
    from fnit.recon_all.place_surface_volume import prepare_placement_volume

    started = time.perf_counter()
    subject = args.subject
    brain = nib.load(subject / "mri/brain.finalsurfs.mgz")
    wm = np.asarray(nib.load(subject / "mri/wm.mgz").dataobj)
    segmentation = np.asarray(nib.load(subject / "mri/aseg.presurf.mgz").dataobj)
    original, faces, metadata = nib.freesurfer.read_geometry(subject / "surf/lh.white", read_metadata=True)
    original = original.astype(np.float32)
    second = np.load(args.second_python).astype(np.float32, copy=False)
    first, first_faces = nib.freesurfer.read_geometry(args.first_source)
    first = first.astype(np.float32)
    if not np.array_equal(faces, first_faces):
        raise SystemExit("ordered first-step faces differ")
    ripped = np.asarray(rip_outside_label(
        len(original), nib.freesurfer.read_label(subject / "label/lh.cortex+hipamyg.label"),
    ), dtype=np.bool_)
    previous_gradient = np.fromfile(args.second_gradient, dtype=STATE)["floats"][:, 6:9]
    proposal = unconstrained_step(first, previous_gradient, ripped)
    cropped = np.any(proposal != first, axis=1) & np.all(second == first, axis=1)
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
        volume, segmentation, original, initial_vertex_normals(original, faces),
        original, ripped, np.full(len(original), -1.0, dtype=np.float32),
        affine, thresholds, hemisphere="lh", surface="pial",
    )
    values = average_marked_values(border[0], border[4], ripped, faces, 5)
    fixed_normals = original_vertex_normals(original, faces)
    ordered = None
    if args.reuse_neighbors:
        neighbor_indices, neighbor_valid, _ = _ordered_neighbors(faces, len(original))
        ordered = (neighbor_indices, neighbor_valid)
    two_offsets, two_candidates = two_ring_neighbors(faces, len(original), ordered_neighbors=ordered)
    prep_seconds = time.perf_counter() - started
    reference = np.fromfile(args.third_gradient, dtype=STATE)["floats"][:, 6:9]
    report = {
        "reference": "pinned source third LH pial gradient",
        "load_before": os.getloadavg(),
        "input_prep_seconds": prep_seconds,
        "cropped_at_start": int(cropped.sum()),
        "reuse_neighbors": args.reuse_neighbors,
        "repeats": [],
    }
    for repeat in range(args.repeats):
        timings = {}
        clock = time.perf_counter()

        def lap(name: str) -> None:
            nonlocal clock
            next_clock = time.perf_counter()
            timings[name] = next_clock - clock
            clock = next_clock

        normals = initial_vertex_normals(second, faces)
        lap("current_normals")
        intensity = intensity_gradient(
            placement, second, normals, ripped, values, border[5], affine,
            brain.header.get_zooms()[:3], weight=0.2, sigma_global=2.0,
        )
        lap("intensity_gradient")
        offsets, candidates = vertex_buckets(second, original, ripped)
        lap("repulsion_candidates")
        repulsion = surface_repulsion_gradient(
            second, normals, original, fixed_normals, ripped, offsets, candidates,
            weight=5.0, cropped=cropped,
        )
        with_repulsion = np.float32(intensity + repulsion)
        lap("repulsion_gradient")
        averaged = average_signed_gradients(with_repulsion, faces, ripped, 2, ordered_neighbors=ordered)
        lap("signed_gradient_average")
        normal = spring_gradient(second, normals, faces, ripped, weight=0.3, direction="normal", ordered_neighbors=ordered)
        with_normal = np.float32(averaged + normal)
        lap("normal_spring")
        basis = tangent_basis(normals)
        lap("tangent_basis")
        scalar = quadratic_curvature(second, normals, basis, ripped, two_offsets, two_candidates)
        with_curvature = np.float32(with_normal + np.float32(scalar[:, None] * normals))
        lap("quadratic_curvature")
        tangent = spring_gradient(second, normals, faces, ripped, weight=0.3, direction="tangent", ordered_neighbors=ordered)
        final = np.float32(with_curvature + tangent)
        lap("tangential_spring")
        intensity_error(placement, second, values, ripped, affine)
        lap("intensity_sse")
        tangential_spring_energy(second, normals, faces, ripped, ordered_neighbors=ordered)
        lap("tangential_spring_sse")
        surface_total_area(second, faces)
        lap("surface_area")
        report["repeats"].append({
            "repeat": repeat,
            "timings_seconds": timings,
            "total_seconds": sum(timings.values()),
            "load_after": os.getloadavg(),
            "third_gradient_exact": bool(np.array_equal(final, reference)),
        })
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not all(item["third_gradient_exact"] for item in report["repeats"]):
        raise SystemExit("timed third LH pial gradient differs from source")


if __name__ == "__main__":
    main()
