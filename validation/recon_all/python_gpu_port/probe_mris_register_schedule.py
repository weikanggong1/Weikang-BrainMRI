"""Continue spherical registration from a verified native-exact saved epoch."""

import argparse
import hashlib
import json
import re
from pathlib import Path
from time import perf_counter

import nibabel.freesurfer.io as fsio
import numpy as np
import tifffile
import torch

from fnit.recon_all.mris_register_atlas import sample_atlas_on_canonical_sphere
from fnit.recon_all.mris_register_blur import blur_atlas_frame
from fnit.recon_all.mris_register_kernels import (
    center_sphere, normalize_mean_curvature, project_sphere,
)
from fnit.recon_all.mris_register_line_search import (
    first_registration_line_search, first_registration_sse,
)
from fnit.recon_all.mris_register_nonlinear import (
    apply_spherical_gradient, average_gradients, correlation_gradient_add,
    face_area_normals, first_area_gradient, first_distance_gradient,
    ordered_neighbors_from_faces, original_chord_distances, registration_orig_area,
    registration_total_area, sphere_arc_distances, sphere_vertex_normals, tangent_basis,
)
from fnit.recon_all.mris_register_parameterization import parameterize_curvature


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(actual, expected):
    values = actual.cpu().numpy()
    delta = np.abs(values - expected)
    return {"exact_vertices": int(np.count_nonzero(np.all(values == expected, axis=1))),
            "total_vertices": len(values), "max_abs_error_mm": float(delta.max()),
            "predicted_positions_sha256": hashlib.sha256(values.tobytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser()
    for name in ("sphere", "smoothwm", "sulc", "atlas", "seed", "native_prefix",
                 "native_log", "final_surface", "report"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--start-epoch", type=int, default=7)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    start = perf_counter()
    sphere, faces = fsio.read_geometry(str(args.sphere))
    smoothwm, smoothwm_faces = fsio.read_geometry(str(args.smoothwm))
    seed, seed_faces = fsio.read_geometry(str(args.seed))
    assert np.array_equal(faces, smoothwm_faces) and np.array_equal(faces, seed_faces)
    atlas = tifffile.imread(args.atlas)
    vertices = torch.from_numpy(sphere)
    original = torch.from_numpy(smoothwm)
    triangles = torch.from_numpy(faces.astype(np.int64))
    canonical = project_sphere(center_sphere(vertices))
    sulc = torch.from_numpy(fsio.read_morph_data(str(args.sulc)).astype(np.float32))
    source_grid = parameterize_curvature(canonical, normalize_mean_curvature(sulc))
    curvature = normalize_mean_curvature(sample_atlas_on_canonical_sphere(
        canonical, blur_atlas_frame(source_grid, 4.0)))
    raw_mean = torch.from_numpy(atlas[3].view(np.float32).copy())
    raw_variance = torch.from_numpy(atlas[4].view(np.float32).copy())
    mean_curve = normalize_mean_curvature(sample_atlas_on_canonical_sphere(
        canonical, blur_atlas_frame(raw_mean, 4.0)))
    mean_grid = parameterize_curvature(canonical, mean_curve)
    variance_grid = blur_atlas_frame(raw_variance, 4.0)
    neighbors, degrees = ordered_neighbors_from_faces(triangles, len(vertices))
    original_distances = original_chord_distances(original, neighbors, degrees)
    original_face_areas, _ = face_area_normals(original, triangles)
    original_area = registration_orig_area(vertices, triangles)
    total_area = registration_total_area()
    setup_seconds = perf_counter() - start

    def objective(positions):
        return first_registration_sse(positions, triangles, neighbors, degrees,
                                      original_distances, original_face_areas,
                                      curvature, mean_grid, variance_grid,
                                      original_area, total_area)

    def force(positions):
        current = project_sphere(positions)
        normals = sphere_vertex_normals(current, triangles)
        distances = sphere_arc_distances(current, neighbors, degrees)
        avg_vertex_dist = float(distances.double().sum() / degrees.sum())
        after_distance = first_distance_gradient(vertices, original, positions, triangles)
        after_area = first_area_gradient(vertices, original, positions, triangles,
                                         after_distance)
        e1, e2 = tangent_basis(normals)
        gradient = correlation_gradient_add(after_area, current, curvature,
                                            e1, e2, mean_grid, variance_grid,
                                            avg_vertex_dist)
        return current, gradient

    schedule = {int(number): int(navgs) for number, navgs in re.findall(
        r"^(\d{3}):.*avgs: (\d+)$", args.native_log.read_text(), re.MULTILINE)}
    current = torch.from_numpy(seed).float()
    rows = []
    for epoch in sorted(number for number in schedule if number >= args.start_epoch):
        reference_path = Path(f"{args.native_prefix}{epoch:04d}")
        if not reference_path.exists():
            break
        start_epoch = perf_counter()
        projected, gradient = force(current)
        force_seconds = perf_counter() - start_epoch
        start_average = perf_counter()
        averaged = average_gradients(gradient, neighbors, degrees, schedule[epoch])
        average_seconds = perf_counter() - start_average
        start_line = perf_counter()
        dt, samples = first_registration_line_search(projected, averaged, objective)
        predicted = apply_spherical_gradient(projected, averaged, dt)
        line_seconds = perf_counter() - start_line
        reference, reference_faces = fsio.read_geometry(str(reference_path))
        assert np.array_equal(faces, reference_faces)
        parity = compare(predicted, reference)
        row = {"epoch": epoch, "navgs": schedule[epoch], "dt": dt,
               "line_samples": samples, "comparison": parity,
               "reference_surface_sha256": sha(reference_path),
               "seconds_excluding_io": {"force": force_seconds,
                                        "average": average_seconds, "line": line_seconds}}
        rows.append(row)
        print(json.dumps({"epoch": epoch, "navgs": schedule[epoch], "dt": dt,
                          **parity, "seconds": row["seconds_excluding_io"]}), flush=True)
        current = predicted
        if parity["exact_vertices"] != len(reference):
            break
    final, final_faces = fsio.read_geometry(str(args.final_surface))
    assert np.array_equal(faces, final_faces)
    report = {"input_sha256": {name: sha(getattr(args, name)) for name in
                                 ("sphere", "smoothwm", "sulc", "atlas", "seed", "native_log")},
              "final_surface_sha256": sha(args.final_surface),
              "start_epoch": args.start_epoch, "setup_seconds": setup_seconds,
              "epochs": rows,
              "first_mismatching_epoch": next((row["epoch"] for row in rows
                                                 if row["comparison"]["exact_vertices"]
                                                 != row["comparison"]["total_vertices"]), None)}
    if rows and report["first_mismatching_epoch"] is None:
        report["last_surface_vs_final"] = compare(current, final)
    args.report.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
