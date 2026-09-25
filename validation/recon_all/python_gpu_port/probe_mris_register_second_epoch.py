"""Propagate sigma-4 registration updates from one rigid seed."""

from __future__ import annotations

import argparse
import hashlib
import json
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
    registration_total_area, sphere_arc_distances, sphere_vertex_normals,
    tangent_basis,
)
from fnit.recon_all.mris_register_parameterization import parameterize_curvature


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(actual: torch.Tensor, expected: np.ndarray) -> dict:
    values = actual.cpu().numpy()
    delta = np.abs(values - expected)
    return {"exact_vectors": int(np.count_nonzero(np.all(values == expected, axis=1))),
            "within_1e-5_vectors": int(np.count_nonzero(np.all(delta <= 1e-5, axis=1))),
            "total_vectors": len(values), "max_abs_error": float(delta.max()),
            "predicted_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
            "reference_sha256": hashlib.sha256(expected.tobytes()).hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("sphere", "smoothwm", "sulc", "rigid", "atlas", "native_first",
                 "native_second", "native_capture", "report"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--native-third", type=Path)
    parser.add_argument("--third-capture", type=Path)
    parser.add_argument("--native-fourth", type=Path)
    parser.add_argument("--fourth-capture", type=Path)
    parser.add_argument("--native-fifth", type=Path)
    parser.add_argument("--fifth-capture", type=Path)
    args = parser.parse_args()
    if (args.native_third is None) != (args.third_capture is None):
        parser.error("--native-third and --third-capture must be given together")
    if (args.native_fourth is None) != (args.fourth_capture is None):
        parser.error("--native-fourth and --fourth-capture must be given together")
    if args.native_fourth is not None and args.native_third is None:
        parser.error("the fourth update requires --native-third")
    if (args.native_fifth is None) != (args.fifth_capture is None):
        parser.error("--native-fifth and --fifth-capture must be given together")
    if args.native_fifth is not None and args.native_fourth is None:
        parser.error("the fifth update requires --native-fourth")
    torch.set_num_threads(args.threads)
    start = perf_counter()
    sphere, faces = fsio.read_geometry(str(args.sphere))
    smoothwm, original_faces = fsio.read_geometry(str(args.smoothwm))
    rigid, rigid_faces = fsio.read_geometry(str(args.rigid))
    native_first, first_faces = fsio.read_geometry(str(args.native_first))
    native_second, second_faces = fsio.read_geometry(str(args.native_second))
    assert all(np.array_equal(faces, other) for other in
               (original_faces, rigid_faces, first_faces, second_faces))
    atlas = tifffile.imread(args.atlas)
    vertices = torch.from_numpy(sphere)
    original = torch.from_numpy(smoothwm)
    triangles = torch.from_numpy(faces.astype(np.int64))
    rigid_tensor = torch.from_numpy(rigid)
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
    setup_seconds = perf_counter() - start
    neighbors, degrees = ordered_neighbors_from_faces(triangles, len(vertices))
    original_distances = original_chord_distances(original, neighbors, degrees)
    original_face_areas, _ = face_area_normals(original, triangles)
    original_area = registration_orig_area(vertices, triangles)
    total_area = registration_total_area()

    def objective(positions: torch.Tensor) -> float:
        return first_registration_sse(positions, triangles, neighbors, degrees,
                                      original_distances, original_face_areas,
                                      curvature, mean_grid, variance_grid,
                                      original_area, total_area)

    def force(positions: torch.Tensor) -> tuple[torch.Tensor, dict]:
        start_force = perf_counter()
        current = project_sphere(positions)
        normals = sphere_vertex_normals(current, triangles)
        distances = sphere_arc_distances(current, neighbors, degrees)
        avg_vertex_dist = float(distances.double().sum() / degrees.sum())
        after_distance = first_distance_gradient(vertices, original, positions, triangles)
        after_area = first_area_gradient(vertices, original, positions, triangles,
                                         after_distance)
        e1, e2 = tangent_basis(normals)
        after_correlation = correlation_gradient_add(after_area, current, curvature,
                                                      e1, e2, mean_grid, variance_grid,
                                                      avg_vertex_dist)
        return after_correlation, {"projected": current, "normals": normals,
                                   "distance": after_distance, "area": after_area,
                                   "avg_vertex_dist": avg_vertex_dist,
                                   "force_seconds": perf_counter() - start_force}

    first_gradient, first_force = force(rigid_tensor)
    start_average = perf_counter()
    first_averaged = average_gradients(first_gradient, neighbors, degrees, 16384)
    first_average_seconds = perf_counter() - start_average
    start_line = perf_counter()
    first_dt, first_samples = first_registration_line_search(first_force["projected"],
                                                              first_averaged, objective)
    predicted_first = apply_spherical_gradient(first_force["projected"],
                                                first_averaged, first_dt)
    first_line_seconds = perf_counter() - start_line
    first_comparison = compare(predicted_first, native_first)

    second_gradient, second_force = force(predicted_first)
    capture = args.native_capture
    read = lambda name: np.fromfile(capture / name, "<f4").reshape(-1, 3)
    second_comparisons = {
        "start": compare(second_force["projected"], read("before_distance_positions.bin")),
        "normals": compare(second_force["normals"], read("before_distance_normals.bin")),
        "distance": compare(second_force["distance"], read("after_distance_gradient.bin")),
        "area": compare(second_force["area"], read("after_area_gradient.bin")),
        "correlation": compare(second_gradient, read("after_correlation_gradient.bin")),
        "before_average": compare(second_gradient, read("before_average_gradient.bin")),
    }
    start_average = perf_counter()
    second_averaged = average_gradients(second_gradient, neighbors, degrees, 4096)
    second_average_seconds = perf_counter() - start_average
    second_comparisons["after_average"] = compare(
        second_averaged, read("after_average_gradient.bin"))
    start_line = perf_counter()
    second_dt, second_samples = first_registration_line_search(second_force["projected"],
                                                                second_averaged, objective)
    predicted_second = apply_spherical_gradient(second_force["projected"],
                                                 second_averaged, second_dt)
    second_line_seconds = perf_counter() - start_line
    second_comparisons["saved_surface"] = compare(predicted_second, native_second)
    third_report = None
    third_seconds = {}
    fourth_report = None
    fourth_seconds = {}
    fifth_report = None
    fifth_seconds = {}
    if args.native_third is not None:
        native_third, third_faces = fsio.read_geometry(str(args.native_third))
        assert np.array_equal(faces, third_faces)
        third_gradient, third_force = force(predicted_second)
        capture = args.third_capture
        read_third = lambda name: np.fromfile(capture / name, "<f4").reshape(-1, 3)
        third_comparisons = {
            "start": compare(third_force["projected"], read_third("before_distance_positions.bin")),
            "normals": compare(third_force["normals"], read_third("before_distance_normals.bin")),
            "distance": compare(third_force["distance"], read_third("after_distance_gradient.bin")),
            "area": compare(third_force["area"], read_third("after_area_gradient.bin")),
            "correlation": compare(third_gradient, read_third("after_correlation_gradient.bin")),
            "before_average": compare(third_gradient, read_third("before_average_gradient.bin")),
        }
        start_average = perf_counter()
        third_averaged = average_gradients(third_gradient, neighbors, degrees, 1024)
        third_seconds["third_average"] = perf_counter() - start_average
        third_comparisons["after_average"] = compare(
            third_averaged, read_third("after_average_gradient.bin"))
        start_line = perf_counter()
        third_dt, third_samples = first_registration_line_search(third_force["projected"],
                                                                  third_averaged, objective)
        predicted_third = apply_spherical_gradient(third_force["projected"],
                                                    third_averaged, third_dt)
        third_seconds["third_line"] = perf_counter() - start_line
        third_seconds["third_force"] = third_force["force_seconds"]
        third_comparisons["saved_surface"] = compare(predicted_third, native_third)
        third_report = {"navgs": 1024, "dt": third_dt, "line_samples": third_samples,
                        "avg_vertex_dist": third_force["avg_vertex_dist"],
                        "comparisons": third_comparisons,
                        "native_surface_sha256": sha(args.native_third),
                        "capture_script_sha256": sha(args.third_capture.parent /
                                                         "capture_mris_register_third_epoch.gdb")}
        if args.native_fourth is not None:
            native_fourth, fourth_faces = fsio.read_geometry(str(args.native_fourth))
            assert np.array_equal(faces, fourth_faces)
            fourth_gradient, fourth_force = force(predicted_third)
            capture = args.fourth_capture
            read_fourth = lambda name: np.fromfile(capture / name, "<f4").reshape(-1, 3)
            fourth_comparisons = {
                "start": compare(fourth_force["projected"], read_fourth("before_distance_positions.bin")),
                "normals": compare(fourth_force["normals"], read_fourth("before_distance_normals.bin")),
                "distance": compare(fourth_force["distance"], read_fourth("after_distance_gradient.bin")),
                "area": compare(fourth_force["area"], read_fourth("after_area_gradient.bin")),
                "correlation": compare(fourth_gradient, read_fourth("after_correlation_gradient.bin")),
                "before_average": compare(fourth_gradient, read_fourth("before_average_gradient.bin")),
            }
            start_average = perf_counter()
            fourth_averaged = average_gradients(fourth_gradient, neighbors, degrees, 256)
            fourth_seconds["fourth_average"] = perf_counter() - start_average
            fourth_comparisons["after_average"] = compare(
                fourth_averaged, read_fourth("after_average_gradient.bin"))
            start_line = perf_counter()
            fourth_dt, fourth_samples = first_registration_line_search(fourth_force["projected"],
                                                                        fourth_averaged, objective)
            predicted_fourth = apply_spherical_gradient(fourth_force["projected"],
                                                         fourth_averaged, fourth_dt)
            fourth_seconds["fourth_line"] = perf_counter() - start_line
            fourth_seconds["fourth_force"] = fourth_force["force_seconds"]
            fourth_comparisons["saved_surface"] = compare(predicted_fourth, native_fourth)
            fourth_report = {"navgs": 256, "dt": fourth_dt, "line_samples": fourth_samples,
                             "avg_vertex_dist": fourth_force["avg_vertex_dist"],
                             "comparisons": fourth_comparisons,
                             "native_surface_sha256": sha(args.native_fourth),
                             "capture_script_sha256": sha(args.fourth_capture.parent /
                                                              "capture_mris_register_fourth_epoch.gdb")}
            if args.native_fifth is not None:
                native_fifth, fifth_faces = fsio.read_geometry(str(args.native_fifth))
                assert np.array_equal(faces, fifth_faces)
                fifth_gradient, fifth_force = force(predicted_fourth)
                capture = args.fifth_capture
                read_fifth = lambda name: np.fromfile(capture / name, "<f4").reshape(-1, 3)
                fifth_comparisons = {
                    "start": compare(fifth_force["projected"], read_fifth("before_distance_positions.bin")),
                    "normals": compare(fifth_force["normals"], read_fifth("before_distance_normals.bin")),
                    "distance": compare(fifth_force["distance"], read_fifth("after_distance_gradient.bin")),
                    "area": compare(fifth_force["area"], read_fifth("after_area_gradient.bin")),
                    "correlation": compare(fifth_gradient, read_fifth("after_correlation_gradient.bin")),
                    "before_average": compare(fifth_gradient, read_fifth("before_average_gradient.bin")),
                }
                start_average = perf_counter()
                fifth_averaged = average_gradients(fifth_gradient, neighbors, degrees, 64)
                fifth_seconds["fifth_average"] = perf_counter() - start_average
                fifth_comparisons["after_average"] = compare(
                    fifth_averaged, read_fifth("after_average_gradient.bin"))
                start_line = perf_counter()
                fifth_dt, fifth_samples = first_registration_line_search(fifth_force["projected"],
                                                                          fifth_averaged, objective)
                predicted_fifth = apply_spherical_gradient(fifth_force["projected"],
                                                            fifth_averaged, fifth_dt)
                fifth_seconds["fifth_line"] = perf_counter() - start_line
                fifth_seconds["fifth_force"] = fifth_force["force_seconds"]
                fifth_comparisons["saved_surface"] = compare(predicted_fifth, native_fifth)
                fifth_report = {"navgs": 64, "dt": fifth_dt, "line_samples": fifth_samples,
                                "avg_vertex_dist": fifth_force["avg_vertex_dist"],
                                "comparisons": fifth_comparisons,
                                "native_surface_sha256": sha(args.native_fifth),
                                "capture_script_sha256": sha(args.fifth_capture.parent /
                                                                 "capture_mris_register_fifth_epoch.gdb")}
    report = {
        "hemisphere": args.sphere.name[:2], "device": "cpu", "threads": args.threads,
        "input_sha256": {key: sha(getattr(args, key)) for key in
                         ("sphere", "smoothwm", "sulc", "rigid", "atlas")},
        "native_sha256": {key: sha(getattr(args, key)) for key in
                          ("native_first", "native_second")},
        "capture_script_sha256": sha(Path(__file__).with_name(
            "capture_mris_register_second_epoch.gdb")),
        "first": {"dt": first_dt, "line_samples": first_samples,
                  "saved_surface": first_comparison},
        "second": {"navgs": 4096, "dt": second_dt, "line_samples": second_samples,
                   "avg_vertex_dist": second_force["avg_vertex_dist"],
                   "comparisons": second_comparisons},
        "third": third_report,
        "fourth": fourth_report,
        "fifth": fifth_report,
        "seconds_excluding_io": {
            "prepare_curvature_and_atlas": setup_seconds,
            "first_force": first_force["force_seconds"],
            "first_average": first_average_seconds, "first_line": first_line_seconds,
            "second_force": second_force["force_seconds"],
            "second_average": second_average_seconds, "second_line": second_line_seconds,
            **third_seconds, **fourth_seconds, **fifth_seconds},
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"first": first_comparison,
                      "second": {k: {"exact": v["exact_vectors"],
                                      "max_error": v["max_abs_error"]}
                                 for k, v in second_comparisons.items()},
                      "first_dt": first_dt, "second_dt": second_dt,
                      "third_dt": None if third_report is None else third_report["dt"],
                      "third": None if third_report is None else {
                          k: {"exact": v["exact_vectors"], "max_error": v["max_abs_error"]}
                          for k, v in third_report["comparisons"].items()},
                      "fourth_dt": None if fourth_report is None else fourth_report["dt"],
                      "fourth": None if fourth_report is None else {
                          k: {"exact": v["exact_vectors"], "max_error": v["max_abs_error"]}
                          for k, v in fourth_report["comparisons"].items()},
                      "fifth_dt": None if fifth_report is None else fifth_report["dt"],
                      "fifth": None if fifth_report is None else {
                          k: {"exact": v["exact_vectors"], "max_error": v["max_abs_error"]}
                          for k, v in fifth_report["comparisons"].items()},
                      "report": str(args.report)}, indent=2))


if __name__ == "__main__":
    main()
