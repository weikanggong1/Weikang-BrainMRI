"""Probe the first conventional-sphere epoch from isolated native snapshots."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import nibabel.freesurfer.io as fsio
import numpy as np
from scipy.optimize import minimize_scalar

from fnit.recon_all.sphere_python import project_radially
from fnit.recon_all.sphere_standard_metric import (
    average_standard_metric, sample_standard_metric_matrix,
)
from fnit.recon_all.sphere_standard_unfold import first_epoch_gradient


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _metric(values: np.ndarray) -> dict:
    lengths = np.sqrt(np.sum(values.astype(np.float64) ** 2, axis=1))
    return {"l2": float(np.linalg.norm(lengths)),
            "max": float(lengths.max()),
            "mean": float(lengths.mean()),
            "nonzero_vertices": int(np.count_nonzero(lengths))}


def _native_epoch_values(path: Path) -> dict:
    text = path.read_text(errors="replace")
    start = text.index("pass 1: epoch 1 of 3")
    span = text[start:text.index("pass 1: epoch 2 of 3", start)]
    gradient = re.search(
        r"grad=([0-9.]+), max_del=([0-9.]+), mean=([0-9.]+), .*?min_dt=([0-9.]+)",
        span)
    selection = re.search(r"sses:.*?min (\d+) \(([0-9.]+)\)", span)
    if gradient is None or selection is None:
        raise ValueError("native first-epoch verbose line-search diagnostics missing")
    return {"gradient_l2_printed": float(gradient.group(1)),
            "gradient_max_printed": float(gradient.group(2)),
            "gradient_mean_printed": float(gradient.group(3)),
            "min_dt_printed": float(gradient.group(4)),
            "selected_candidate": int(selection.group(1)),
            "selected_dt_printed": float(selection.group(2))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("smoothwm", type=Path)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("native_verbose", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--project-start", action="store_true",
                        help="project a preceding repair-step snapshot at epoch entry")
    args = parser.parse_args()
    original_xyz, original_faces = fsio.read_geometry(str(args.smoothwm))
    before, faces = fsio.read_geometry(str(args.before))
    after, after_faces = fsio.read_geometry(str(args.after))
    if not np.array_equal(faces, original_faces) or not np.array_equal(faces, after_faces):
        raise ValueError("native snapshot face order differs from metric source")
    t0 = time.perf_counter()
    offsets, ids, raw, sampling = sample_standard_metric_matrix(
        original_xyz, faces)
    original_dist, _, _ = average_standard_metric(offsets, ids, raw)
    t1 = time.perf_counter()
    start = (project_radially(np.asarray(before, np.float32), already_sphere=True)
             if args.project_start else np.asarray(before, np.float32))
    distance, area, total, geometry = first_epoch_gradient(
        start, faces, original_xyz, offsets, ids, original_dist)
    t2 = time.perf_counter()
    native = _native_epoch_values(args.native_verbose)

    def predicted(dt: float) -> np.ndarray:
        shifted = (start.astype(np.float64)
                   + dt * total.astype(np.float64)).astype(np.float32)
        return project_radially(shifted, already_sphere=True)

    def error(dt: float) -> float:
        trial = predicted(dt).astype(np.float64) - after.astype(np.float64)
        return float(np.mean(np.sum(trial * trial, axis=1)))

    approximate_dt = native["selected_dt_printed"]
    fit = minimize_scalar(error, bounds=(approximate_dt * 0.7,
                                         approximate_dt * 1.3),
                          method="bounded", options={"xatol": 1e-10})
    area_active = np.any(area != 0, axis=1)

    def safe_error(dt: float) -> float:
        difference = predicted(dt)[~area_active].astype(np.float64) - after[~area_active]
        return float(np.mean(np.sum(difference * difference, axis=1)))

    safe_fit = minimize_scalar(safe_error, bounds=(approximate_dt * 0.7,
                                                   approximate_dt * 1.3),
                               method="bounded", options={"xatol": 1e-10})
    result = {"sampling": sampling, "geometry": geometry,
              "sha256": {"smoothwm": _sha256(args.smoothwm),
                         "before_snapshot": _sha256(args.before),
                         "after_snapshot": _sha256(args.after),
                         "native_verbose": _sha256(args.native_verbose)},
              "seconds": {"matrix_sampling_and_average": t1 - t0,
                          "first_gradient": t2 - t1},
              "native": native,
              "distance_force": _metric(distance),
              "negative_area_force": _metric(area),
              "total_force": _metric(total),
              "first_vertex_distance": distance[0].tolist(),
              "first_vertex_area": area[0].tolist(),
              "first_vertex_total": total[0].tolist(),
              "native_actual_update": _metric(after - before),
              "pre_epoch_radial_projection_exact_components": int(np.count_nonzero(start == before)),
              "pre_epoch_radial_projection_total_components": int(start.size)}
    for label, dt in (("printed_dt", approximate_dt), ("fitted_dt", fit.x),
                      ("safe_fitted_dt", safe_fit.x)):
        trial = predicted(dt)
        delta = np.linalg.norm(trial.astype(np.float64) - after.astype(np.float64), axis=1)
        worst = int(np.argmax(delta))
        result[label] = {"dt": float(dt),
                         "area_active_vertices": int(np.count_nonzero(area_active)),
                         "max_safe_vertex_error_mm": float(delta[~area_active].max()),
                         "max_area_active_vertex_error_mm": float(delta[area_active].max()),
                         "worst_vertex": worst,
                         "worst_vertex_area_active": bool(area_active[worst]),
                         "exact_coordinate_components": int(np.count_nonzero(trial == after)),
                         "vertices_with_error_le_1e-6_mm": int(np.count_nonzero(delta <= 1e-6)),
                         "vertices_with_error_le_1e-5_mm": int(np.count_nonzero(delta <= 1e-5)),
                         "total_coordinate_components": int(after.size),
                         "median_vertex_error_mm": float(np.median(delta)),
                         "max_vertex_error_mm": float(delta.max()),
                         "rms_vertex_error_mm": float(np.sqrt(np.mean(delta * delta)))}
    output = json.dumps(result, indent=2)
    if args.report:
        args.report.write_text(output + "\n")
    print(output)


if __name__ == "__main__":
    main()
