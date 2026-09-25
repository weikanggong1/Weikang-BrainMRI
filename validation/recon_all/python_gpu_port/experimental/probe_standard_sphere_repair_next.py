"""Probe isolated conventional-sphere repair and next-epoch checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import nibabel.freesurfer.io as fsio
import numpy as np

from fnit.recon_all import (
    sphere_standard_line_search as line_search_module,
    sphere_standard_metric as metric_module,
    sphere_standard_unfold as unfold_module,
)
from fnit.recon_all.sphere_python import project_radially
from fnit.recon_all.sphere_standard_line_search import first_epoch_line_search
from fnit.recon_all.sphere_standard_metric import (
    average_standard_metric, sample_standard_metric_matrix,
)
from fnit.recon_all.sphere_standard_python import project_before_standard_unfold
from fnit.recon_all.sphere_standard_unfold import (
    _face_geometry, first_epoch_gradient,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _native_steps(path: Path) -> list[dict]:
    content = path.read_text(errors="replace")
    gradients = re.findall(r"grad=([0-9.]+), max_del=([0-9.]+), mean=([0-9.]+)", content)
    choices = re.findall(r"sses: ([0-9. ]+) min (\d+) \(([0-9.]+)\)", content)
    if len(gradients) != len(choices):
        raise ValueError("native gradient/line-search diagnostic count differs")
    return [{"gradient_l2_printed": float(g[0]),
             "gradient_max_printed": float(g[1]),
             "gradient_mean_printed": float(g[2]),
             "candidate_sse_printed": [float(v) for v in sses.split()],
             "selected_index": int(index), "selected_dt_printed": float(dt)}
            for g, (sses, index, dt) in zip(gradients, choices)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inflated", type=Path)
    parser.add_argument("smoothwm", type=Path)
    parser.add_argument("snapshot_prefix", type=Path)
    parser.add_argument("native_verbose", type=Path)
    parser.add_argument("--step", action="append", required=True,
                        help="snapshot index:distance weight, e.g. 0:1e-6")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    original, faces = fsio.read_geometry(str(args.smoothwm))
    inflated, inflated_faces = fsio.read_geometry(str(args.inflated))
    if not np.array_equal(faces, inflated_faces):
        raise ValueError("inflated and smoothwm face order differs")
    faces = np.asarray(faces, np.int32)
    projected = project_before_standard_unfold(inflated)
    projected_area, _ = _face_geometry(projected, faces)
    negative = float(-np.sum(projected_area[projected_area < 0], dtype=np.float64))
    analytic_area = float(np.float32(4 * np.pi * 100 * 100))
    negative_pct = 100 * negative / (negative + analytic_area)
    decision = {"negative_faces": int(np.count_nonzero(projected_area < 0)),
                "negative_area_mm2": negative,
                "negative_area_pct": negative_pct,
                "native_threshold_pct": 0.005,
                "repair_required": bool(negative_pct > 0.005)}
    t0 = time.perf_counter()
    offsets, ids, raw, _ = sample_standard_metric_matrix(original, faces)
    original_dist, _, _ = average_standard_metric(offsets, ids, raw)
    matrix_seconds = time.perf_counter() - t0
    orig_area, _ = _face_geometry(np.asarray(original, np.float32), faces)
    orig_area = np.abs(orig_area)
    orig_total = np.float32(np.sum(orig_area, dtype=np.float64))
    native = _native_steps(args.native_verbose)
    result = {"input_sha256": {"inflated": _sha256(args.inflated),
                                "smoothwm": _sha256(args.smoothwm),
                                "native_verbose": _sha256(args.native_verbose)},
              "reference_commit": "d932c45",
              "requested_step_specs": args.step,
              "requested_step_specs_sha256": hashlib.sha256(
                  "\n".join(args.step).encode()).hexdigest(),
              "implementation_sha256": {
                  "metric": _sha256(Path(metric_module.__file__)),
                  "unfold": _sha256(Path(unfold_module.__file__)),
                  "line_search": _sha256(Path(line_search_module.__file__)),
                  "probe": _sha256(Path(__file__))},
              "decision": decision,
              "matrix_seconds_including_jit": matrix_seconds,
              "steps": []}
    for spec in args.step:
        index_text, weight_text = spec.split(":", 1)
        index = int(index_text)
        weight = float(weight_text)
        before_path = Path(f"{args.snapshot_prefix}{index:04d}")
        after_path = Path(f"{args.snapshot_prefix}{index + 1:04d}")
        before, before_faces = fsio.read_geometry(str(before_path))
        after, after_faces = fsio.read_geometry(str(after_path))
        if not np.array_equal(faces, before_faces) or not np.array_equal(faces, after_faces):
            raise ValueError("native checkpoint face order differs")
        start = (np.asarray(before, np.float32) if index == 0 else
                 project_radially(np.asarray(before, np.float32), already_sphere=True))
        t1 = time.perf_counter()
        distance_force, area_force, gradient, geometry = first_epoch_gradient(
            start, faces, original, offsets, ids, original_dist, weight)
        gradient_seconds = time.perf_counter() - t1
        t2 = time.perf_counter()
        search = first_epoch_line_search(start, gradient, faces, offsets, ids,
                                         original_dist, orig_area, orig_total, weight)
        line_search_seconds = time.perf_counter() - t2
        shifted = (start.astype(np.float64) + search["selected_dt"]
                   * gradient.astype(np.float64)).astype(np.float32)
        predicted = project_radially(shifted, already_sphere=True)
        delta = np.linalg.norm(predicted.astype(np.float64) - after.astype(np.float64), axis=1)
        lengths = np.linalg.norm(gradient.astype(np.float64), axis=1)
        reference = native[index]
        step = {"index": index, "distance_weight": weight,
                "before_sha256": _sha256(before_path), "after_sha256": _sha256(after_path),
                "geometry": geometry, "gradient_seconds_including_jit": gradient_seconds,
                "line_search_seconds_including_jit": line_search_seconds,
                "gradient": {"l2": float(np.linalg.norm(lengths)),
                             "max": float(lengths.max()),
                             "mean": float(lengths.mean())},
                "native": reference, "line_search": search,
                "update": {"exact_components": int(np.count_nonzero(predicted == after)),
                           "total_components": int(after.size),
                           "vertices_le_1e-5_mm": int(np.count_nonzero(delta <= 1e-5)),
                           "vertices": len(delta),
                           "max_error_mm": float(delta.max()),
                           "rms_error_mm": float(np.sqrt(np.mean(delta * delta))),
                           "worst_vertex": int(np.argmax(delta))}}
        step["decision_matches_native"] = bool(
            search["selected_index"] == reference["selected_index"] and
            f"{search['selected_dt']:.3f}" == f"{reference['selected_dt_printed']:.3f}")
        step["coordinates_within_1e-5_mm"] = bool(np.all(delta <= 1e-5))
        result["steps"].append(step)
        if not step["decision_matches_native"] or not step["coordinates_within_1e-5_mm"]:
            from scipy.optimize import minimize_scalar

            def fit_error(dt: float) -> float:
                shifted = (start.astype(np.float64) + dt * gradient.astype(np.float64)).astype(np.float32)
                trial = project_radially(shifted, already_sphere=True)
                difference = trial.astype(np.float64) - after.astype(np.float64)
                return float(np.mean(np.sum(difference * difference, axis=1)))

            chosen = search["selected_dt"]
            fitted = minimize_scalar(fit_error, bounds=(chosen * 0.99, chosen * 1.01),
                                     method="bounded", options={"xatol": 1e-12})
            shifted = (start.astype(np.float64) + fitted.x * gradient.astype(np.float64)).astype(np.float32)
            fit_xyz = project_radially(shifted, already_sphere=True)
            fit_delta = np.linalg.norm(fit_xyz.astype(np.float64) - after.astype(np.float64), axis=1)
            step["diagnostic_fitted_dt"] = {
                "dt": float(fitted.x), "max_vertex_error_mm": float(fit_delta.max()),
                "rms_vertex_error_mm": float(np.sqrt(np.mean(fit_delta * fit_delta))),
                "exact_components": int(np.count_nonzero(fit_xyz == after)),
                "largest_vertex_residuals": [
                    {"vertex": int(v), "error_mm": float(fit_delta[v]),
                     "before": start[v].tolist(), "native_after": after[v].tolist(),
                     "predicted_after": fit_xyz[v].tolist(),
                     "distance_force": distance_force[v].tolist(),
                     "area_force": area_force[v].tolist(),
                     "total_force": gradient[v].tolist()}
                    for v in np.argsort(fit_delta)[-10:][::-1]]}
            result["first_divergent_step"] = index
            break
    output = json.dumps(result, indent=2)
    if args.report:
        args.report.write_text(output + "\n")
    print(output)


if __name__ == "__main__":
    main()
