"""Trace the default 1024-average sphere prefix through saved native updates."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import nibabel.freesurfer.io as fsio
import numpy as np

from fnit.recon_all.sphere_python import project_radially
from fnit.recon_all.sphere_standard_average import average_standard_gradient
from fnit.recon_all.sphere_standard_line_search import (
    first_epoch_line_search, first_epoch_sse,
)
from fnit.recon_all.sphere_standard_metric import (
    average_standard_metric, sample_standard_metric_matrix,
)
from fnit.recon_all.sphere_standard_nonlinear import (
    nonlinear_epoch_gradient, nonlinear_epoch_line_search, one_ring_metric,
)
from fnit.recon_all.sphere_standard_python import project_before_standard_unfold
from fnit.recon_all.sphere_standard_unfold import _face_geometry, first_epoch_gradient
from probe_standard_sphere_continuous import _compare, _coordinate_sha
from probe_standard_sphere_repair_next import _sha256


def _schedule(hemisphere: str) -> list[tuple[str, float, int]]:
    full = (1024, 256, 64, 16, 4, 1, 0)
    first = [("initial_repair", weight, average)
             for weight in (1e-6, 1e-5, 1e-3, 1e-2, 0.1)
             for average in full] if hemisphere == "lh" else []
    epochs = [(f"unfold_epoch_{epoch}", weight, average)
              for epoch, weight in ((1, 0.1), (2, 1.0))
              for average in full]
    nonlinear = [("nonlinear_repair", weight, average)
                 for weight in (1e-6, 1e-5, 1e-3, 1e-2, 0.1)
                 for average in (32, 8, 2, 0)]
    return first + epochs + nonlinear


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hemisphere", choices=("lh", "rh"))
    parser.add_argument("inflated", type=Path)
    parser.add_argument("smoothwm", type=Path)
    parser.add_argument("snapshot_prefix", type=Path)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--full-default-prefix", type=int)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--native-log", type=Path)
    parser.add_argument("--resume-from-report", type=Path)
    args = parser.parse_args()
    inflated, faces = fsio.read_geometry(str(args.inflated))
    smoothwm, smoothwm_faces = fsio.read_geometry(str(args.smoothwm))
    if not np.array_equal(faces, smoothwm_faces):
        raise ValueError("inflated and smoothwm face order differs")
    faces = np.asarray(faces, np.int32)
    schedule = _schedule(args.hemisphere)
    native_capture = "threads=4 seed=1234 niterations=1 write_iterations=1 remove_negative=0"
    if args.full_default_prefix is not None:
        if args.full_default_prefix < 1:
            parser.error("--full-default-prefix must be positive")
        stage, weight = (("initial_repair", 1e-6) if args.hemisphere == "lh"
                         else ("unfold_epoch_1", 0.1))
        schedule = [(stage, weight, 1024)] * args.full_default_prefix
        if args.hemisphere == "lh" and args.full_default_prefix >= 4:
            # The frozen -n25 native log switches to navgs=256 after update 2.
            if args.full_default_prefix > 6:
                parser.error("LH full-default schedule is verified only through update 5")
            for index in range(3, args.full_default_prefix):
                schedule[index] = (stage, weight, 256)
        if args.hemisphere == "rh" and args.full_default_prefix > 9:
            parser.error("RH full-default schedule is verified only through update 8")
        native_capture = "threads=4 seed=1234 default niterations=25 write_iterations=1"
    if args.max_steps is not None:
        schedule = schedule[:args.max_steps]
    resume_index = 0
    resume = None
    t0 = time.perf_counter()
    if args.resume_from_report:
        if args.full_default_prefix is None:
            parser.error("--resume-from-report requires --full-default-prefix")
        previous = json.loads(args.resume_from_report.read_text())
        if previous["hemisphere"] != args.hemisphere or not previous["steps"]:
            raise ValueError("resume report has the wrong hemisphere or no steps")
        if previous["input_sha256"] != {"inflated": _sha256(args.inflated),
                                        "smoothwm": _sha256(args.smoothwm)}:
            raise ValueError("resume report has different original inputs")
        if any(step["output_comparison"]["exact_components"] !=
               step["output_comparison"]["total_components"]
               for step in previous["steps"]):
            raise ValueError("resume report contains an unmatched update")
        last = previous["steps"][-1]
        resume_index = last["index"] + 1
        if resume_index >= len(schedule):
            raise ValueError("resume report already reaches the requested boundary")
        checkpoint = Path(f"{args.snapshot_prefix}{resume_index:04d}")
        xyz, checkpoint_faces = fsio.read_geometry(str(checkpoint))
        if not np.array_equal(faces, checkpoint_faces):
            raise ValueError("resume checkpoint has different ordered faces")
        if (_sha256(checkpoint) != last["native_after_sha256"] or
                _coordinate_sha(xyz) != last["python_after_xyz_sha256"]):
            raise ValueError("resume checkpoint differs from proven Python state")
        resume = {"report_sha256": _sha256(args.resume_from_report),
                  "checkpoint_sha256": _sha256(checkpoint),
                  "python_coordinate_sha256": _coordinate_sha(xyz),
                  "next_index": resume_index}
        projection_seconds = 0.0
    else:
        xyz = project_radially(project_before_standard_unfold(inflated), already_sphere=True)
        projection_seconds = time.perf_counter() - t0
    resume_load_seconds = time.perf_counter() - t0 if resume else 0.0
    t0 = time.perf_counter()
    offsets, neighbors, raw, _ = sample_standard_metric_matrix(smoothwm, faces)
    distances, _, _ = average_standard_metric(offsets, neighbors, raw)
    matrix_seconds = time.perf_counter() - t0
    t0 = time.perf_counter()
    local_offsets, local_neighbors, local_distances, old_avg = one_ring_metric(
        faces, len(xyz), offsets, neighbors, distances)
    one_ring_seconds = time.perf_counter() - t0
    original_area, _ = _face_geometry(np.asarray(smoothwm, np.float32), faces)
    original_area = np.abs(original_area)
    original_total = np.float32(np.sum(original_area, dtype=np.float64))
    report = {"reference_commit": "d932c45", "hemisphere": args.hemisphere,
              "native_capture": native_capture,
              "input_sha256": {"inflated": _sha256(args.inflated),
                                "smoothwm": _sha256(args.smoothwm)},
              "native_log_sha256": _sha256(args.native_log) if args.native_log else None,
              "schedule_sha256": hashlib.sha256(json.dumps(schedule).encode()).hexdigest(),
              "implementation_sha256": {
                  "average": _sha256(Path(average_standard_gradient.py_func.__code__.co_filename)),
                  "metric": _sha256(Path(sample_standard_metric_matrix.__code__.co_filename)),
                  "gradient": _sha256(Path(first_epoch_gradient.__code__.co_filename)),
                  "line_search": _sha256(Path(first_epoch_line_search.__code__.co_filename)),
                  "nonlinear": _sha256(Path(nonlinear_epoch_gradient.__code__.co_filename)),
                  "probe": _sha256(Path(__file__))},
              "projection_seconds": projection_seconds,
              "resume_from": resume, "resume_load_seconds": resume_load_seconds,
              "matrix_seconds_including_jit": matrix_seconds,
              "one_ring_seconds": one_ring_seconds, "steps": []}
    for index in range(resume_index, len(schedule)):
        stage, weight, averages = schedule[index]
        before_path = Path(f"{args.snapshot_prefix}{index:04d}")
        after_path = Path(f"{args.snapshot_prefix}{index + 1:04d}")
        if not before_path.exists() or not after_path.exists():
            report["first_unavailable_checkpoint"] = index
            break
        before, before_faces = fsio.read_geometry(str(before_path))
        after, after_faces = fsio.read_geometry(str(after_path))
        if not np.array_equal(faces, before_faces) or not np.array_equal(faces, after_faces):
            raise ValueError("native checkpoint face order differs")
        t0 = time.perf_counter()
        same_integration = (args.full_default_prefix is not None and index > 0
                            and (stage, weight, averages) == schedule[index - 1])
        if index == 0 or same_integration:
            start, reference_start = xyz, before
        else:
            start = project_radially(xyz, already_sphere=True)
            reference_start = project_radially(before, already_sphere=True)
        input_comparison = _compare(start, reference_start)
        step = {"index": index, "stage": stage, "distance_weight": weight,
                "entry_projection_applied": index > 0 and not same_integration,
                "gradient_averages": averages,
                "native_before_sha256": _sha256(before_path),
                "native_after_sha256": _sha256(after_path),
                "python_before_xyz_sha256": _coordinate_sha(start),
                "input_comparison": input_comparison,
                "projection_seconds": time.perf_counter() - t0}
        if input_comparison["exact_components"] != before.size:
            report["first_divergent_step"] = index
            report["first_divergence_phase"] = "input"
            report["steps"].append(step)
            break
        t0 = time.perf_counter()
        if stage == "nonlinear_repair":
            raw_gradient, geometry = nonlinear_epoch_gradient(
                start, faces, smoothwm, local_offsets, local_neighbors,
                local_distances, old_avg, weight)
        else:
            _, _, raw_gradient, geometry = first_epoch_gradient(
                start, faces, smoothwm, offsets, neighbors, distances, weight)
        gradient_seconds = time.perf_counter() - t0
        t0 = time.perf_counter()
        gradient = average_standard_gradient(
            raw_gradient, local_offsets, local_neighbors, averages)
        average_seconds = time.perf_counter() - t0
        t0 = time.perf_counter()
        if stage == "nonlinear_repair":
            search = nonlinear_epoch_line_search(
                start, gradient, faces, local_offsets, local_neighbors,
                local_distances, original_area, original_total, weight)
        else:
            search = first_epoch_line_search(
                start, gradient, faces, offsets, neighbors, distances,
                original_area, original_total, weight)
        search_seconds = time.perf_counter() - t0
        t0 = time.perf_counter()
        shifted = (start.astype(np.float64) + search["selected_dt"]
                   * gradient.astype(np.float64)).astype(np.float32)
        xyz = project_radially(shifted, already_sphere=True)
        update_seconds = time.perf_counter() - t0
        comparison = _compare(xyz, after)
        step.update({"geometry": geometry,
                     "raw_gradient_l2": float(np.linalg.norm(raw_gradient.astype(np.float64))),
                     "averaged_gradient_l2": float(np.linalg.norm(gradient.astype(np.float64))),
                     "line_search": search,
                     "gradient_seconds_including_jit": gradient_seconds,
                     "average_seconds_including_jit": average_seconds,
                     "search_seconds_including_jit": search_seconds,
                     "update_seconds": update_seconds,
                     "python_after_xyz_sha256": _coordinate_sha(xyz),
                     "output_comparison": comparison})
        report["steps"].append(step)
        if comparison["exact_components"] != after.size:
            error = np.linalg.norm(xyz.astype(np.float64) - after.astype(np.float64), axis=1)
            step["largest_vertex_residuals"] = [
                {"vertex": int(v), "error_mm": float(error[v]),
                 "python_after": xyz[v].tolist(), "native_after": after[v].tolist(),
                 "gradient": gradient[v].tolist()}
                for v in np.argsort(error)[-10:][::-1]]
            from scipy.optimize import minimize_scalar

            def fit_error(dt: float) -> float:
                fitted_shift = (start.astype(np.float64) + dt
                                * gradient.astype(np.float64)).astype(np.float32)
                fitted_xyz = project_radially(fitted_shift, already_sphere=True)
                delta = fitted_xyz.astype(np.float64) - after.astype(np.float64)
                return float(np.mean(np.sum(delta * delta, axis=1)))

            chosen = search["selected_dt"]
            fit = minimize_scalar(fit_error, bounds=(0, max(2 * chosen, 1e-6)),
                                  method="bounded", options={"xatol": 1e-9})
            fitted_shift = (start.astype(np.float64) + fit.x
                            * gradient.astype(np.float64)).astype(np.float32)
            fitted_xyz = project_radially(fitted_shift, already_sphere=True)
            step["scalar_dt_fit"] = {"dt": float(fit.x),
                                     "comparison": _compare(fitted_xyz, after)}
            if stage != "nonlinear_repair":
                components = []
                for candidate in search["candidates"]:
                    dt = candidate["dt"]
                    trial = ((start.astype(np.float64) + dt * gradient.astype(np.float64))
                             .astype(np.float32) if dt else start)
                    trial = project_radially(trial, already_sphere=True) if dt else trial
                    components.append(first_epoch_sse(
                        trial, faces, offsets, neighbors, distances,
                        original_area, original_total, weight))
                step["candidate_sse_components"] = components
            report["first_divergent_step"] = index
            report["first_divergence_phase"] = "update"
            break
    output = json.dumps(report, indent=2) + "\n"
    if args.report:
        args.report.write_text(output)
    print(output)


if __name__ == "__main__":
    main()
