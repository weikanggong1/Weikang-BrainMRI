"""Propagate conventional sphere checkpoints from inflated without native state injection."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import nibabel.freesurfer.io as fsio
import numpy as np

from fnit.recon_all import (
    sphere_standard_line_search as line_search_module,
    sphere_standard_metric as metric_module,
    sphere_standard_nonlinear as nonlinear_module,
    sphere_standard_python as projection_module,
    sphere_standard_unfold as unfold_module,
)
from fnit.recon_all.sphere_python import project_radially
from fnit.recon_all.sphere_standard_line_search import first_epoch_line_search
from fnit.recon_all.sphere_standard_metric import (
    average_standard_metric, sample_standard_metric_matrix,
)
from fnit.recon_all.sphere_standard_python import (
    project_before_standard_unfold, write_standard_sphere_surface,
)
from fnit.recon_all.sphere_standard_nonlinear import (
    nonlinear_epoch_gradient, nonlinear_epoch_line_search, one_ring_metric,
)
from fnit.recon_all.sphere_standard_unfold import _face_geometry, first_epoch_gradient
from probe_standard_sphere_repair_next import _native_steps, _sha256


def _coordinate_sha(xyz: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(xyz, np.float32).tobytes()).hexdigest()


def _volume_info_values(info: dict) -> dict:
    return {key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in info.items()}


def _compare(xyz: np.ndarray, native: np.ndarray) -> dict:
    error = np.linalg.norm(xyz.astype(np.float64) - native.astype(np.float64), axis=1)
    return {"exact_components": int(np.count_nonzero(xyz == native)),
            "total_components": int(native.size),
            "vertices_le_1e-5_mm": int(np.count_nonzero(error <= 1e-5)),
            "vertices": len(error),
            "max_error_mm": float(error.max()),
            "rms_error_mm": float(np.sqrt(np.mean(error * error))),
            "worst_vertex": int(np.argmax(error))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inflated", type=Path)
    parser.add_argument("smoothwm", type=Path)
    parser.add_argument("snapshot_prefix", type=Path)
    parser.add_argument("native_verbose", type=Path)
    parser.add_argument("--step", action="append", required=True,
                        help="ordered index:distance weight, e.g. 0:1e-6")
    parser.add_argument("--nonlinear-next", action="store_true",
                        help="check the first one-ring nonlinear-area update after the ordered steps")
    parser.add_argument("--nonlinear-weight", action="append", type=float,
                        help="ordered nonlinear-area distance weight, beginning at 1e-6")
    parser.add_argument("--native-final-surface", type=Path,
                        help="check the isolated command output after its final radial projection")
    parser.add_argument("--python-final-surface", type=Path,
                        help="write a Python surface using the inflated input volume geometry")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    specs = [(int(index), float(weight)) for index, weight in
             (spec.split(":", 1) for spec in args.step)]
    if [index for index, _ in specs] != list(range(len(specs))):
        raise ValueError("continuous steps must start at 0 with no gaps")
    inflated, faces, source_volume_info = fsio.read_geometry(
        str(args.inflated), read_metadata=True)
    smoothwm, smoothwm_faces = fsio.read_geometry(str(args.smoothwm))
    if not np.array_equal(faces, smoothwm_faces):
        raise ValueError("inflated and smoothwm face order differs")
    faces = np.asarray(faces, np.int32)
    t0 = time.perf_counter()
    xyz = project_before_standard_unfold(inflated)
    # MRISintegrate projects once more before its first saved state.
    xyz = project_radially(xyz, already_sphere=True)
    projection_seconds = time.perf_counter() - t0
    t0 = time.perf_counter()
    offsets, ids, raw, _ = sample_standard_metric_matrix(smoothwm, faces)
    distances, _, _ = average_standard_metric(offsets, ids, raw)
    metric_seconds = time.perf_counter() - t0
    original_face_area, _ = _face_geometry(np.asarray(smoothwm, np.float32), faces)
    original_face_area = np.abs(original_face_area)
    original_total = np.float32(np.sum(original_face_area, dtype=np.float64))
    native = _native_steps(args.native_verbose)
    report = {"reference_commit": "d932c45",
              "input_sha256": {"inflated": _sha256(args.inflated),
                                "smoothwm": _sha256(args.smoothwm),
                                "native_verbose": _sha256(args.native_verbose)},
              "requested_step_specs": args.step,
              "requested_step_specs_sha256": hashlib.sha256(
                  "\n".join(args.step).encode()).hexdigest(),
              "implementation_sha256": {
                  "metric": _sha256(Path(metric_module.__file__)),
                  "nonlinear": _sha256(Path(nonlinear_module.__file__)),
                  "unfold": _sha256(Path(unfold_module.__file__)),
                  "line_search": _sha256(Path(line_search_module.__file__)),
                  "projection": _sha256(Path(projection_module.__file__)),
                  "probe": _sha256(Path(__file__))},
              "projection_seconds": projection_seconds,
              "matrix_seconds_including_jit": metric_seconds,
              "steps": []}
    for index, weight in specs:
        before_path = Path(f"{args.snapshot_prefix}{index:04d}")
        after_path = Path(f"{args.snapshot_prefix}{index + 1:04d}")
        before, before_faces = fsio.read_geometry(str(before_path))
        after, after_faces = fsio.read_geometry(str(after_path))
        if not np.array_equal(faces, before_faces) or not np.array_equal(faces, after_faces):
            raise ValueError("native checkpoint face order differs")
        t0 = time.perf_counter()
        start = xyz if index == 0 else project_radially(xyz, already_sphere=True)
        pre_projection_seconds = time.perf_counter() - t0
        reference_start = before if index == 0 else project_radially(before, already_sphere=True)
        input_compare = _compare(start, reference_start)
        step = {"index": index, "distance_weight": weight,
                "native_before_sha256": _sha256(before_path),
                "native_after_sha256": _sha256(after_path),
                "python_before_xyz_sha256": _coordinate_sha(start),
                "pre_projection_seconds": pre_projection_seconds,
                "input_compare": input_compare}
        if input_compare["max_error_mm"] > 1e-5:
            report["first_divergent_step"] = index
            report["first_divergence_phase"] = "input"
            report["steps"].append(step)
            break
        t0 = time.perf_counter()
        _, _, gradient, geometry = first_epoch_gradient(
            start, faces, smoothwm, offsets, ids, distances, weight)
        gradient_seconds = time.perf_counter() - t0
        t0 = time.perf_counter()
        search = first_epoch_line_search(start, gradient, faces, offsets, ids,
                                         distances, original_face_area, original_total, weight)
        line_search_seconds = time.perf_counter() - t0
        t0 = time.perf_counter()
        shifted = (start.astype(np.float64) + search["selected_dt"]
                   * gradient.astype(np.float64)).astype(np.float32)
        xyz = project_radially(shifted, already_sphere=True)
        update_seconds = time.perf_counter() - t0
        output_compare = _compare(xyz, after)
        reference = native[index]
        decision_matches = (search["selected_index"] == reference["selected_index"]
                            and f"{search['selected_dt']:.3f}"
                            == f"{reference['selected_dt_printed']:.3f}")
        step.update({"geometry": geometry, "native": reference,
                     "line_search": search,
                     "gradient_seconds_including_jit": gradient_seconds,
                     "line_search_seconds_including_jit": line_search_seconds,
                     "update_seconds": update_seconds,
                     "python_after_xyz_sha256": _coordinate_sha(xyz),
                     "output_compare": output_compare,
                     "decision_matches_native": decision_matches})
        report["steps"].append(step)
        if not decision_matches or output_compare["max_error_mm"] > 1e-5:
            error = np.linalg.norm(xyz.astype(np.float64) - after.astype(np.float64), axis=1)
            input_error = np.linalg.norm(start.astype(np.float64)
                                         - reference_start.astype(np.float64), axis=1)
            largest = np.argsort(error)[-10:][::-1]
            _, _, native_gradient, _ = first_epoch_gradient(
                reference_start, faces, smoothwm, offsets, ids, distances, weight)
            native_search = first_epoch_line_search(
                reference_start, native_gradient, faces, offsets, ids, distances,
                original_face_area, original_total, weight)
            native_shifted = (reference_start.astype(np.float64)
                              + native_search["selected_dt"]
                              * native_gradient.astype(np.float64)).astype(np.float32)
            native_predicted = project_radially(native_shifted, already_sphere=True)
            step["first_divergence_diagnostic"] = {
                "largest_vertex_residuals": [
                    {"vertex": int(v), "output_error_mm": float(error[v]),
                     "input_error_mm": float(input_error[v]),
                     "python_after": xyz[v].tolist(), "native_after": after[v].tolist(),
                     "gradient_delta": (gradient[v] - native_gradient[v]).tolist()}
                    for v in largest],
                "native_input_counterfactual": {
                    "selected_index": native_search["selected_index"],
                    "selected_dt": native_search["selected_dt"],
                    "output_compare": _compare(native_predicted, after)},
            }
            report["first_divergent_step"] = index
            report["first_divergence_phase"] = "update"
            break
    nonlinear_weights = args.nonlinear_weight or ([1e-6] if args.nonlinear_next else [])
    if nonlinear_weights and nonlinear_weights[0] != 1e-6:
        raise ValueError("nonlinear sequence must begin with the first 1e-6 fold repair")
    report["nonlinear_weights"] = nonlinear_weights
    for nonlinear_index, weight in enumerate(nonlinear_weights):
        if len(report["steps"]) != len(specs) + nonlinear_index or "first_divergent_step" in report:
            break
        index = len(specs) + nonlinear_index
        before_path = Path(f"{args.snapshot_prefix}{index:04d}")
        after_path = Path(f"{args.snapshot_prefix}{index + 1:04d}")
        before, before_faces = fsio.read_geometry(str(before_path))
        after, after_faces = fsio.read_geometry(str(after_path))
        if not np.array_equal(faces, before_faces) or not np.array_equal(faces, after_faces):
            raise ValueError("native nonlinear checkpoint face order differs")
        t0 = time.perf_counter()
        start = project_radially(xyz, already_sphere=True)
        reference_start = project_radially(before, already_sphere=True)
        input_compare = _compare(start, reference_start)
        if nonlinear_index == 0:
            local_offsets, local_ids, local_distances, old_average_neighbors = one_ring_metric(
                faces, len(xyz), offsets, ids, distances)
        step = {"index": index, "stage": "nonlinear_area_fold_repair",
                "distance_weight": weight,
                "native_before_sha256": _sha256(before_path),
                "native_after_sha256": _sha256(after_path),
                "python_before_xyz_sha256": _coordinate_sha(start),
                "input_compare": input_compare,
                "full_metric_entries": len(ids), "one_ring_entries": len(local_ids),
                "old_average_neighbors": float(old_average_neighbors),
                "stage_setup_seconds": time.perf_counter() - t0}
        if input_compare["max_error_mm"] <= 1e-5:
            t0 = time.perf_counter()
            gradient, geometry = nonlinear_epoch_gradient(
                start, faces, smoothwm, local_offsets, local_ids,
                local_distances, old_average_neighbors, weight)
            gradient_seconds = time.perf_counter() - t0
            lengths = np.linalg.norm(gradient.astype(np.float64), axis=1)
            t0 = time.perf_counter()
            search = nonlinear_epoch_line_search(
                start, gradient, faces, local_offsets, local_ids,
                local_distances, original_face_area, original_total, weight)
            search_seconds = time.perf_counter() - t0
            t0 = time.perf_counter()
            shifted = (start.astype(np.float64) + search["selected_dt"]
                       * gradient.astype(np.float64)).astype(np.float32)
            xyz = project_radially(shifted, already_sphere=True)
            update_seconds = time.perf_counter() - t0
            output_compare = _compare(xyz, after)
            reference = native[index]
            decision_matches = (search["selected_index"] == reference["selected_index"]
                                and f"{search['selected_dt']:.3f}"
                                == f"{reference['selected_dt_printed']:.3f}")
            step.update({"geometry": geometry, "native": reference,
                         "gradient": {"l2": float(np.linalg.norm(lengths)),
                                      "max": float(lengths.max()),
                                      "mean": float(lengths.mean())},
                         "line_search": search,
                         "gradient_seconds_including_jit": gradient_seconds,
                         "line_search_seconds_including_jit": search_seconds,
                         "update_seconds": update_seconds,
                         "python_after_xyz_sha256": _coordinate_sha(xyz),
                         "output_compare": output_compare,
                         "decision_matches_native": decision_matches})
            if not decision_matches or output_compare["max_error_mm"] > 1e-5:
                errors = np.linalg.norm(xyz.astype(np.float64) - after.astype(np.float64), axis=1)
                step["largest_vertex_residuals"] = [
                    {"vertex": int(v), "error_mm": float(errors[v]),
                     "python_after": xyz[v].tolist(), "native_after": after[v].tolist(),
                     "gradient": gradient[v].tolist()}
                    for v in np.argsort(errors)[-10:][::-1]]
                report["first_divergent_step"] = index
                report["first_divergence_phase"] = "nonlinear_update"
        else:
            report["first_divergent_step"] = index
            report["first_divergence_phase"] = "nonlinear_input"
        report["steps"].append(step)
    if (args.native_final_surface and "first_divergent_step" not in report
            and len(report["steps"]) == len(specs) + len(nonlinear_weights)):
        final_xyz, final_faces, native_volume_info = fsio.read_geometry(
            str(args.native_final_surface), read_metadata=True)
        if not np.array_equal(faces, final_faces):
            raise ValueError("native final surface face order differs")
        t0 = time.perf_counter()
        predicted_final = project_radially(xyz, already_sphere=True)
        source_info = _volume_info_values(source_volume_info)
        native_info = _volume_info_values(native_volume_info)
        report["final_surface"] = {
            "native_sha256": _sha256(args.native_final_surface),
            "native_xyz_sha256": _coordinate_sha(final_xyz),
            "python_xyz_sha256": _coordinate_sha(predicted_final),
            "native_faces_sha256": hashlib.sha256(np.ascontiguousarray(final_faces).tobytes()).hexdigest(),
            "source_faces_sha256": hashlib.sha256(np.ascontiguousarray(faces).tobytes()).hexdigest(),
            "ordered_faces_exact": bool(np.array_equal(faces, final_faces)),
            "native_volume_info": native_info,
            "volume_info_matches_input": native_info == source_info,
            "comparison": _compare(predicted_final, final_xyz),
            "projection_seconds": time.perf_counter() - t0}
        if args.python_final_surface:
            write_standard_sphere_surface(args.python_final_surface, predicted_final,
                                          faces, args.inflated)
            written_xyz, written_faces, written_info = fsio.read_geometry(
                str(args.python_final_surface), read_metadata=True)
            report["final_surface"]["python_file"] = {
                "sha256": _sha256(args.python_final_surface),
                "xyz_sha256": _coordinate_sha(written_xyz),
                "faces_sha256": hashlib.sha256(np.ascontiguousarray(written_faces).tobytes()).hexdigest(),
                "ordered_faces_exact": bool(np.array_equal(written_faces, final_faces)),
                "volume_info_matches_native": _volume_info_values(written_info) == native_info,
                "comparison": _compare(written_xyz, final_xyz)}
        if report["final_surface"]["comparison"]["max_error_mm"] > 1e-5:
            report["first_divergence_phase"] = "final_projection"
    output = json.dumps(report, indent=2)
    if args.report:
        args.report.write_text(output + "\n")
    print(output)


if __name__ == "__main__":
    main()
