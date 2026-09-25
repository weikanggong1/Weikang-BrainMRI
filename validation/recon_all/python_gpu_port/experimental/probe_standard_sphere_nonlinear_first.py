"""Check one native conventional-sphere nonlinear-area update."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import nibabel.freesurfer.io as fsio
import numpy as np

from fnit.recon_all import (
    sphere_standard_line_search as line_module,
    sphere_standard_metric as metric_module,
    sphere_standard_nonlinear as nonlinear_module,
    sphere_standard_unfold as unfold_module,
)
from fnit.recon_all.sphere_python import project_radially
from fnit.recon_all.sphere_standard_metric import (
    average_standard_metric, sample_standard_metric_matrix,
)
from fnit.recon_all.sphere_standard_nonlinear import (
    nonlinear_epoch_gradient, nonlinear_epoch_line_search, one_ring_metric,
)
from fnit.recon_all.sphere_standard_unfold import _face_geometry
from probe_standard_sphere_repair_next import _native_steps, _sha256


def _coordinate_sha(xyz: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(xyz, np.float32).tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("smoothwm", type=Path)
    parser.add_argument("snapshot_prefix", type=Path)
    parser.add_argument("native_verbose", type=Path)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--distance-weight", type=float, default=1e-6)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    before_path = Path(f"{args.snapshot_prefix}{args.index:04d}")
    after_path = Path(f"{args.snapshot_prefix}{args.index + 1:04d}")
    original, faces = fsio.read_geometry(str(args.smoothwm))
    before, before_faces = fsio.read_geometry(str(before_path))
    after, after_faces = fsio.read_geometry(str(after_path))
    if not (np.array_equal(faces, before_faces) and np.array_equal(faces, after_faces)):
        raise ValueError("native snapshot face order differs")
    faces = np.asarray(faces, np.int32)
    start = project_radially(np.asarray(before, np.float32), already_sphere=True)
    t0 = time.perf_counter()
    offsets, ids, raw, _ = sample_standard_metric_matrix(original, faces)
    distances, _, _ = average_standard_metric(offsets, ids, raw)
    local_offsets, local_ids, local_distances, old_average_neighbors = one_ring_metric(
        faces, len(original), offsets, ids, distances)
    matrix_seconds = time.perf_counter() - t0
    original_area, _ = _face_geometry(np.asarray(original, np.float32), faces)
    original_area = np.abs(original_area)
    original_total = np.float32(np.sum(original_area, dtype=np.float64))
    t0 = time.perf_counter()
    gradient, geometry = nonlinear_epoch_gradient(
        start, faces, original, local_offsets, local_ids, local_distances,
        old_average_neighbors, args.distance_weight)
    gradient_seconds = time.perf_counter() - t0
    t0 = time.perf_counter()
    search = nonlinear_epoch_line_search(
        start, gradient, faces, local_offsets, local_ids, local_distances,
        original_area, original_total, args.distance_weight)
    line_search_seconds = time.perf_counter() - t0
    shifted = (start.astype(np.float64) + search["selected_dt"]
               * gradient.astype(np.float64)).astype(np.float32)
    predicted = project_radially(shifted, already_sphere=True)
    error = np.linalg.norm(predicted.astype(np.float64) - after.astype(np.float64), axis=1)
    lengths = np.linalg.norm(gradient.astype(np.float64), axis=1)
    native = _native_steps(args.native_verbose)[args.index]
    report = {"reference_commit": "d932c45",
              "index": args.index, "distance_weight": args.distance_weight,
              "input_sha256": {"smoothwm": _sha256(args.smoothwm),
                                "native_before": _sha256(before_path),
                                "native_after": _sha256(after_path),
                                "native_verbose": _sha256(args.native_verbose)},
              "implementation_sha256": {
                  "metric": _sha256(Path(metric_module.__file__)),
                  "unfold": _sha256(Path(unfold_module.__file__)),
                  "line_search": _sha256(Path(line_module.__file__)),
                  "nonlinear": _sha256(Path(nonlinear_module.__file__)),
                  "probe": _sha256(Path(__file__))},
              "python_before_xyz_sha256": _coordinate_sha(start),
              "python_after_xyz_sha256": _coordinate_sha(predicted),
              "matrix_seconds_including_jit": matrix_seconds,
              "full_metric_entries": len(ids),
              "one_ring_entries": len(local_ids),
              "gradient_seconds_including_jit": gradient_seconds,
              "line_search_seconds_including_jit": line_search_seconds,
              "geometry": geometry,
              "gradient": {"l2": float(np.linalg.norm(lengths)),
                           "max": float(lengths.max()),
                           "mean": float(lengths.mean())},
              "native": native, "line_search": search,
              "comparison": {"exact_components": int(np.count_nonzero(predicted == after)),
                             "total_components": int(after.size),
                             "vertices_le_1e-5_mm": int(np.count_nonzero(error <= 1e-5)),
                             "vertices": len(error),
                             "max_error_mm": float(error.max()),
                             "rms_error_mm": float(np.sqrt(np.mean(error * error))),
                             "worst_vertex": int(np.argmax(error))}}
    output = json.dumps(report, indent=2)
    if args.report:
        args.report.write_text(output + "\n")
    print(output)


if __name__ == "__main__":
    main()
