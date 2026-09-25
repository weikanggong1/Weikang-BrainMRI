"""Compare first quick-sphere Python gradient with a passive native GDB capture."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import nibabel.freesurfer as fs
import numpy as np

from fnit.recon_all.inflate_python import matrix
from fnit.recon_all.smooth_surface_python import ordered_neighbors
from fnit.recon_all.sphere_quick_python import (
    average_gradients, nonlinear_area_gradient, reference_face_areas,
)


def mean_magnitude(values: np.ndarray) -> float:
    squared = values[:, 0] * values[:, 0]
    squared += values[:, 1] * values[:, 1]
    squared += values[:, 2] * values[:, 2]
    return sum(math.sqrt(float(value)) for value in squared) / len(squared)


def average_with_reciprocal(gradient: np.ndarray, faces: np.ndarray,
                            repeats: int) -> np.ndarray:
    current = gradient.copy()
    indices, degree = matrix(ordered_neighbors(faces, len(current)))
    reciprocal = np.float32(1.0) / (degree + 1).astype(np.float32)
    for _ in range(repeats):
        updated = current.copy()
        for slot in range(indices.shape[1]):
            valid = degree > slot
            updated[valid] += current[indices[valid, slot]]
        current = updated * reciprocal[:, None]
    return current


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_surface", type=Path)
    parser.add_argument("native_positions", type=Path)
    parser.add_argument("native_gradient", type=Path)
    parser.add_argument("--native-before-average", type=Path)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, faces = fs.read_geometry(args.input_surface)
    before = np.fromfile(args.native_positions, dtype="<f4").reshape(-1, 3)
    native = np.fromfile(args.native_gradient, dtype="<f4").reshape(-1, 3)
    assert len(before) == len(native) == len(source)
    reference_areas = reference_face_areas(source, faces)
    total_area = float(np.sum(reference_areas, dtype=np.float64))
    candidate, sse, negative = nonlinear_area_gradient(before, faces, reference_areas,
                                                        total_area, 10)
    raw_candidate = candidate.copy()
    reciprocal_candidate = average_with_reciprocal(candidate, faces, 128)
    candidate = average_gradients(candidate, faces, 128)
    diff = candidate.astype(np.float64) - native.astype(np.float64)
    reciprocal_diff = reciprocal_candidate.astype(np.float64) - native.astype(np.float64)
    report = {
        "vertices": len(before),
        "faces": len(faces),
        "negative_faces": negative,
        "python_start_sse": sse,
        "native_gradient_mean_magnitude": mean_magnitude(native),
        "python_gradient_mean_magnitude": mean_magnitude(candidate),
        "exact_gradient_values": int(np.count_nonzero(candidate == native)),
        "total_gradient_values": int(candidate.size),
        "max_absolute_gradient_error": float(np.max(np.abs(diff))),
        "mean_absolute_gradient_error": float(np.mean(np.abs(diff))),
        "first_native_gradient": native[0].tolist(),
        "first_python_gradient": candidate[0].tolist(),
        "reciprocal_exact_gradient_values": int(np.count_nonzero(reciprocal_candidate == native)),
        "reciprocal_max_absolute_gradient_error": float(np.max(np.abs(reciprocal_diff))),
        "reciprocal_mean_absolute_gradient_error": float(np.mean(np.abs(reciprocal_diff))),
        "reciprocal_mean_magnitude": mean_magnitude(reciprocal_candidate),
        "first_reciprocal_gradient": reciprocal_candidate[0].tolist(),
    }
    if args.native_before_average:
        native_raw = np.fromfile(args.native_before_average, dtype="<f4").reshape(-1, 3)
        assert native_raw.shape == native.shape
        raw_diff = raw_candidate.astype(np.float64) - native_raw.astype(np.float64)
        native_raw_averaged = average_with_reciprocal(native_raw, faces, 128)
        source_avg_diff = native_raw_averaged.astype(np.float64) - native.astype(np.float64)
        report.update({
            "raw_exact_gradient_values": int(np.count_nonzero(raw_candidate == native_raw)),
            "raw_max_absolute_error": float(np.max(np.abs(raw_diff))),
            "raw_mean_absolute_error": float(np.mean(np.abs(raw_diff))),
            "first_native_raw": native_raw[0].tolist(),
            "first_python_raw": raw_candidate[0].tolist(),
            "native_raw_reaveraged_exact_values": int(np.count_nonzero(native_raw_averaged == native)),
            "native_raw_reaveraged_max_absolute_error": float(np.max(np.abs(source_avg_diff))),
            "native_raw_reaveraged_mean_magnitude": mean_magnitude(native_raw_averaged),
        })
    if args.snapshot:
        snapshot, snapshot_faces = fs.read_geometry(args.snapshot)
        report["native_positions_match_snapshot"] = bool(np.array_equal(before, snapshot))
        report["faces_match_snapshot"] = bool(np.array_equal(faces, snapshot_faces))
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
