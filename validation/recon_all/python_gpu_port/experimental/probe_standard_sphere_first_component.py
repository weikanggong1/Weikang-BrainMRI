"""Compare native-style rounding variants at the first LH sphere repair update."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import nibabel.freesurfer.io as fsio
import numpy as np
from numba import njit

from fnit.recon_all.sphere_python import project_radially
from fnit.recon_all.sphere_standard_line_search import first_epoch_line_search
from fnit.recon_all.sphere_standard_metric import (
    average_standard_metric, sample_standard_metric_matrix,
)
from fnit.recon_all.sphere_standard_python import project_before_standard_unfold
from fnit.recon_all.sphere_standard_unfold import _face_geometry, first_epoch_gradient
from probe_standard_sphere_repair_next import _sha256


@njit
def _scalar_projection(xyz: np.ndarray) -> np.ndarray:
    result = np.empty_like(xyz)
    for vertex in range(len(xyz)):
        x = np.float64(xyz[vertex, 0])
        y = np.float64(xyz[vertex, 1])
        z = np.float64(xyz[vertex, 2])
        distance = math.sqrt(x * x + y * y + z * z)
        d = 0.0 if distance < 1.1920928955078125e-7 else 1.0 - 100.0 / distance
        result[vertex, 0] = np.float32(x - d * x)
        result[vertex, 1] = np.float32(y - d * y)
        result[vertex, 2] = np.float32(z - d * z)
    return result


def _compare(predicted: np.ndarray, native: np.ndarray) -> dict:
    error = np.linalg.norm(predicted.astype(np.float64) - native.astype(np.float64), axis=1)
    locations = np.argwhere(predicted != native)
    return {"exact_components": int(np.count_nonzero(predicted == native)),
            "vertices_le_1e-5_mm": int(np.count_nonzero(error <= 1e-5)),
            "max_error_mm": float(error.max()),
            "mismatched_components": [
                {"vertex": int(v), "component": int(c),
                 "predicted": float(predicted[v, c]), "native": float(native[v, c])}
                for v, c in locations[:30]]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inflated", type=Path)
    parser.add_argument("smoothwm", type=Path)
    parser.add_argument("native_before", type=Path)
    parser.add_argument("native_after", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    inflated, faces = fsio.read_geometry(str(args.inflated))
    original, original_faces = fsio.read_geometry(str(args.smoothwm))
    before, before_faces = fsio.read_geometry(str(args.native_before))
    after, after_faces = fsio.read_geometry(str(args.native_after))
    if not (np.array_equal(faces, original_faces) and
            np.array_equal(faces, before_faces) and np.array_equal(faces, after_faces)):
        raise ValueError("surface face order differs")
    faces = np.asarray(faces, np.int32)
    start = project_radially(project_before_standard_unfold(inflated), already_sphere=True)
    if not np.array_equal(start, before):
        raise ValueError("initial Python sphere coordinates differ from native")
    offsets, ids, raw, _ = sample_standard_metric_matrix(original, faces)
    distances, _, _ = average_standard_metric(offsets, ids, raw)
    orig_area, _ = _face_geometry(np.asarray(original, np.float32), faces)
    orig_area = np.abs(orig_area)
    original_total = np.float32(np.sum(orig_area, dtype=np.float64))
    _, _, gradient, _ = first_epoch_gradient(
        start, faces, original, offsets, ids, distances, 1e-6)
    search = first_epoch_line_search(
        start, gradient, faces, offsets, ids, distances,
        orig_area, original_total, 1e-6)
    dt = np.float32(search["selected_dt"])
    variants = {}
    for direction, trial_dt in [
        ("minus_ulp", float(np.nextafter(dt, np.float32(0)))),
        ("selected", float(dt)),
        ("plus_ulp", float(np.nextafter(dt, np.float32(np.inf))))]:
        shifted = (start.astype(np.float64)
                   + trial_dt * gradient.astype(np.float64)).astype(np.float32)
        variants[direction + "_vector_projection"] = _compare(
            project_radially(shifted, already_sphere=True), after)
        variants[direction + "_scalar_projection"] = _compare(
            _scalar_projection(shifted), after)
    shifted_float = np.float32(start + np.float32(dt * gradient))
    variants["selected_float_multiply_add"] = _compare(
        _scalar_projection(shifted_float), after)
    result = {"source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "input_sha256": {"inflated": _sha256(args.inflated),
                                "smoothwm": _sha256(args.smoothwm),
                                "native_before": _sha256(args.native_before),
                                "native_after": _sha256(args.native_after)},
              "selected_dt": float(dt), "variants": variants}
    output = json.dumps(result, indent=2)
    if args.report:
        args.report.write_text(output + "\n")
    print(output)


if __name__ == "__main__":
    main()
