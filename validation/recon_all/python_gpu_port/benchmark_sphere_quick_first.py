"""Compare the isolated Python q-sphere first update with native snapshots."""

from __future__ import annotations

import argparse
from time import perf_counter

import nibabel.freesurfer.io as fsio
import numpy as np
from scipy.optimize import minimize_scalar

from fnit.recon_all.sphere_quick_python import (
    _line_minimize,
    average_gradients,
    nonlinear_area_gradient,
    projected_step,
    reference_face_areas,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_surface")
    parser.add_argument("native_snapshot_0")
    parser.add_argument("native_snapshot_1")
    parser.add_argument("--native-dt", type=float, required=True)
    args = parser.parse_args()

    original, faces = fsio.read_geometry(args.input_surface)
    before, before_faces = fsio.read_geometry(args.native_snapshot_0)
    after, after_faces = fsio.read_geometry(args.native_snapshot_1)
    assert np.array_equal(faces, before_faces) and np.array_equal(faces, after_faces)

    start = perf_counter()
    reference = reference_face_areas(original, faces)
    original_total = float(np.sum(reference, dtype=np.float64))
    gradient, sse, negative = nonlinear_area_gradient(before, faces, reference, original_total, 10)
    gradient_seconds = perf_counter() - start
    start = perf_counter()
    gradient = average_gradients(gradient, faces, 128)
    averaging_seconds = perf_counter() - start

    def error_at(dt: float) -> float:
        delta = projected_step(before, gradient, dt).astype(np.float64) - after.astype(np.float64)
        return float(np.mean(np.sum(delta * delta, axis=1)))

    fitted = minimize_scalar(error_at, bounds=(args.native_dt * .8, args.native_dt * 1.2),
                             method="bounded", options={"xatol": 1e-4})
    independent_dt = _line_minimize(before, gradient, faces, original_total, 10)
    for label, dt in (("independent", independent_dt), ("native", args.native_dt),
                      ("fitted", fitted.x)):
        predicted = projected_step(before, gradient, dt)
        residual = np.linalg.norm(predicted.astype(np.float64) - after.astype(np.float64), axis=1)
        exact = int(np.count_nonzero(np.all(predicted == after, axis=1)))
        print(label, "dt", dt, "exact_vertices", exact, "median_mm", np.median(residual),
              "p95_mm", np.percentile(residual, 95), "max_mm", np.max(residual))
    print("vertices", len(before), "faces", len(faces), "negative_faces", negative,
          "reference_total_mm2", original_total, "python_initial_sse", sse)
    print("python_gradient_seconds", gradient_seconds, "python_128_averages_seconds", averaging_seconds)


if __name__ == "__main__":
    main()
