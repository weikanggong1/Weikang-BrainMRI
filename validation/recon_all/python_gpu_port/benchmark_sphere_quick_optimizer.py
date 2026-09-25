"""Benchmark the experimental Python q-sphere optimizer on native projected input."""

from __future__ import annotations

import argparse
from time import perf_counter

import nibabel.freesurfer.io as fsio
import numpy as np

from fnit.recon_all.sphere_quick_python import (
    quick_sphere_from_projected,
    reference_face_areas,
)


def _area_volume(vertices: np.ndarray, faces: np.ndarray) -> tuple[float, float]:
    a = vertices[faces[:, 0]].astype(np.float64)
    b = vertices[faces[:, 1]].astype(np.float64)
    c = vertices[faces[:, 2]].astype(np.float64)
    area = float(np.sum(np.linalg.norm(np.cross(b - a, c - a), axis=1) * .5))
    volume = float(np.sum(np.einsum("ij,ij->i", a, np.cross(b, c))) / 6)
    return area, volume


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_inflated")
    parser.add_argument("native_projected")
    parser.add_argument("native_final")
    parser.add_argument("--iterations", type=int, default=25,
                        help="25 for fixed recon-all; 1 for isolated native -N 1 diagnostic")
    parser.add_argument("--initial-momentum", type=str,
                        help="Native diagnostic state for a conditional optimizer-only replay")
    args = parser.parse_args()

    input_xyz, faces = fsio.read_geometry(args.input_inflated)
    projected, projected_faces = fsio.read_geometry(args.native_projected)
    native, native_faces = fsio.read_geometry(args.native_final)
    assert np.array_equal(faces, projected_faces) and np.array_equal(faces, native_faces)
    reference = reference_face_areas(input_xyz, faces)
    momentum = (None if args.initial_momentum is None else
                np.fromfile(args.initial_momentum, dtype="<f4").reshape(-1, 3))
    start = perf_counter()
    predicted, trace = quick_sphere_from_projected(
        projected, faces, reference, float(np.sum(reference, dtype=np.float64)),
        niterations=args.iterations, initial_momentum=momentum)
    seconds = perf_counter() - start

    residual = np.linalg.norm(predicted.astype(np.float64) - native.astype(np.float64), axis=1)
    print("vertices", len(projected), "ordered_faces", len(faces), "python_seconds", seconds)
    print("updates", len(trace), "updates_by_k",
          {k: sum(row[0] == k for row in trace) for k in (10, 40, 160, 640)})
    print("exact_vertices", int(np.count_nonzero(np.all(predicted == native, axis=1))),
          "median_mm", float(np.median(residual)), "p95_mm", float(np.percentile(residual, 95)),
          "max_mm", float(np.max(residual)))
    print("python_area_volume", _area_volume(predicted, faces),
          "native_area_volume", _area_volume(native, faces))


if __name__ == "__main__":
    main()
