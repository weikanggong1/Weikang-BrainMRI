"""Compare default mris_sphere's frozen pre-unfold projection with Python."""

from __future__ import annotations

import argparse
import json
from time import perf_counter

import nibabel.freesurfer.io as fsio
import numpy as np

from fnit.recon_all.sphere_standard_python import project_before_standard_unfold
from fnit.recon_all.sphere_python import initial_scale


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inflated")
    parser.add_argument("native_after")
    parser.add_argument("--native-before")
    parser.add_argument("--native-final")
    args = parser.parse_args()
    vertices, faces = fsio.read_geometry(args.inflated)
    native, native_faces = fsio.read_geometry(args.native_after)
    started = perf_counter()
    predicted = project_before_standard_unfold(vertices)
    seconds = perf_counter() - started
    distance = np.linalg.norm(predicted.astype(np.float64) - native.astype(np.float64), axis=1)
    result = {
        "vertices": len(vertices),
        "faces": len(faces),
        "ordered_faces_equal": bool(np.array_equal(faces, native_faces)),
        "exact_vertices": int(np.count_nonzero(np.all(predicted == native, axis=1))),
        "distance_median_mm": float(np.median(distance)),
        "distance_p95_mm": float(np.quantile(distance, .95)),
        "distance_max_mm": float(np.max(distance)),
        "python_seconds": seconds,
    }
    if args.native_before:
        before, before_faces = fsio.read_geometry(args.native_before)
        result["before_exact_vertices"] = int(np.count_nonzero(
            np.all(initial_scale(vertices) == before, axis=1)))
        result["before_ordered_faces_equal"] = bool(np.array_equal(faces, before_faces))
    if args.native_final:
        final, final_faces = fsio.read_geometry(args.native_final)
        final_distance = np.linalg.norm(predicted.astype(np.float64) - final.astype(np.float64), axis=1)
        result["final_exact_vertices"] = int(np.count_nonzero(np.all(predicted == final, axis=1)))
        result["final_ordered_faces_equal"] = bool(np.array_equal(faces, final_faces))
        result["final_distance_median_mm"] = float(np.median(final_distance))
        result["final_distance_p95_mm"] = float(np.quantile(final_distance, .95))
        result["final_distance_max_mm"] = float(np.max(final_distance))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
