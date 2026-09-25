"""Benchmark the native-free fixed ``mris_sphere -q`` geometry chain."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import nibabel.freesurfer as fs
import numpy as np

from fnit.recon_all.sphere_quick_python import quick_sphere_from_inflated


def area_volume(vertices: np.ndarray, faces: np.ndarray) -> tuple[float, float]:
    a = vertices[faces[:, 0]].astype(np.float64)
    b = vertices[faces[:, 1]].astype(np.float64)
    c = vertices[faces[:, 2]].astype(np.float64)
    area = float(np.sum(np.linalg.norm(np.cross(b - a, c - a), axis=1) * .5))
    volume = float(np.sum(np.einsum("ij,ij->i", a, np.cross(b, c))) / 6)
    return area, volume


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_inflated", type=Path)
    parser.add_argument("native_final", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    original, faces = fs.read_geometry(args.input_inflated)
    native, native_faces = fs.read_geometry(args.native_final)
    if not np.array_equal(faces, native_faces):
        raise ValueError("native and Python input ordered faces differ")
    start = perf_counter()
    predicted, trace = quick_sphere_from_inflated(original, faces)
    seconds = perf_counter() - start
    residual = np.linalg.norm(predicted.astype(np.float64) - native.astype(np.float64), axis=1)
    report = {
        "vertices": len(original),
        "ordered_faces": len(faces),
        "python_cpu_seconds": seconds,
        "updates": len(trace),
        "updates_by_k": {str(k): sum(row[0] == k for row in trace) for k in (10, 40, 160, 640)},
        "exact_vertices": int(np.count_nonzero(np.all(predicted == native, axis=1))),
        "median_vertex_error_mm": float(np.median(residual)),
        "p95_vertex_error_mm": float(np.percentile(residual, 95)),
        "max_vertex_error_mm": float(np.max(residual)),
        "python_area_volume": area_volume(predicted, faces),
        "native_area_volume": area_volume(native, faces),
    }
    data = json.dumps(report, indent=2) + "\n"
    if args.output is not None:
        args.output.write_text(data)
    print(data)


if __name__ == "__main__":
    main()
