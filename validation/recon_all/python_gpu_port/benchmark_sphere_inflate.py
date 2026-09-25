"""Compare the isolated sphere-inflation loop to a native ``after`` snapshot."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from fnit.recon_all.sphere_python import inflate_before_quick_sphere
from benchmark_sphere_projection import area_volume, read_geometry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("native_after", type=Path)
    parser.add_argument("iterations", type=int)
    args = parser.parse_args()
    input_xyz, faces = read_geometry(args.input)
    native_xyz, native_faces = read_geometry(args.native_after)
    start = time.perf_counter()
    python_xyz = inflate_before_quick_sphere(input_xyz, faces.astype(np.int32),
                                             iterations=args.iterations)
    seconds = time.perf_counter() - start
    distance = np.linalg.norm(python_xyz.astype(np.float64) - native_xyz.astype(np.float64), axis=1)
    equal = np.all(python_xyz == native_xyz, axis=1)
    native_area, native_volume = area_volume(native_xyz, native_faces)
    python_area, python_volume = area_volume(python_xyz, faces)
    print(json.dumps({
        "iterations": args.iterations, "python_seconds": seconds,
        "vertices": len(python_xyz), "faces": len(faces),
        "identical_vertex_count": int(equal.sum()),
        "identical_face_count": int(np.all(faces == native_faces, axis=1).sum()),
        "first_different_vertex": int(np.flatnonzero(~equal)[0]) if not equal.all() else None,
        "median_vertex_error_mm": float(np.median(distance)),
        "p99_vertex_error_mm": float(np.percentile(distance, 99)),
        "max_vertex_error_mm": float(distance.max()),
        "native_area_mm2": native_area, "python_area_mm2": python_area,
        "native_enclosed_volume_mm3": native_volume,
        "python_enclosed_volume_mm3": python_volume,
    }, indent=2))


if __name__ == "__main__":
    main()
