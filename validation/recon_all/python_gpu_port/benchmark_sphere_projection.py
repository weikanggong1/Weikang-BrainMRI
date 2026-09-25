"""Paired native/Python check for mris_sphere's pre-optimization projection."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from fnit.recon_all.sphere_python import project_surface_before_quick_sphere


def read_geometry(path):
    data = path.read_bytes()
    start = data.index(b"\n\n", 3) + 2
    nvertices = int.from_bytes(data[start:start + 4], "big")
    nfaces = int.from_bytes(data[start + 4:start + 8], "big")
    coordinates = np.frombuffer(data[start + 8:start + 8 + 12 * nvertices],
                                dtype=">f4").reshape(-1, 3)
    face_start = start + 8 + 12 * nvertices
    faces = np.frombuffer(data[face_start:face_start + 12 * nfaces],
                          dtype=">i4").reshape(-1, 3)
    return coordinates, faces


def area_volume(xyz, faces):
    corners = xyz.astype(np.float64)[faces]
    cross = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    area = np.linalg.norm(cross, axis=1).sum() * 0.5
    volume = np.einsum("ij,ij->", corners[:, 0], np.cross(corners[:, 1], corners[:, 2])) / 6
    return float(area), float(volume)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("native", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    start = time.perf_counter()
    project_surface_before_quick_sphere(args.input, args.output)
    seconds = time.perf_counter() - start
    python_xyz, python_faces = read_geometry(args.output)
    native_xyz, native_faces = read_geometry(args.native)
    distance = np.linalg.norm(python_xyz.astype(np.float64) - native_xyz.astype(np.float64), axis=1)
    equal = np.all(python_xyz == native_xyz, axis=1)
    native_area, native_volume = area_volume(native_xyz, native_faces)
    python_area, python_volume = area_volume(python_xyz, python_faces)
    print(json.dumps({
        "python_seconds": seconds,
        "vertices": len(python_xyz), "faces": len(python_faces),
        "identical_vertex_count": int(equal.sum()),
        "identical_face_count": int(np.all(python_faces == native_faces, axis=1).sum()),
        "first_different_vertex": int(np.flatnonzero(~equal)[0]) if not equal.all() else None,
        "median_vertex_error_mm": float(np.median(distance)),
        "max_vertex_error_mm": float(distance.max()),
        "native_area_mm2": native_area,
        "python_area_mm2": python_area,
        "native_enclosed_volume_mm3": native_volume,
        "python_enclosed_volume_mm3": python_volume,
    }, indent=2))


if __name__ == "__main__":
    main()
