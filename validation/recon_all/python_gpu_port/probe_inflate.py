"""Compare ordered triangle geometry for a fixed mris_inflate replay."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def read_surface(path: Path):
    data = path.read_bytes()
    if data[:3] != b"\xff\xff\xfe":
        raise ValueError(f"not a FreeSurfer triangle surface: {path}")
    body = memoryview(data)[data.index(b"\n\n", 3) + 2:]
    nvertices, nfaces = np.frombuffer(body[:8], dtype=">i4")
    xyz = np.frombuffer(body[8:8 + 12 * nvertices], dtype=">f4").reshape(-1, 3)
    faces = np.frombuffer(body[8 + 12 * nvertices:8 + 12 * (nvertices + nfaces)],
                          dtype=">i4").reshape(-1, 3)
    return xyz, faces, hashlib.sha256(data).hexdigest()


def compare(a: Path, b: Path):
    xyz_a, faces_a, hash_a = read_surface(a)
    xyz_b, faces_b, hash_b = read_surface(b)
    if xyz_a.shape != xyz_b.shape or faces_a.shape != faces_b.shape:
        raise ValueError(f"different shapes: {a}, {b}")
    vertex_equal = np.all(xyz_a == xyz_b, axis=1)
    face_equal = np.all(faces_a == faces_b, axis=1)
    distance = np.linalg.norm(xyz_a.astype(np.float64) - xyz_b.astype(np.float64), axis=1)
    first = int(np.flatnonzero(~vertex_equal)[0]) if not vertex_equal.all() else None
    return {"left": str(a), "right": str(b), "left_sha256": hash_a,
            "right_sha256": hash_b, "vertices": len(xyz_a), "faces": len(faces_a),
            "identical_vertex_count": int(vertex_equal.sum()),
            "identical_face_count": int(face_equal.sum()),
            "first_different_vertex": first,
            "first_left_xyz": xyz_a[first].astype(float).tolist() if first is not None else None,
            "first_right_xyz": xyz_b[first].astype(float).tolist() if first is not None else None,
            "median_vertex_distance_mm": float(np.median(distance)),
            "max_vertex_distance_mm": float(distance.max()),
            "rms_vertex_distance_mm": float(np.sqrt(np.mean(distance ** 2)))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(args.left, args.right), indent=2))


if __name__ == "__main__":
    main()
