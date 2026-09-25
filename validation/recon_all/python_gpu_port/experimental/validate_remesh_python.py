"""Compare a Python remesh surface with a fresh native stage output."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import nibabel.freesurfer as fs
import numpy as np


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    native_vertices, native_faces, native_info = fs.read_geometry(
        args.native, read_metadata=True)
    python_vertices, python_faces, python_info = fs.read_geometry(
        args.python, read_metadata=True)
    keys = sorted(set(native_info) | set(python_info))
    metadata_exact = all(key in native_info and key in python_info
                         and np.array_equal(native_info[key], python_info[key])
                         for key in keys)
    result = {
        "input_sha256": digest(args.input),
        "native_sha256": digest(args.native),
        "python_sha256": digest(args.python),
        "vertices": len(native_vertices),
        "faces": len(native_faces),
        "ordered_vertex_coordinates_exact": bool(np.array_equal(
            native_vertices, python_vertices)),
        "ordered_face_indices_exact": bool(np.array_equal(native_faces, python_faces)),
        "volume_info_exact": metadata_exact,
        "different_coordinate_values": (int(np.count_nonzero(native_vertices != python_vertices))
                                        if native_vertices.shape == python_vertices.shape else None),
        "different_face_indices": (int(np.count_nonzero(native_faces != python_faces))
                                   if native_faces.shape == python_faces.shape else None),
    }
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return int(not (result["ordered_vertex_coordinates_exact"]
                    and result["ordered_face_indices_exact"]
                    and result["volume_info_exact"]))


if __name__ == "__main__":
    raise SystemExit(main())
