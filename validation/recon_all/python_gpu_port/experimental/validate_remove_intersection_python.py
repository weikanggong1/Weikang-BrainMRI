"""Compare an isolated zero-intersection Python surface with FreeSurfer output."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import nibabel.freesurfer as fs
import numpy as np


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    original = fs.read_geometry(args.input, read_metadata=True)
    native = fs.read_geometry(args.native, read_metadata=True)
    python = fs.read_geometry(args.python, read_metadata=True)
    def metadata_equal(left, right):
        return set(left) == set(right) and all(np.array_equal(left[k], right[k]) for k in left)
    result = {
        "input_sha256": sha256(args.input),
        "native_sha256": sha256(args.native),
        "python_sha256": sha256(args.python),
        "vertices": len(native[0]),
        "faces": len(native[1]),
        "python_input_bytes_exact": sha256(args.input) == sha256(args.python),
        "native_python_ordered_vertices_exact": bool(np.array_equal(native[0], python[0])),
        "native_python_ordered_faces_exact": bool(np.array_equal(native[1], python[1])),
        "native_python_volume_info_exact": metadata_equal(native[2], python[2]),
        "native_input_ordered_vertices_exact": bool(np.array_equal(native[0], original[0])),
        "native_input_ordered_faces_exact": bool(np.array_equal(native[1], original[1])),
    }
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not all(value for key, value in result.items() if key.endswith("_exact")):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
