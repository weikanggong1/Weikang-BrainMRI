"""Compare full independent topology canonical spheres with installed RAM snapshots."""

from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path

import nibabel.freesurfer.io as fsio
import numpy as np

from fnit.recon_all import topology_preflight_python as preflight


def _compare(python: np.ndarray, native: np.ndarray) -> dict:
    left = np.ascontiguousarray(python, np.float32)
    right = np.ascontiguousarray(native, np.float32)
    if left.shape != right.shape:
        raise ValueError(f"coordinate shape differs: {left.shape}, {right.shape}")
    same = left.view(np.uint32) == right.view(np.uint32)
    differing = np.flatnonzero(~same.ravel())
    first = None
    if len(differing):
        index = int(differing[0])
        vertex, axis = divmod(index, 3)
        first = {"vertex": vertex, "axis": "xyz"[axis],
                 "python_hex": f"0x{int(left.view(np.uint32)[vertex, axis]):08x}",
                 "native_hex": f"0x{int(right.view(np.uint32)[vertex, axis]):08x}",
                 "python_mm": float(left[vertex, axis]),
                 "native_mm": float(right[vertex, axis])}
    return {"matching_components": int(same.sum()), "total_components": int(same.size),
            "matching_vertices": int(np.all(same, axis=1).sum()),
            "total_vertices": int(len(same)), "first_mismatch": first,
            "max_absolute_mm": float(np.max(np.abs(left.astype(np.float64) - right)))}


def validate(sphere_path: Path, native_smoothed: Path, native_centered: Path) -> dict:
    sphere, faces = fsio.read_geometry(str(sphere_path))
    sphere = np.ascontiguousarray(sphere, np.float32)
    faces = np.ascontiguousarray(faces, np.int32)
    native_smooth = np.fromfile(native_smoothed, dtype="<f4").reshape(-1, 3)
    native_center = np.fromfile(native_centered, dtype="<f4").reshape(-1, 3)
    smoothed = preflight.project_and_smooth_sphere(sphere, faces)
    centered, iterations = preflight.center_sphere(smoothed)
    return {"input_sha256": sha256(sphere_path.read_bytes()).hexdigest(),
            "native_sha256": {"smoothed": sha256(native_smoothed.read_bytes()).hexdigest(),
                              "centered": sha256(native_centered.read_bytes()).hexdigest()},
            "center_iterations": iterations,
            "smoothed": _compare(smoothed, native_smooth),
            "centered": _compare(centered, native_center),
            "native_center_change": _compare(native_center, native_smooth)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sphere", type=Path, required=True)
    parser.add_argument("--native-smoothed", type=Path, required=True)
    parser.add_argument("--native-centered", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.sphere, args.native_smoothed, args.native_centered)
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
