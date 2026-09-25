"""Quantify the first native nonlinear sphere update after exact rigid search."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import nibabel.freesurfer.io as fsio
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("rigid_surface", type=Path)
    parser.add_argument("first_step_surface", type=Path)
    args = parser.parse_args()
    rigid, rigid_faces = fsio.read_geometry(str(args.rigid_surface))
    step, step_faces = fsio.read_geometry(str(args.first_step_surface))
    displacement = step.astype(np.float64) - rigid.astype(np.float64)
    distance = np.linalg.norm(displacement, axis=1)
    print(json.dumps({
        "rigid_sha256": hashlib.sha256(args.rigid_surface.read_bytes()).hexdigest(),
        "first_step_sha256": hashlib.sha256(args.first_step_surface.read_bytes()).hexdigest(),
        "vertices": len(rigid),
        "faces_equal": bool(np.array_equal(rigid_faces, step_faces)),
        "exact_unchanged_vertices": int(np.count_nonzero(np.all(step == rigid, axis=1))),
        "median_displacement_mm": float(np.median(distance)),
        "p95_displacement_mm": float(np.percentile(distance, 95)),
        "max_displacement_mm": float(distance.max()),
        "first_rigid_vertex": rigid[0].tolist(),
        "first_step_vertex": step[0].tolist(),
    }, indent=2))


if __name__ == "__main__":
    main()
