"""Locate second LH pial normal mismatches relative to ripped vertices."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np


STATE = np.dtype([("floats", "<f4", 9), ("flags", "<i4", 3)])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--clear", type=Path, required=True)
    parser.add_argument("--module", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.module))
    from fnit.recon_all.place_surface_normals import initial_vertex_normals

    xyz, faces = nib.freesurfer.read_geometry(args.mesh)
    current = initial_vertex_normals(xyz.astype(np.float32), faces)
    source = np.fromfile(args.clear, dtype=STATE)
    expected = source["floats"][:, 3:6]
    different = np.flatnonzero(np.any(current != expected, axis=1))
    ripped = source["flags"][:, 0] != 0
    report = {
        "mismatched_vertices": len(different),
        "mismatched_components": int(np.count_nonzero(current != expected)),
        "ripped_mismatched_vertices": int(np.count_nonzero(ripped[different])),
        "active_mismatched_vertices": int(np.count_nonzero(~ripped[different])),
        "first_mismatched_vertices": different[:20].tolist(),
        "first_active_mismatched_vertices": different[~ripped[different]][:20].tolist(),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
