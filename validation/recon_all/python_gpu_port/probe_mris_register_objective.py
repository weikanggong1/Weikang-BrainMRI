"""Check a frozen native rigid-search input against the Python objective."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import nibabel as nib
import nibabel.freesurfer.io as fsio
import numpy as np
import torch

from fnit.recon_all.mris_register_kernels import (
    center_sphere, project_sphere, rigid_grid_angle,
)
from fnit.recon_all.mris_register_objective import rigid_sse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sphere", type=Path)
    parser.add_argument("native_curvature", type=Path)
    parser.add_argument("native_target_grid", type=Path)
    parser.add_argument("--grid-indices", type=int, nargs=3, default=(0, 0, 0))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    vertices, _ = fsio.read_geometry(str(args.sphere))
    curvature = fsio.read_morph_data(str(args.native_curvature)).astype(np.float32)
    grid = np.asarray(nib.load(str(args.native_target_grid)).dataobj,
                      dtype=np.float32)
    vertex_tensor = project_sphere(center_sphere(torch.from_numpy(vertices).to(args.device)))
    indices = tuple(args.grid_indices)
    angles = tuple(rigid_grid_angle(i) for i in indices)
    start = perf_counter()
    score = rigid_sse(vertex_tensor, torch.from_numpy(curvature).to(args.device),
                      torch.from_numpy(grid[:, :, 3].T.copy()).to(args.device),
                      torch.from_numpy(grid[:, :, 4].T.copy()).to(args.device),
                      angles)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    print(json.dumps({
        "grid_indices": indices,
        "angles_degrees": [float(np.degrees(x)) for x in angles],
        "objective": score,
        "seconds_excluding_io": perf_counter() - start,
        "vertices": len(vertices),
        "sha256": {key: hashlib.sha256(path.read_bytes()).hexdigest()
                   for key, path in (("sphere", args.sphere),
                                     ("curvature", args.native_curvature),
                                     ("target", args.native_target_grid))},
    }, indent=2))


if __name__ == "__main__":
    main()
