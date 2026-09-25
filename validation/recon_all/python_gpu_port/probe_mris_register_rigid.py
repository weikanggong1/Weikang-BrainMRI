"""Locate the first mismatch in a frozen zero-iteration rigid registration."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import nibabel.freesurfer.io as fsio
import numpy as np
import torch

from fnit.recon_all.mris_register_kernels import (
    center_sphere, project_sphere, rigid_grid_angle, rotate_sphere,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("sphere", type=Path)
    p.add_argument("native_rigid", type=Path)
    p.add_argument("--angles", type=float, nargs=3, required=True)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    xyz, faces = fsio.read_geometry(str(args.sphere))
    native, native_faces = fsio.read_geometry(str(args.native_rigid))
    angles = tuple(rigid_grid_angle(round(degrees * 2))
                   for degrees in args.angles)
    v = center_sphere(torch.tensor(xyz, dtype=torch.float32, device=args.device))
    results = []
    matched_kernel_seconds = None
    for nproj in range(4):
        start = perf_counter()
        if nproj:
            v = project_sphere(v)
        predicted = rotate_sphere(v, angles)
        if predicted.is_cuda:
            torch.cuda.synchronize(predicted.device)
        if nproj == 1:
            matched_kernel_seconds = perf_counter() - start
        pred = predicted.cpu().numpy()
        abs_error = np.abs(pred - native)
        results.append({
            "projections": nproj,
            "exact_vertices": int(np.count_nonzero(np.all(pred == native, axis=1))),
            "max_abs_error_mm": float(abs_error.max()),
            "median_vertex_error_mm": float(np.median(np.linalg.norm(pred - native, axis=1))),
            "first_candidate": pred[0].tolist(),
            "first_native": native[0].tolist(),
        })
    print(json.dumps({
        "device": args.device,
        "input_sha256": hashlib.sha256(args.sphere.read_bytes()).hexdigest(),
        "native_rigid_sha256": hashlib.sha256(args.native_rigid.read_bytes()).hexdigest(),
        "angles_degrees_from_native_log": args.angles,
        "vertices": len(xyz),
        "faces_equal": bool(np.array_equal(faces, native_faces)),
        "one_projection_and_rotation_seconds_excluding_io": matched_kernel_seconds,
        "results": results,
    }, indent=2))


if __name__ == "__main__":
    main()
