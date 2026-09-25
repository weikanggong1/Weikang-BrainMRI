"""Compare the fixed zero-iteration mris_register projection with native output."""

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
    center_sphere,
    normalize_mean_curvature,
    project_sphere,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_sphere", type=Path)
    parser.add_argument("input_sulc", type=Path)
    parser.add_argument("native_zero_iteration_sphere", type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    xyz, faces = fsio.read_geometry(str(args.input_sphere))
    native_xyz, native_faces = fsio.read_geometry(str(args.native_zero_iteration_sphere))
    start = perf_counter()
    predicted = center_sphere(torch.from_numpy(xyz.astype(np.float32)).to(args.device))
    # One initial projection, then 9+7+7+7+7 zero-step integration epochs.
    for _ in range(38):
        predicted = project_sphere(predicted)
    if predicted.is_cuda:
        torch.cuda.synchronize(predicted.device)
    elapsed = perf_counter() - start
    result = predicted.cpu().numpy()

    sulc = fsio.read_morph_data(str(args.input_sulc)).astype(np.float32)
    sulc_tensor = torch.from_numpy(sulc).to(args.device)
    mean = float(sulc_tensor.to(torch.float64).mean().item())
    std = float(sulc_tensor.to(torch.float64).std(correction=0).item())
    normalized = normalize_mean_curvature(sulc_tensor)
    report = {
        "device": args.device,
        "input_sha256": sha256(args.input_sphere),
        "sulc_sha256": sha256(args.input_sulc),
        "native_diagnostic_sha256": sha256(args.native_zero_iteration_sphere),
        "vertices": len(xyz),
        "faces": len(faces),
        "ordered_faces_equal": bool(np.array_equal(faces, native_faces)),
        "exact_vertices": int(np.count_nonzero(np.all(result == native_xyz, axis=1))),
        "max_abs_coordinate_error_mm": float(np.max(np.abs(result - native_xyz))),
        "projection_seconds_excluding_io": elapsed,
        "sulc_mean_before_normalization": mean,
        "sulc_std_before_normalization": std,
        "normalized_sulc_mean": float(normalized.to(torch.float64).mean().item()),
        "normalized_sulc_std": float(normalized.to(torch.float64).std(correction=0).item()),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
