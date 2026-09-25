"""Trace normalized sulc through source parameterization and sigma-0.5 blur."""

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

from fnit.recon_all.mris_register_atlas import sample_atlas_on_canonical_sphere
from fnit.recon_all.mris_register_blur import blur_atlas_frame
from fnit.recon_all.mris_register_kernels import (
    center_sphere, normalize_mean_curvature, project_sphere,
)
from fnit.recon_all.mris_register_parameterization import parameterize_curvature


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("sphere", type=Path)
    p.add_argument("sulc", type=Path)
    p.add_argument("native_mgz", type=Path)
    p.add_argument("--projections", type=int, default=31)
    p.add_argument("--sigma", type=float, default=0.5)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    vertices, _ = fsio.read_geometry(str(args.sphere))
    sulc = fsio.read_morph_data(str(args.sulc)).astype(np.float32)
    native = np.asanyarray(nib.load(str(args.native_mgz)).dataobj)[:, :, 0].T
    start = perf_counter()
    current = center_sphere(torch.from_numpy(vertices.astype(np.float32)).to(args.device))
    for _ in range(args.projections):
        current = project_sphere(current)
    normalized_sulc = normalize_mean_curvature(torch.from_numpy(sulc).to(args.device))
    grid = parameterize_curvature(current, normalized_sulc)
    blurred = blur_atlas_frame(grid, args.sigma)
    sampled = sample_atlas_on_canonical_sphere(current, blurred)
    normalized = normalize_mean_curvature(sampled)
    predicted_tensor = parameterize_curvature(current, normalized)
    if predicted_tensor.is_cuda:
        torch.cuda.synchronize(predicted_tensor.device)
    elapsed = perf_counter() - start
    predicted = predicted_tensor.cpu().numpy()
    diff = np.abs(predicted - native)
    print(json.dumps({
        "projections": args.projections,
        "sigma": args.sigma,
        "device": args.device,
        "input_sphere_sha256": hashlib.sha256(args.sphere.read_bytes()).hexdigest(),
        "sulc_sha256": hashlib.sha256(args.sulc.read_bytes()).hexdigest(),
        "native_mgz_sha256": hashlib.sha256(args.native_mgz.read_bytes()).hexdigest(),
        "torch_seconds_excluding_io": elapsed,
        "pixels": predicted.size,
        "exact_pixels": int(np.count_nonzero(predicted == native)),
        "max_abs_error": float(diff.max()),
        "median_abs_error": float(np.median(diff)),
        "mean_abs_error": float(diff.mean()),
        "candidate_mean_std": [float(predicted.mean()), float(predicted.std())],
        "native_mean_std": [float(native.mean()), float(native.std())],
    }, indent=2))


if __name__ == "__main__":
    main()
