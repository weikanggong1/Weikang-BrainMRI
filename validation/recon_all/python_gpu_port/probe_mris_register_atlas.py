"""Compare a frozen TIFF atlas sampled onto the canonical sphere with native."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import nibabel.freesurfer.io as fsio
import numpy as np
import tifffile
import torch

from fnit.recon_all.mris_register_atlas import sample_atlas_on_canonical_sphere
from fnit.recon_all.mris_register_kernels import normalize_mean_curvature


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("sphere", type=Path)
    p.add_argument("atlas", type=Path)
    p.add_argument("native_curv", type=Path)
    p.add_argument("--native-normalized-curv", type=Path)
    p.add_argument("--frame", type=int, default=3)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    vertices, _ = fsio.read_geometry(str(args.sphere))
    atlas = tifffile.imread(args.atlas)[args.frame].view(np.float32)
    native = fsio.read_morph_data(str(args.native_curv)).astype(np.float32)
    start = perf_counter()
    pred_tensor = sample_atlas_on_canonical_sphere(
        torch.from_numpy(vertices.astype(np.float32)).to(args.device),
        torch.from_numpy(atlas.copy()).to(args.device),
    )
    if pred_tensor.is_cuda:
        torch.cuda.synchronize(pred_tensor.device)
    elapsed = perf_counter() - start
    pred = pred_tensor.cpu().numpy()
    diff = np.abs(pred - native)
    report = {
        "device": args.device,
        "input_sphere_sha256": hashlib.sha256(args.sphere.read_bytes()).hexdigest(),
        "atlas_sha256": hashlib.sha256(args.atlas.read_bytes()).hexdigest(),
        "native_curv_sha256": hashlib.sha256(args.native_curv.read_bytes()).hexdigest(),
        "frame": args.frame,
        "vertices": len(vertices),
        "exact_values": int(np.count_nonzero(pred == native)),
        "max_abs_error": float(diff.max()),
        "median_abs_error": float(np.median(diff)),
        "python_sampling_seconds_excluding_io": elapsed,
        "native_mean_std": [float(native.mean()), float(native.std())],
        "candidate_mean_std": [float(pred.mean()), float(pred.std())],
        "first_native": native[:8].tolist(),
        "first_candidate": pred[:8].tolist(),
    }
    if args.native_normalized_curv:
        normalized_native = fsio.read_morph_data(str(args.native_normalized_curv)).astype(np.float32)
        normalized_pred = normalize_mean_curvature(torch.from_numpy(pred)).numpy()
        report["normalized_curv_sha256"] = hashlib.sha256(args.native_normalized_curv.read_bytes()).hexdigest()
        report["normalized_exact_values"] = int(np.count_nonzero(normalized_pred == normalized_native))
        report["normalized_max_abs_error"] = float(np.abs(normalized_pred - normalized_native).max())
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
