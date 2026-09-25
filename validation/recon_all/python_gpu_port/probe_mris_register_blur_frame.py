"""Compare full spherical blur output against validation-only native MGZ."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import nibabel as nib
import numpy as np
import tifffile
import torch

from fnit.recon_all.mris_register_blur import blur_atlas_frame


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("atlas", type=Path)
    p.add_argument("native_mgz", type=Path)
    p.add_argument("--frame", type=int, default=4)
    p.add_argument("--sigma", type=float, default=0.5)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    atlas = tifffile.imread(args.atlas)[args.frame].view(np.float32)
    native = np.asanyarray(nib.load(str(args.native_mgz)).dataobj)[:, :, args.frame].T
    start = perf_counter()
    pred_tensor = blur_atlas_frame(torch.from_numpy(atlas.copy()).to(args.device), args.sigma)
    if pred_tensor.is_cuda:
        torch.cuda.synchronize(pred_tensor.device)
    elapsed = perf_counter() - start
    pred = pred_tensor.cpu().numpy()
    diff = np.abs(pred - native)
    print(json.dumps({
        "device": args.device,
        "sigma": args.sigma,
        "frame": args.frame,
        "atlas_sha256": hashlib.sha256(args.atlas.read_bytes()).hexdigest(),
        "native_mgz_sha256": hashlib.sha256(args.native_mgz.read_bytes()).hexdigest(),
        "grid_shape": list(pred.shape),
        "exact_pixels": int(np.count_nonzero(pred == native)),
        "pixels": pred.size,
        "max_abs_error": float(diff.max()),
        "median_abs_error": float(np.median(diff)),
        "torch_seconds_excluding_io": elapsed,
    }, indent=2))


if __name__ == "__main__":
    main()
