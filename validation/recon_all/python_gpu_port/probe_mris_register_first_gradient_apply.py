"""Paired conditional first nonlinear geometry check from observed native state."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import nibabel.freesurfer.io as fsio
import numpy as np
import torch

from fnit.recon_all.mris_register_nonlinear import apply_spherical_gradient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("positions", type=Path)
    parser.add_argument("gradient", type=Path)
    parser.add_argument("native_first_step", type=Path)
    parser.add_argument("--dt", type=float, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    native, _ = fsio.read_geometry(str(args.native_first_step))
    positions = np.fromfile(args.positions, dtype="<f4").reshape(-1, 3)
    gradient = np.fromfile(args.gradient, dtype="<f4").reshape(-1, 3)
    assert positions.shape == gradient.shape == native.shape
    start = perf_counter()
    prediction = apply_spherical_gradient(torch.from_numpy(positions).to(args.device),
                                          torch.from_numpy(gradient).to(args.device), args.dt)
    if prediction.is_cuda:
        torch.cuda.synchronize(prediction.device)
    elapsed = perf_counter() - start
    predicted = prediction.cpu().numpy()
    print(json.dumps({
        "device": args.device,
        "vertices": len(native),
        "dt_from_native": args.dt,
        "exact_ordered_vertices": int(np.count_nonzero(np.all(predicted == native, axis=1))),
        "max_abs_coordinate_error_mm": float(np.max(np.abs(predicted - native))),
        "seconds_excluding_io": elapsed,
        "sha256": {key: hashlib.sha256(path.read_bytes()).hexdigest()
                   for key, path in (("positions", args.positions),
                                     ("gradient", args.gradient),
                                     ("native_first_step", args.native_first_step))},
    }, indent=2))


if __name__ == "__main__":
    main()
