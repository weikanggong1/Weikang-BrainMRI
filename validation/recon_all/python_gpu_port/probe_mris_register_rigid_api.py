"""Validate the public native-free rigid registration tensor API."""

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

from fnit.recon_all.mris_register_rigid import register_rigid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sphere", type=Path)
    parser.add_argument("sulc", type=Path)
    parser.add_argument("atlas", type=Path)
    parser.add_argument("native_rigid_surface", type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    vertices, faces = fsio.read_geometry(str(args.sphere))
    sulc = fsio.read_morph_data(str(args.sulc)).astype(np.float32)
    atlas = tifffile.imread(args.atlas)
    native, native_faces = fsio.read_geometry(str(args.native_rigid_surface))
    device = args.device
    start = perf_counter()
    predicted, angles, score, evaluations = register_rigid(
        torch.from_numpy(vertices).to(device), torch.from_numpy(sulc).to(device),
        torch.from_numpy(atlas[3].view(np.float32).copy()).to(device),
        torch.from_numpy(atlas[4].view(np.float32).copy()).to(device))
    if predicted.is_cuda:
        torch.cuda.synchronize(predicted.device)
    elapsed = perf_counter() - start
    actual = predicted.cpu().numpy()
    print(json.dumps({
        "device": device,
        "seconds_excluding_io": elapsed,
        "angles_degrees": [float(np.degrees(a)) for a in angles],
        "objective": score,
        "angle_evaluations": evaluations,
        "vertices": len(vertices),
        "exact_ordered_vertices": int(np.count_nonzero(np.all(actual == native, axis=1))),
        "max_abs_coordinate_error_mm": float(np.max(np.abs(actual - native))),
        "faces_equal": bool(np.array_equal(faces, native_faces)),
        "input_sha256": {key: hashlib.sha256(path.read_bytes()).hexdigest()
                         for key, path in (("sphere", args.sphere),
                                           ("sulc", args.sulc),
                                           ("atlas", args.atlas),
                                           ("native_rigid_surface", args.native_rigid_surface))},
    }, indent=2))


if __name__ == "__main__":
    main()
