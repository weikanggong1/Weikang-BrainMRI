"""Paired first nonlinear 16384-average checkpoint on frozen registration inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import nibabel.freesurfer.io as fsio
import numpy as np
import torch

from fnit.recon_all.mris_register_nonlinear import (
    average_gradients, ordered_neighbors_from_faces,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sphere", type=Path)
    parser.add_argument("before", type=Path)
    parser.add_argument("native_after", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--iterations", type=int, default=16384)
    args = parser.parse_args()
    vertices, faces = fsio.read_geometry(str(args.sphere))
    before = np.fromfile(args.before, dtype="<f4").reshape(-1, 3)
    after = np.fromfile(args.native_after, dtype="<f4").reshape(-1, 3)
    assert before.shape == after.shape == vertices.shape
    start = perf_counter()
    neighbors, degrees = ordered_neighbors_from_faces(
        torch.from_numpy(faces.astype(np.int64)).to(args.device), len(vertices))
    adjacency_seconds = perf_counter() - start
    start = perf_counter()
    prediction = average_gradients(torch.from_numpy(before).to(args.device),
                                   neighbors, degrees, args.iterations)
    if prediction.is_cuda:
        torch.cuda.synchronize(prediction.device)
    averaging_seconds = perf_counter() - start
    predicted = prediction.cpu().numpy()
    print(json.dumps({
        "device": args.device,
        "vertices": len(vertices),
        "iterations": args.iterations,
        "exact_gradient_vectors": int(np.count_nonzero(np.all(predicted == after, axis=1))),
        "max_abs_gradient_error": float(np.max(np.abs(predicted - after))),
        "seconds_excluding_io": {"ordered_adjacency": adjacency_seconds,
                                 "averaging": averaging_seconds},
        "sha256": {key: hashlib.sha256(path.read_bytes()).hexdigest()
                   for key, path in (("sphere", args.sphere), ("before", args.before),
                                     ("native_after", args.native_after))},
        "predicted_sha256": hashlib.sha256(predicted.tobytes()).hexdigest(),
    }, indent=2))


if __name__ == "__main__":
    main()
