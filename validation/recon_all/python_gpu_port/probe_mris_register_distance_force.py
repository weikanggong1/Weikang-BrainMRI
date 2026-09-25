"""Paired first mris_register distance-force checkpoint on frozen native inputs."""

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
    distance_gradient, ordered_neighbors_from_faces,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sphere", type=Path)
    parser.add_argument("positions", type=Path)
    parser.add_argument("normals", type=Path)
    parser.add_argument("current_distances", type=Path)
    parser.add_argument("original_distances", type=Path)
    parser.add_argument("native_gradient", type=Path)
    parser.add_argument("--avg-nbrs", type=float, required=True)
    parser.add_argument("--orig-area", type=float, required=True)
    parser.add_argument("--total-area", type=float, required=True)
    parser.add_argument("--weight", type=float, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    sphere, faces = fsio.read_geometry(str(args.sphere))
    positions = np.fromfile(args.positions, dtype="<f4").reshape(-1, 3)
    normals = np.fromfile(args.normals, dtype="<f4").reshape(-1, 3)
    native = np.fromfile(args.native_gradient, dtype="<f4").reshape(-1, 3)
    current = np.fromfile(args.current_distances, dtype="<f4")
    original = np.fromfile(args.original_distances, dtype="<f4")
    assert positions.shape == normals.shape == native.shape == sphere.shape
    start = perf_counter()
    neighbors, degrees = ordered_neighbors_from_faces(
        torch.from_numpy(faces.astype(np.int64)).to(args.device), len(sphere))
    degree_cpu = degrees.cpu().numpy()
    width = neighbors.shape[1]
    current_padded = np.zeros((len(sphere), width), dtype=np.float32)
    original_padded = np.zeros_like(current_padded)
    offset = 0
    for vertex, degree in enumerate(degree_cpu):
        current_padded[vertex, :degree] = current[offset:offset + degree]
        original_padded[vertex, :degree] = original[offset:offset + degree]
        offset += degree
    assert offset == len(current) == len(original)
    preparation_seconds = perf_counter() - start
    start = perf_counter()
    prediction = distance_gradient(
        torch.from_numpy(positions).to(args.device), torch.from_numpy(normals).to(args.device),
        neighbors, degrees, torch.from_numpy(current_padded).to(args.device),
        torch.from_numpy(original_padded).to(args.device), args.avg_nbrs,
        args.orig_area, args.total_area, args.weight)
    if prediction.is_cuda:
        torch.cuda.synchronize(prediction.device)
    kernel_seconds = perf_counter() - start
    predicted = prediction.cpu().numpy()
    print(json.dumps({
        "device": args.device,
        "vertices": len(sphere),
        "faces": len(faces),
        "native_observed_scalars": {"avg_nbrs": args.avg_nbrs, "orig_area": args.orig_area,
                                    "total_area": args.total_area, "weight": args.weight},
        "exact_ordered_gradient_vectors": int(np.count_nonzero(np.all(predicted == native, axis=1))),
        "max_abs_gradient_error": float(np.max(np.abs(predicted - native))),
        "seconds_excluding_io": {"adjacency_and_distance_packing": preparation_seconds,
                                 "distance_force": kernel_seconds},
        "sha256": {key: hashlib.sha256(path.read_bytes()).hexdigest()
                   for key, path in (("sphere", args.sphere), ("positions", args.positions),
                                     ("normals", args.normals),
                                     ("native_current_distances", args.current_distances),
                                     ("native_original_distances", args.original_distances),
                                     ("native_gradient", args.native_gradient))},
        "predicted_sha256": hashlib.sha256(predicted.tobytes()).hexdigest(),
    }, indent=2))


if __name__ == "__main__":
    main()
