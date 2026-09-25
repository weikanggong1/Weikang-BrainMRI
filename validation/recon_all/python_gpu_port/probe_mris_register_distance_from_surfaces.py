"""Check native-free ordered edge lengths and first distance force from surfaces."""

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
    original_chord_distances, sphere_arc_distances,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("sphere", "current_positions", "original_positions", "normals",
                 "native_current_distances", "native_original_distances", "native_gradient"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--avg-nbrs", type=float, required=True)
    parser.add_argument("--orig-area", type=float, required=True)
    parser.add_argument("--total-area", type=float, required=True)
    parser.add_argument("--weight", type=float, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    sphere, faces = fsio.read_geometry(str(args.sphere))
    read_xyz = lambda path: np.fromfile(path, dtype="<f4").reshape(-1, 3)
    current = read_xyz(args.current_positions)
    original = read_xyz(args.original_positions)
    normals = read_xyz(args.normals)
    native_gradient = read_xyz(args.native_gradient)
    native_current = np.fromfile(args.native_current_distances, dtype="<f4")
    native_original = np.fromfile(args.native_original_distances, dtype="<f4")
    assert current.shape == original.shape == normals.shape == native_gradient.shape == sphere.shape
    start = perf_counter()
    neighbors, degrees = ordered_neighbors_from_faces(
        torch.from_numpy(faces.astype(np.int64)).to(args.device), len(sphere))
    adjacency_seconds = perf_counter() - start
    current_tensor = torch.from_numpy(current).to(args.device)
    original_tensor = torch.from_numpy(original).to(args.device)
    start = perf_counter()
    current_distances = sphere_arc_distances(current_tensor, neighbors, degrees)
    original_distances = original_chord_distances(original_tensor, neighbors, degrees)
    if current_distances.is_cuda:
        torch.cuda.synchronize(current_distances.device)
    distances_seconds = perf_counter() - start
    mask = torch.arange(neighbors.shape[1], device=args.device)[None, :] < degrees[:, None]
    packed_current = current_distances[mask].cpu().numpy()
    packed_original = original_distances[mask].cpu().numpy()
    start = perf_counter()
    gradient = distance_gradient(current_tensor, torch.from_numpy(normals).to(args.device),
                                 neighbors, degrees, current_distances, original_distances,
                                 args.avg_nbrs, args.orig_area, args.total_area, args.weight)
    if gradient.is_cuda:
        torch.cuda.synchronize(gradient.device)
    force_seconds = perf_counter() - start
    predicted = gradient.cpu().numpy()
    print(json.dumps({
        "device": args.device,
        "vertices": len(sphere),
        "ordered_edges": len(native_current),
        "exact_current_arc_edges": int(np.count_nonzero(packed_current == native_current)),
        "exact_original_chord_edges": int(np.count_nonzero(packed_original == native_original)),
        "exact_distance_gradient_vectors": int(np.count_nonzero(np.all(predicted == native_gradient, axis=1))),
        "max_abs_gradient_error": float(np.max(np.abs(predicted - native_gradient))),
        "seconds_excluding_io": {"adjacency": adjacency_seconds,
                                 "edge_distances": distances_seconds,
                                 "distance_force": force_seconds},
        "sha256": {key: hashlib.sha256(path.read_bytes()).hexdigest()
                   for key, path in (("sphere", args.sphere), ("current_positions", args.current_positions),
                                     ("original_positions", args.original_positions), ("normals", args.normals),
                                     ("native_current_distances", args.native_current_distances),
                                     ("native_original_distances", args.native_original_distances),
                                     ("native_gradient", args.native_gradient))},
        "predicted_sha256": {"current_distances": hashlib.sha256(packed_current.tobytes()).hexdigest(),
                             "original_distances": hashlib.sha256(packed_original.tobytes()).hexdigest(),
                             "gradient": hashlib.sha256(predicted.tobytes()).hexdigest()},
    }, indent=2))


if __name__ == "__main__":
    main()
