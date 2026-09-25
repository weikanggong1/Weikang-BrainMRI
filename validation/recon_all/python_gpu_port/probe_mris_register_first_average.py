"""Check source-order PyTorch one-ring averaging against a frozen GDB checkpoint."""

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
    average_gradients_once, ordered_neighbors_from_faces,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sphere", type=Path)
    parser.add_argument("offsets", type=Path)
    parser.add_argument("neighbors", type=Path)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    vertices, faces = fsio.read_geometry(str(args.sphere))
    before = np.fromfile(args.before, dtype="<f4").reshape(-1, 3)
    after = np.fromfile(args.after, dtype="<f4").reshape(-1, 3)
    native_offsets = np.fromfile(args.offsets, dtype="<i4")
    native_neighbors = np.fromfile(args.neighbors, dtype="<i4")
    assert before.shape == after.shape == vertices.shape
    assert native_offsets.shape == (len(vertices) + 1,)
    face_tensor = torch.from_numpy(faces.astype(np.int64)).to(args.device)
    start = perf_counter()
    neighbors, degrees = ordered_neighbors_from_faces(face_tensor, len(vertices))
    adjacency_seconds = perf_counter() - start
    neighbor_cpu = neighbors.cpu().numpy()
    degree_cpu = degrees.cpu().numpy()
    offsets = np.r_[0, np.cumsum(degree_cpu, dtype=np.int64)].astype("<i4")
    packed = np.concatenate([neighbor_cpu[i, :degree] for i, degree in enumerate(degree_cpu)]).astype("<i4")
    gradient = torch.from_numpy(before).to(args.device)
    start = perf_counter()
    prediction = average_gradients_once(gradient, neighbors, degrees)
    if prediction.is_cuda:
        torch.cuda.synchronize(prediction.device)
    kernel_seconds = perf_counter() - start
    predicted = prediction.cpu().numpy()
    report = {
        "device": args.device,
        "vertices": len(vertices),
        "faces": len(faces),
        "native_original_num_averages": 16384,
        "debugger_num_averages_for_this_isolated_kernel": 1,
        "exact_ordered_neighbor_offsets": bool(np.array_equal(offsets, native_offsets)),
        "exact_ordered_neighbor_entries": int(np.count_nonzero(packed == native_neighbors)),
        "ordered_neighbor_entries": len(native_neighbors),
        "exact_ordered_gradient_vectors": int(np.count_nonzero(np.all(predicted == after, axis=1))),
        "max_abs_gradient_error": float(np.max(np.abs(predicted - after))),
        "seconds_excluding_io": {"ordered_adjacency": adjacency_seconds,
                                 "one_average": kernel_seconds},
        "sha256": {key: hashlib.sha256(path.read_bytes()).hexdigest()
                   for key, path in (("sphere", args.sphere), ("native_offsets", args.offsets),
                                     ("native_neighbors", args.neighbors), ("gradient_before", args.before),
                                     ("native_gradient_after", args.after))},
        "predicted_gradient_sha256": hashlib.sha256(predicted.tobytes()).hexdigest(),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
