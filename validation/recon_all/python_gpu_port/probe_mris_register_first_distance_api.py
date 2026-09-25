"""Paired first registration distance gradient from original surface inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import nibabel.freesurfer.io as fsio
import numpy as np
import torch

from fnit.recon_all.mris_register_kernels import project_sphere
from fnit.recon_all.mris_register_nonlinear import (
    first_distance_gradient, ordered_neighbors_from_faces,
    registration_orig_area, registration_total_area, sphere_vertex_normals,
    three_hop_avg_nbrs,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_sphere", type=Path)
    parser.add_argument("smoothwm", type=Path)
    parser.add_argument("rigid_sphere", type=Path)
    parser.add_argument("native_gradient", type=Path)
    parser.add_argument("native_normals", type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    sphere, faces = fsio.read_geometry(str(args.input_sphere))
    original, original_faces = fsio.read_geometry(str(args.smoothwm))
    rigid, rigid_faces = fsio.read_geometry(str(args.rigid_sphere))
    assert np.array_equal(faces, original_faces) and np.array_equal(faces, rigid_faces)
    native = np.fromfile(args.native_gradient, dtype="<f4").reshape(-1, 3)
    native_normals = np.fromfile(args.native_normals, dtype="<f4").reshape(-1, 3)
    assert sphere.shape == original.shape == rigid.shape == native.shape == native_normals.shape
    faces_tensor = torch.from_numpy(faces.astype(np.int64)).to(args.device)
    sphere_tensor = torch.from_numpy(sphere).to(args.device)
    original_tensor = torch.from_numpy(original).to(args.device)
    rigid_tensor = torch.from_numpy(rigid).to(args.device)
    start = perf_counter()
    predicted = first_distance_gradient(sphere_tensor, original_tensor,
                                        rigid_tensor, faces_tensor)
    if predicted.is_cuda:
        torch.cuda.synchronize(predicted.device)
    stage_seconds = perf_counter() - start
    predicted = predicted.cpu().numpy()
    start = perf_counter()
    normals = sphere_vertex_normals(project_sphere(rigid_tensor), faces_tensor).cpu().numpy()
    neighbors, degrees = ordered_neighbors_from_faces(faces_tensor, len(sphere))
    scalars = {"avg_nbrs": three_hop_avg_nbrs(neighbors, degrees),
               "orig_area": registration_orig_area(sphere_tensor, faces_tensor),
               "total_area": registration_total_area()}
    diagnostics_seconds = perf_counter() - start
    print(json.dumps({
        "device": args.device,
        "vertices": len(sphere),
        "faces": len(faces),
        "exact_normal_vectors": int(np.count_nonzero(np.all(normals == native_normals, axis=1))),
        "exact_distance_gradient_vectors": int(np.count_nonzero(np.all(predicted == native, axis=1))),
        "max_abs_gradient_error": float(np.max(np.abs(predicted - native))),
        "source_derived_scalars": scalars,
        "seconds_excluding_io": {"first_distance_gradient": stage_seconds,
                                 "repeated_diagnostics": diagnostics_seconds},
        "sha256": {key: hashlib.sha256(path.read_bytes()).hexdigest()
                   for key, path in (("input_sphere", args.input_sphere), ("smoothwm", args.smoothwm),
                                     ("rigid_sphere", args.rigid_sphere),
                                     ("native_gradient", args.native_gradient),
                                     ("native_normals", args.native_normals))},
        "predicted_gradient_sha256": hashlib.sha256(predicted.tobytes()).hexdigest(),
        "predicted_normals_sha256": hashlib.sha256(normals.tobytes()).hexdigest(),
    }, indent=2))


if __name__ == "__main__":
    main()
