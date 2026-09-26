"""Paired CPU/CUDA force calculation at the first smoothwm update."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import nibabel.freesurfer.io as fsio
import numpy as np
import tifffile
import torch

from fnit.recon_all.mris_register_atlas import sample_atlas_on_canonical_sphere
from fnit.recon_all.mris_register_blur import blur_atlas_frame
from fnit.recon_all.mris_register_kernels import normalize_mean_curvature, project_sphere
from fnit.recon_all.mris_register_nonlinear import (
    correlation_gradient_add, first_area_gradient, first_distance_gradient,
    ordered_neighbors_from_faces, sphere_arc_distances, sphere_vertex_normals,
    tangent_basis,
)
from fnit.recon_all.mris_register_parameterization import parameterize_curvature


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("sphere", "smoothwm", "seed", "atlas", "raw", "report"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--cuda-device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    sphere, faces = fsio.read_geometry(str(args.sphere))
    smoothwm, smooth_faces = fsio.read_geometry(str(args.smoothwm))
    seed, seed_faces = fsio.read_geometry(str(args.seed))
    assert np.array_equal(faces, smooth_faces) and np.array_equal(faces, seed_faces)
    vertices = torch.from_numpy(sphere.astype(np.float32))
    original = torch.from_numpy(smoothwm.astype(np.float32))
    current = torch.from_numpy(seed.astype(np.float32))
    triangles = torch.from_numpy(faces.astype(np.int64))
    raw = torch.from_numpy(np.fromfile(args.raw, dtype='<f4').copy())
    atlas = tifffile.imread(args.atlas)
    source = parameterize_curvature(current, normalize_mean_curvature(raw))
    curvature = normalize_mean_curvature(sample_atlas_on_canonical_sphere(
        current, blur_atlas_frame(source, 4.0)))
    mean = torch.from_numpy(atlas[6].view(np.float32).copy())
    mean_curve = normalize_mean_curvature(sample_atlas_on_canonical_sphere(
        current, blur_atlas_frame(mean, 4.0)))
    target = parameterize_curvature(current, mean_curve)
    variance = blur_atlas_frame(torch.from_numpy(atlas[7].view(np.float32).copy()), 4.0)

    def calculate(device: str) -> tuple[np.ndarray, float]:
        v, o, c, f = (tensor.to(device) for tensor in
                      (vertices, original, current, triangles))
        curve, target_grid, variance_grid = (tensor.to(device) for tensor in
                                            (curvature, target, variance))
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)
        started = perf_counter()
        projected = project_sphere(c)
        neighbors, degrees = ordered_neighbors_from_faces(f, len(c))
        normals = sphere_vertex_normals(projected, f)
        distances = sphere_arc_distances(projected, neighbors, degrees)
        avg_vertex_dist = float(distances.double().sum() / degrees.sum())
        area = first_area_gradient(
            v, o, c, f, first_distance_gradient(v, o, c, f))
        e1, e2 = tangent_basis(normals)
        result = correlation_gradient_add(
            area, projected, curve, e1, e2, target_grid, variance_grid,
            avg_vertex_dist, l_corr=0.05)
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)
        seconds = perf_counter() - started
        return result.cpu().numpy(), seconds

    cpu, cpu_seconds = calculate("cpu")
    gpu, gpu_seconds = calculate(args.cuda_device)
    delta = np.abs(cpu - gpu)
    report = {"cuda_device": args.cuda_device, "dtype": "float32",
              "tf32": True, "vertices": len(cpu),
              "exact_vertices": int(np.all(cpu == gpu, axis=1).sum()),
              "within_1e-5_vertices": int(np.all(delta <= 1e-5, axis=1).sum()),
              "within_1e-4_vertices": int(np.all(delta <= 1e-4, axis=1).sum()),
              "max_abs_error": float(delta.max()),
              "p95_abs_error": float(np.percentile(delta, 95)),
              "cpu_force_seconds": cpu_seconds,
              "gpu_force_seconds_excluding_transfer": gpu_seconds}
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
