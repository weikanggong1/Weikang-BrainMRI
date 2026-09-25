"""Paired sigma-4 first nonlinear gradient from frozen registration inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import nibabel as nib
import nibabel.freesurfer.io as fsio
import numpy as np
import tifffile
import torch

from fnit.recon_all.mris_register_atlas import sample_atlas_on_canonical_sphere
from fnit.recon_all.mris_register_blur import blur_atlas_frame
from fnit.recon_all.mris_register_kernels import (
    center_sphere, normalize_mean_curvature, project_sphere,
)
from fnit.recon_all.mris_register_nonlinear import (
    correlation_gradient_add, first_area_gradient, first_distance_gradient,
    ordered_neighbors_from_faces, sphere_arc_distances, sphere_vertex_normals,
    tangent_basis,
)
from fnit.recon_all.mris_register_parameterization import parameterize_curvature


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("sphere", "smoothwm", "sulc", "rigid", "atlas", "native_grid",
                 "native_capture", "native_after_area"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    read = lambda name, width: np.fromfile(args.native_capture / ("correlation_" + name + ".bin"),
                                           dtype="<f4").reshape(-1, width)
    native_curve = read("curvature", 1)[:, 0]
    native_e1, native_e2 = read("e1", 3), read("e2", 3)
    native_after = read("after_correlation", 3)
    native_area = np.fromfile(args.native_after_area, dtype="<f4").reshape(-1, 3)
    native_grid = np.asarray(nib.load(str(args.native_grid)).dataobj, dtype=np.float32)
    sphere, faces = fsio.read_geometry(str(args.sphere))
    smoothwm, faces_original = fsio.read_geometry(str(args.smoothwm))
    rigid, faces_rigid = fsio.read_geometry(str(args.rigid))
    assert np.array_equal(faces, faces_original) and np.array_equal(faces, faces_rigid)
    atlas = tifffile.imread(args.atlas)
    device = args.device
    vertices = torch.from_numpy(sphere).to(device)
    original = torch.from_numpy(smoothwm).to(device)
    rigid_tensor = torch.from_numpy(rigid).to(device)
    triangles = torch.from_numpy(faces.astype(np.int64)).to(device)
    start = perf_counter()
    canonical = project_sphere(center_sphere(vertices))
    sulc = fsio.read_morph_data(str(args.sulc)).astype(np.float32)
    sulc_tensor = torch.from_numpy(sulc).to(device)
    source_grid = parameterize_curvature(canonical, normalize_mean_curvature(sulc_tensor))
    curvature = normalize_mean_curvature(sample_atlas_on_canonical_sphere(
        canonical, blur_atlas_frame(source_grid, 4.0)))
    source_seconds = perf_counter() - start
    start = perf_counter()
    raw_mean = torch.from_numpy(atlas[3].view(np.float32).copy()).to(device)
    raw_variance = torch.from_numpy(atlas[4].view(np.float32).copy()).to(device)
    mean_curve = normalize_mean_curvature(sample_atlas_on_canonical_sphere(
        canonical, blur_atlas_frame(raw_mean, 4.0)))
    mean_grid = parameterize_curvature(canonical, mean_curve)
    variance_grid = blur_atlas_frame(raw_variance, 4.0)
    target_seconds = perf_counter() - start
    start = perf_counter()
    positions = project_sphere(rigid_tensor.float())
    normals = sphere_vertex_normals(positions, triangles)
    e1, e2 = tangent_basis(normals)
    neighbors, degrees = ordered_neighbors_from_faces(triangles, len(vertices))
    distances = sphere_arc_distances(positions, neighbors, degrees)
    avg_vertex_dist = float(distances.double().sum() / degrees.sum())
    distance = first_distance_gradient(vertices, original, rigid_tensor, triangles)
    after_area = first_area_gradient(vertices, original, rigid_tensor, triangles, distance)
    prediction = correlation_gradient_add(after_area, positions, curvature, e1, e2,
                                          mean_grid, variance_grid, avg_vertex_dist)
    if prediction.is_cuda:
        torch.cuda.synchronize(prediction.device)
    force_seconds = perf_counter() - start
    pred = prediction.cpu().numpy()
    e1_np, e2_np = e1.cpu().numpy(), e2.cpu().numpy()
    curv_np = curvature.cpu().numpy()
    mean_np, variance_np = mean_grid.cpu().numpy(), variance_grid.cpu().numpy()
    area_np = after_area.cpu().numpy()
    result = {
        "hemisphere": args.sphere.name[:2], "device": device,
        "vertices": len(sphere), "faces": len(faces),
        "exact_source_curvature": int(np.count_nonzero(curv_np == native_curve)),
        "exact_target_mean_pixels": int(np.count_nonzero(mean_np == native_grid[:, :, 3].T)),
        "exact_target_variance_pixels": int(np.count_nonzero(variance_np == native_grid[:, :, 4].T)),
        "exact_e1_vertices": int(np.count_nonzero(np.all(e1_np == native_e1, axis=1))),
        "exact_e2_vertices": int(np.count_nonzero(np.all(e2_np == native_e2, axis=1))),
        "exact_after_area_vertices": int(np.count_nonzero(np.all(area_np == native_area, axis=1))),
        "exact_after_correlation_vertices": int(np.count_nonzero(np.all(pred == native_after, axis=1))),
        "max_after_correlation_abs_error": float(np.max(np.abs(pred - native_after))),
        "avg_vertex_dist": avg_vertex_dist,
        "seconds_excluding_io": {"source_curve": source_seconds,
                                 "target_grids": target_seconds,
                                 "distance_area_correlation": force_seconds},
        "input_sha256": {key: hashlib.sha256(path.read_bytes()).hexdigest()
                         for key, path in (("sphere", args.sphere), ("smoothwm", args.smoothwm),
                                           ("sulc", args.sulc), ("rigid", args.rigid),
                                           ("atlas", args.atlas), ("native_grid", args.native_grid),
                                           ("native_after_area", args.native_after_area))},
        "predicted_gradient_sha256": hashlib.sha256(pred.tobytes()).hexdigest(),
        "native_gradient_sha256": hashlib.sha256(native_after.tobytes()).hexdigest(),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
