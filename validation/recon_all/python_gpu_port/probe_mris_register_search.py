"""Frozen one-pass sigma-4 rigid search and ordered-vertex parity gate."""

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
from fnit.recon_all.mris_register_parameterization import parameterize_curvature
from fnit.recon_all.mris_register_kernels import (
    center_sphere, normalize_mean_curvature, project_sphere, rotate_sphere,
)
from fnit.recon_all.mris_register_objective import rigid_search


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sphere", type=Path)
    parser.add_argument("sulc", type=Path)
    parser.add_argument("atlas", type=Path)
    parser.add_argument("native_source_grid", type=Path)
    parser.add_argument("target_grid", type=Path)
    parser.add_argument("native_curvature", type=Path)
    parser.add_argument("native_rigid_surface", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--source-only", action="store_true")
    args = parser.parse_args()
    vertices, faces = fsio.read_geometry(str(args.sphere))
    native_curv = fsio.read_morph_data(str(args.native_curvature)).astype(np.float32)
    sulc = fsio.read_morph_data(str(args.sulc)).astype(np.float32)
    source = np.asarray(nib.load(str(args.native_source_grid)).dataobj, dtype=np.float32)
    target = np.asarray(nib.load(str(args.target_grid)).dataobj, dtype=np.float32)
    atlas = tifffile.imread(args.atlas)
    native_vertices, native_faces = fsio.read_geometry(str(args.native_rigid_surface))
    device = args.device
    projected = project_sphere(center_sphere(torch.from_numpy(vertices).to(device)))
    start = perf_counter()
    initial_curve = normalize_mean_curvature(torch.from_numpy(sulc).to(device))
    initial_grid = parameterize_curvature(projected, initial_curve)
    blurred_grid = blur_atlas_frame(initial_grid, 4.0)
    source_curve = normalize_mean_curvature(sample_atlas_on_canonical_sphere(
        projected, blurred_grid))
    predicted_grid = parameterize_curvature(projected, source_curve).cpu().numpy()
    native_grid = source[:, :, 0].T
    target_start = perf_counter()
    raw_mean = torch.from_numpy(atlas[3].view(np.float32).copy()).to(device)
    raw_variance = torch.from_numpy(atlas[4].view(np.float32).copy()).to(device)
    smoothed_mean = blur_atlas_frame(raw_mean, 4.0)
    mean_curve = normalize_mean_curvature(sample_atlas_on_canonical_sphere(
        projected, smoothed_mean))
    mean_grid = parameterize_curvature(projected, mean_curve)
    variance_grid = blur_atlas_frame(raw_variance, 4.0)
    target_seconds = perf_counter() - target_start
    predicted_mean = mean_grid.cpu().numpy()
    predicted_variance = variance_grid.cpu().numpy()
    computed_curv = source_curve.cpu().numpy()
    source_seconds = perf_counter() - start
    result = {
        "device": device,
        "vertices": len(vertices),
        "faces_equal": bool(np.array_equal(faces, native_faces)),
        "source_grid_exact_pixels": int(np.count_nonzero(predicted_grid == native_grid)),
        "source_grid_max_abs_error": float(np.max(np.abs(predicted_grid - native_grid))),
        "target_mean_exact_pixels": int(np.count_nonzero(predicted_mean == target[:, :, 3].T)),
        "target_mean_max_abs_error": float(np.max(np.abs(predicted_mean - target[:, :, 3].T))),
        "target_variance_exact_pixels": int(np.count_nonzero(predicted_variance == target[:, :, 4].T)),
        "target_variance_max_abs_error": float(np.max(np.abs(predicted_variance - target[:, :, 4].T))),
        "target_grids_seconds_excluding_io": target_seconds,
        "curvature_exact_vertices": int(np.count_nonzero(computed_curv == native_curv)),
        "curvature_max_abs_error": float(np.max(np.abs(computed_curv - native_curv))),
        "source_curve_seconds_excluding_io": source_seconds,
        "input_sha256": {key: hashlib.sha256(path.read_bytes()).hexdigest()
                         for key, path in (("sphere", args.sphere),
                                           ("sulc", args.sulc),
                                           ("atlas", args.atlas),
                                           ("native_source_grid", args.native_source_grid),
                                           ("target_grid", args.target_grid),
                                           ("native_curvature", args.native_curvature),
                                           ("native_rigid_surface", args.native_rigid_surface))},
    }
    if not args.source_only:
        start = perf_counter()
        angles, score, evaluations = rigid_search(
            projected, source_curve,
            mean_grid, variance_grid)
        prediction = rotate_sphere(projected, angles)
        if prediction.is_cuda:
            torch.cuda.synchronize(prediction.device)
        predicted = prediction.cpu().numpy()
        result.update({
            "angles_degrees": [float(np.degrees(a)) for a in angles],
            "objective": score,
            "angle_evaluations": evaluations,
            "search_and_rotation_seconds_excluding_io": perf_counter() - start,
            "exact_ordered_vertices": int(np.count_nonzero(np.all(predicted == native_vertices, axis=1))),
            "max_abs_coordinate_error_mm": float(np.max(np.abs(predicted - native_vertices))),
            "median_vertex_error_mm": float(np.median(np.linalg.norm(predicted - native_vertices, axis=1))),
        })
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
