"""Native-free first rigid pass of FreeSurfer spherical registration."""

from __future__ import annotations

import torch

from .mris_register_atlas import sample_atlas_on_canonical_sphere
from .mris_register_blur import blur_atlas_frame
from .mris_register_kernels import (
    center_sphere, normalize_mean_curvature, project_sphere, rotate_sphere,
)
from .mris_register_objective import rigid_search
from .mris_register_parameterization import parameterize_curvature


@torch.no_grad()
def register_rigid(vertices: torch.Tensor, sulc: torch.Tensor,
                   atlas_mean: torch.Tensor, atlas_variance: torch.Tensor
                   ) -> tuple[torch.Tensor, tuple[float, float, float], float, int]:
    """Return sigma-4 rigid vertices, angles, objective and angle evaluations.

    Atlas frames have shape (512, 256) and float32 values. TIFF integer
    storage must be reinterpreted as float32 by the caller before invocation.
    All inputs must be on the same device; faces retain their input ordering.
    """
    sphere = project_sphere(center_sphere(vertices))
    source_grid = parameterize_curvature(sphere, normalize_mean_curvature(sulc))
    source_curve = normalize_mean_curvature(sample_atlas_on_canonical_sphere(
        sphere, blur_atlas_frame(source_grid, 4.0)))
    mean_curve = normalize_mean_curvature(sample_atlas_on_canonical_sphere(
        sphere, blur_atlas_frame(atlas_mean, 4.0)))
    mean_grid = parameterize_curvature(sphere, mean_curve)
    variance_grid = blur_atlas_frame(atlas_variance, 4.0)
    angles, score, evaluations = rigid_search(sphere, source_curve,
                                              mean_grid, variance_grid)
    return rotate_sphere(sphere, angles), angles, score, evaluations
