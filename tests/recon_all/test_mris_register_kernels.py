"""Numerical checks for isolated spherical-registration kernels."""

import math

import torch

from fnit.recon_all.mris_register_kernels import (
    center_sphere,
    normalize_mean_curvature,
    project_sphere,
    rigid_grid_angle,
    rotate_sphere,
)


def test_center_and_project_source_order():
    points = torch.tensor([[10.0, 0.0, 0.0], [-10.0, 0.0, 0.0],
                           [0.0, 10.0, 0.0], [0.0, -10.0, 0.0]])
    centered = center_sphere(points)
    assert torch.equal(centered, points)
    assert torch.equal(project_sphere(centered), points * 10.0)
    assert torch.equal(project_sphere(torch.zeros(1, 3)), torch.zeros(1, 3))


def test_curvature_normalization_skips_ripped_vertices():
    values = torch.tensor([1.0, 2.0, 3.0, 99.0])
    ripped = torch.tensor([False, False, False, True])
    output = normalize_mean_curvature(values, ripped)
    torch.testing.assert_close(output[:3],
                               torch.tensor([-1.2247449, 0.0, 1.2247449]))
    assert output[3] == values[3]


def test_rigid_grid_rotation_source_convention():
    assert math.isclose(math.degrees(rigid_grid_angle(21)), 10.5,
                        abs_tol=1e-6)
    point = torch.tensor([[1.0, 0.0, 0.0]])
    rotated = rotate_sphere(point, (math.pi / 2, 0.0, 0.0))
    torch.testing.assert_close(rotated, torch.tensor([[0.0, -1.0, 0.0]]),
                               atol=1e-7, rtol=0)
