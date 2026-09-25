"""Analytical check for FreeSurfer-style spherical curvature rasterization."""

import torch

from fnit.recon_all.mris_register_parameterization import (
    parameterize_curvature,
)


def test_constant_vertex_values_fill_entire_grid():
    vertices = torch.tensor([[100, 0, 0], [0, 100, 0], [-100, 0, 0],
                             [0, -100, 0], [0, 0, 100], [0, 0, -100]],
                            dtype=torch.float32)
    output = parameterize_curvature(vertices, torch.ones(6), 32, 64)
    assert output.shape == (64, 32)
    assert torch.equal(output, torch.ones_like(output))
