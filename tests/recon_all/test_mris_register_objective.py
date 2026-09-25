"""Small numerical checks for the native rigid-search objective."""

import math

import torch

from fnit.recon_all.mris_register_objective import (
    _fast_atan2, rigid_search, rigid_sse,
)


def test_fast_atan2_quadrants():
    x = torch.tensor([1.0, 0.0, -1.0, 0.0, 1.0])
    y = torch.tensor([0.0, 1.0, 0.0, -1.0, 1.0])
    values = _fast_atan2(y, x)
    torch.testing.assert_close(values,
                               torch.tensor([0.0, math.pi / 2, math.pi,
                                             -math.pi / 2, math.pi / 4]),
                               atol=1e-5, rtol=0)


def test_constant_atlas_objective_and_search():
    vertices = torch.tensor([[100.0, 0.0, 0.0], [0.0, 100.0, 0.0],
                             [0.0, 0.0, 100.0], [-100.0, 0.0, 0.0]])
    curvature = torch.ones(4)
    mean = torch.zeros((16, 8))
    variance = torch.ones((16, 8))
    assert rigid_sse(vertices, curvature, mean, variance,
                     (0.0, 0.0, 0.0)) == 4.0
    angles, score, count = rigid_search(vertices, curvature, mean, variance,
                                        max_degrees=4.0, min_degrees=1.0,
                                        nangles=2)
    assert len(angles) == 3
    assert score == 4.0
    assert 1 <= count <= 4 ** 3
