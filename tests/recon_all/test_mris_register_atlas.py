"""Geometry checks for spherical atlas sampling."""

import torch

from fnit.recon_all.mris_register_atlas import (
    sample_atlas_on_canonical_sphere,
)


def test_spherical_atlas_axes_and_poles():
    vertices = torch.tensor([[100, 0, 0], [0, 100, 0], [-100, 0, 0],
                             [0, -100, 0], [0, 0, 100], [0, 0, -100]],
                            dtype=torch.float32)
    polar = torch.arange(256, dtype=torch.float32).expand(512, 256)
    azimuth = torch.arange(512, dtype=torch.float32)[:, None].expand(512, 256)
    torch.testing.assert_close(sample_atlas_on_canonical_sphere(vertices, polar),
                               torch.tensor([128, 128, 128, 128, 0, 255],
                                            dtype=torch.float32), atol=1e-4, rtol=0)
    torch.testing.assert_close(sample_atlas_on_canonical_sphere(vertices, azimuth)[:4],
                               torch.tensor([0, 128, 256, 384], dtype=torch.float32),
                               atol=1e-4, rtol=0)
