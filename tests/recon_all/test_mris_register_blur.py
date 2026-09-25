"""Analytical check for spherical registration blur."""

import torch

from fnit.recon_all.mris_register_blur import blur_atlas_frame


def test_constant_frame_is_preserved_at_poles_and_equator():
    frame = torch.full((64, 32), 3.0)
    result = blur_atlas_frame(frame, sigma=0.5)
    torch.testing.assert_close(result, frame, atol=1e-6, rtol=0)
