"""Installed FreeSurfer 8.2 pial first-step displacement regression."""

import numpy as np

from fnit.recon_all.place_surface_step import unconstrained_step_with_offsets


def test_first_pial_clipped_step_matches_installed_v80852_bits():
    # Installed RAM/GDB at LH first pial iteration, dt=0.5, max_mm=0.3.
    xyz = np.array([[0xC1CCF074, 0x422561D3, 0x4265DE24]], np.uint32).view(np.float32)
    gradient = np.array([[0xC048FDF7, 0x3FEC26CF, 0x40964E5F]], np.uint32).view(np.float32)
    moved, offsets = unconstrained_step_with_offsets(
        xyz, gradient, np.array([False]), dt=0.5, max_mm=0.3,
    )
    np.testing.assert_array_equal(offsets.view(np.uint32)[0],
                                  [0xBE225043, 0x3DBEB4FF, 0x3E72C355])
    np.testing.assert_array_equal(moved.view(np.uint32)[0],
                                  [0xC1CE3515, 0x4225C12D, 0x4266D0E7])
