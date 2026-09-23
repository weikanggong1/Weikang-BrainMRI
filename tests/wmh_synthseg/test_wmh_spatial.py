"""Independent orientation and interpolation checks for WMH-SynthSeg geometry."""

import numpy as np
import torch

from freesurfer_torch.wmh_synthseg.spatial import (
    align_volume_to_ref, get_ras_axes, myzoom_torch,
)


def test_axis_swap_and_flip_preserve_world_coordinates():
    image = torch.arange(24).reshape(2, 3, 4)
    affine = np.array([
        [0, -3, 0, 8],
        [2, 0, 0, 9],
        [0, 0, 4, 10],
        [0, 0, 0, 1],
    ], dtype=float)
    assert get_ras_axes(affine).tolist() == [1, 0, 2]

    aligned, aligned_affine = align_volume_to_ref(image, affine, return_aff=True)
    assert torch.equal(aligned, image.swapaxes(0, 1).flip((0,)))
    np.testing.assert_array_equal(aligned_affine, np.array([
        [3, 0, 0, 2],
        [0, 2, 0, 9],
        [0, 0, 4, 10],
        [0, 0, 0, 1],
    ], dtype=float))
    for i, j, k in ((0, 0, 0), (1, 1, 2), (2, 1, 3)):
        old_voxel = np.array([j, 2 - i, k, 1])
        new_voxel = np.array([i, j, k, 1])
        np.testing.assert_array_equal(aligned_affine @ new_voxel, affine @ old_voxel)


def test_trilinear_zoom_uses_half_voxel_coordinates_and_clamped_edges():
    ii, jj, kk = np.indices((3, 4, 2))
    linear = (ii + 2 * jj + 4 * kk).astype(np.float32)
    image = torch.from_numpy(np.stack((linear, linear + 10), axis=-1))
    factor = np.array([2.0, 0.5, 2.0])
    affine = np.diag([2.0, 3.0, 4.0, 1.0])
    affine[:3, 3] = (7, -2, 10)

    zoomed, zoomed_affine = myzoom_torch(image, factor, device='cpu', aff=affine)
    assert zoomed.shape == (6, 2, 4, 2)
    delta = (1 - factor) / (2 * factor)
    for i, j, k in ((0, 0, 0), (2, 1, 1), (5, 1, 3)):
        point = np.clip(delta + np.array([i, j, k]) / factor,
                        [0, 0, 0], [2, 3, 1])
        expected = point @ [1, 2, 4]
        np.testing.assert_allclose(zoomed[i, j, k].numpy(), [expected, expected + 10], atol=1e-5)
    np.testing.assert_allclose(zoomed_affine[:3, :3], affine[:3, :3] / factor)
    np.testing.assert_allclose(zoomed_affine[:3, 3],
                               (affine @ np.r_[delta, 1])[:3])
