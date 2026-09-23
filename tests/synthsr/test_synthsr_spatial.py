"""Small-array checks for SynthSR's voxel grid and orientation changes."""

import numpy as np
from scipy.ndimage import gaussian_filter

from freesurfer_torch.synthsr.spatial import (
    align_volume_to_ref,
    crop_volume_with_idx,
    pad_volume,
    resample_volume,
)


def test_upsample_uses_half_voxel_grid_and_updates_affine():
    image = np.broadcast_to(np.arange(3, dtype=float)[:, None, None], (3, 4, 5))
    affine = np.diag([2.0, 1.0, 1.0, 1.0])

    output, new_affine = resample_volume(image, affine, blur=False)

    assert output.shape == (6, 4, 5)
    np.testing.assert_allclose(output[:, 0, 0], [0, 0.25, 0.75, 1.25, 1.75, 2])
    np.testing.assert_allclose(new_affine, np.array([
        [1, 0, 0, -0.5],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]))
    assert np.array_equal(affine, np.diag([2.0, 1.0, 1.0, 1.0]))


def test_downsample_blurs_before_interpolation():
    image = np.broadcast_to(np.arange(6, dtype=float)[None, None, :], (3, 4, 6))
    affine = np.diag([1.0, 1.0, 0.5, 1.0])
    output, new_affine = resample_volume(image, affine)

    filtered = gaussian_filter(image, sigma=[0, 0, 0.5])
    expected = 0.5 * (filtered[0, 0, 0::2] + filtered[0, 0, 1::2])
    assert output.shape == (3, 4, 3)
    np.testing.assert_allclose(output[0, 0], expected, atol=1e-12)
    np.testing.assert_allclose(new_affine[:3, 3], [0, 0, 0.25])
    np.testing.assert_allclose(np.diag(new_affine)[:3], [1, 1, 1])


def test_align_and_inverse_preserve_voxels_and_world_coordinates():
    image = np.arange(3 * 4 * 5).reshape(3, 4, 5)
    affine = np.array([
        [0, -2, 0, 10],
        [1, 0, 0, 20],
        [0, 0, -3, 30],
        [0, 0, 0, 1],
    ], dtype=float)

    aligned, aligned_affine = align_volume_to_ref(image, affine, return_aff=True)
    np.testing.assert_array_equal(aligned, np.flip(np.transpose(image, (1, 0, 2)), (0, 2)))
    np.testing.assert_allclose(aligned_affine[:3, :3], np.diag([2, 1, 3]))
    np.testing.assert_allclose(aligned_affine[:3, 3], [4, 20, 18])

    restored, restored_affine = align_volume_to_ref(
        aligned, aligned_affine, aff_ref=affine, return_aff=True
    )
    np.testing.assert_array_equal(restored, image)
    np.testing.assert_allclose(restored_affine, affine)


def test_padding_and_crop_are_inverse_even_with_an_extra_channel_axis():
    image = np.arange(17 * 18 * 19 * 2).reshape(17, 18, 19, 2)
    affine = np.array([
        [1, 0, 0, 10],
        [0, 2, 0, 20],
        [0, 0, 3, 30],
        [0, 0, 0, 1],
    ], dtype=float)

    padded, padded_affine, indices = pad_volume(
        image, [32, 32, 32], aff=affine, return_pad_idx=True
    )
    assert padded.shape == (32, 32, 32, 2)
    np.testing.assert_array_equal(indices, [7, 7, 6, 24, 25, 25])
    np.testing.assert_allclose(padded_affine[:3, 3], [3, 6, 12])

    restored, restored_affine = crop_volume_with_idx(padded, indices, aff=padded_affine)
    np.testing.assert_array_equal(restored, image)
    np.testing.assert_allclose(restored_affine, affine)
