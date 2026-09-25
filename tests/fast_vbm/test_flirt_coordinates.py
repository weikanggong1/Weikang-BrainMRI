"""FSL scaled-mm and world-RAS coordinate conversion tests."""

import numpy as np
from fnit.flirt.coordinates import (
    flirt_to_world_affine,
    flirt_to_world_pull,
    voxel_to_fsl_scaled_mm,
)


def test_fsl_scaled_mm_uses_storage_handedness_not_world_ras():
    neurological = np.diag([2.0, 3.0, 4.0, 1.0])
    radiological = np.diag([-2.0, 3.0, 4.0, 1.0])

    np.testing.assert_allclose(
        voxel_to_fsl_scaled_mm(neurological, (5, 6, 7)),
        [[-2, 0, 0, 8], [0, 3, 0, 0], [0, 0, 4, 0], [0, 0, 0, 1]],
    )
    np.testing.assert_allclose(
        voxel_to_fsl_scaled_mm(radiological, (5, 6, 7)),
        np.diag([2.0, 3.0, 4.0, 1.0]),
    )


def test_fsl_scaled_mm_can_use_stored_pixdim_for_sheared_sform():
    sheared = np.array(
        [[1.0, 0.3, 0.0, 4.0], [0.0, 2.0, 0.2, -3.0],
         [0.0, 0.0, 3.0, 2.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=float,
    )

    scaled = voxel_to_fsl_scaled_mm(
        sheared, (5, 6, 7), voxel_sizes=(1.0, 2.0, 3.0)
    )

    np.testing.assert_allclose(
        scaled,
        [[-1, 0, 0, 4], [0, 2, 0, 0], [0, 0, 3, 0], [0, 0, 0, 1]],
    )


def test_flirt_scaled_mm_conversion_preserves_coordinate_mapping_and_direction():
    moving_vox2world = np.array(
        [[-2, 0, 0, 30], [0, 3, 0, -12], [0, 0, 4, 8], [0, 0, 0, 1]],
        dtype=float,
    )
    fixed_vox2world = np.array(
        [[1.5, 0, 0, -20], [0, 2.5, 0, 5], [0, 0, 3.5, -9], [0, 0, 0, 1]],
        dtype=float,
    )
    flirt = np.array(
        [[1.02, 0.03, 0, 7], [-0.01, 0.98, 0.02, -4], [0, 0.01, 1.04, 3], [0, 0, 0, 1]],
        dtype=float,
    )
    moving_shape, fixed_shape = (11, 12, 13), (21, 22, 23)
    forward = flirt_to_world_affine(
        flirt,
        moving_vox2world,
        fixed_vox2world,
        moving_shape,
        fixed_shape,
    )
    pull = flirt_to_world_pull(
        flirt,
        moving_vox2world,
        fixed_vox2world,
        moving_shape,
        fixed_shape,
    )
    moving_voxel = np.array([4.0, 5.0, 6.0, 1.0])
    moving_fsl = voxel_to_fsl_scaled_mm(moving_vox2world, moving_shape)
    fixed_fsl = voxel_to_fsl_scaled_mm(fixed_vox2world, fixed_shape)
    expected_fixed_world = (
        fixed_vox2world
        @ np.linalg.inv(fixed_fsl)
        @ flirt
        @ moving_fsl
        @ moving_voxel
    )
    moving_world = moving_vox2world @ moving_voxel

    np.testing.assert_allclose(forward @ moving_world, expected_fixed_world)
    np.testing.assert_allclose(pull @ expected_fixed_world, moving_world)
    np.testing.assert_allclose(pull @ forward, np.eye(4), atol=1e-12)
    assert not np.allclose(forward, flirt)
