"""Analytic tests for the independent PyTorch FLIRT-compatible affine stage."""

import numpy as np
import torch

from freesurfer_torch.fast_vbm.linear import (
    affine_from_parameters,
    flirt_to_world_affine,
    flirt_to_world_pull,
    register_affine,
    resample_to_fixed,
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


def test_affine_parameters_apply_translation_about_moving_center():
    parameters = torch.zeros(12)
    parameters[3:6] = torch.tensor([3.0, -2.0, 1.5])
    moving_center = torch.tensor([5.0, 6.0, 7.0])
    fixed_center = torch.tensor([-1.0, 2.0, 4.0])

    matrix = affine_from_parameters(parameters, moving_center, fixed_center)

    mapped_center = matrix[:3, :3] @ moving_center + matrix[:3, 3]
    torch.testing.assert_close(mapped_center, fixed_center + parameters[3:6])
    torch.testing.assert_close(matrix[:3, :3], torch.eye(3))


def test_resampling_uses_forward_moving_to_fixed_direction():
    moving = torch.zeros((5, 5, 5))
    moving[1, 2, 2] = 1
    forward = torch.eye(4)
    forward[0, 3] = 1

    warped, valid = resample_to_fixed(
        moving, np.eye(4), np.eye(4), moving.shape, forward
    )

    assert torch.argmax(warped).item() == np.ravel_multi_index((2, 2, 2), moving.shape)
    assert warped[2, 2, 2] == 1
    assert not bool(valid[0, 2, 2])
    assert bool(valid[2, 2, 2])


def test_identity_registration_returns_inverse_pair_and_exact_resampling():
    coordinates = np.indices((7, 8, 9), dtype=np.float32)
    image = np.exp(
        -(
            (coordinates[0] - 2.1) ** 2 / 3
            + (coordinates[1] - 4.3) ** 2 / 5
            + (coordinates[2] - 5.2) ** 2 / 4
        )
    ).astype(np.float32)

    result = register_affine(
        image,
        image,
        np.eye(4),
        np.eye(4),
        device="cpu",
        strides=(1,),
        steps=(0,),
        learning_rates=(0.01,),
    )

    np.testing.assert_allclose(result.moving_to_fixed_world, np.eye(4), atol=1e-6)
    np.testing.assert_allclose(result.fixed_to_moving_world, np.eye(4), atol=1e-6)
    torch.testing.assert_close(result.warped, torch.from_numpy(image), atol=1e-6, rtol=0)
    assert result.qc["degrees_of_freedom"] == 12
    assert result.qc["equivalent_to_fsl_flirt"] is False
    assert result.qc["correlation"] > 0.999999


def test_cpu_optimization_is_repeatable():
    coordinates = np.indices((6, 7, 8), dtype=np.float32)
    moving = np.exp(-sum((axis - centre) ** 2 for axis, centre in zip(
        coordinates, (2.0, 3.0, 4.0)
    )) / 3).astype(np.float32)
    fixed = np.roll(moving, 1, axis=0)
    arguments = dict(
        moving=moving,
        fixed=fixed,
        moving_affine=np.eye(4),
        fixed_affine=np.eye(4),
        device="cpu",
        strides=(2, 1),
        steps=(2, 2),
        learning_rates=(0.02, 0.01),
    )

    first = register_affine(**arguments)
    second = register_affine(**arguments)

    np.testing.assert_array_equal(
        first.moving_to_fixed_world, second.moving_to_fixed_world
    )
    torch.testing.assert_close(first.warped, second.warped, atol=0, rtol=0)
    assert first.qc["random_initialization"] is False
