"""Analytic and integration tests for the PyTorch FNIRT-style backend."""

import numpy as np
import pytest
import surfa as sf
import torch

import freesurfer_torch
from freesurfer_torch.fast_vbm.fnirt_backend import (
    FNIRTVBMResult,
    PyTorchFNIRTRegistration,
    _constrain_fnirt_field,
    _coordinate_grid,
    _sample,
    cubic_bspline_field,
)
from freesurfer_torch.fast_vbm.legacy_registration import pull_jacobian
from freesurfer_torch.fast_vbm.registration import register_gm


def _volume(shape=(9, 10, 11), vox2world=None):
    if vox2world is None:
        vox2world = np.eye(4)
    coordinates = np.indices(shape, dtype=np.float32)
    center = (np.asarray(shape, dtype=np.float32) - 1) / 2
    data = np.exp(
        -sum(
            (coordinates[axis] - center[axis]) ** 2 / (4 + axis)
            for axis in range(3)
        )
    ).astype(np.float32)
    return sf.Volume(
        data, geometry=sf.ImageGeometry(shape, vox2world=vox2world)
    )


def _identity(moving, fixed):
    return sf.Affine(np.eye(4), source=moving, target=fixed, space="world")


def test_cubic_bspline_partition_of_unity_preserves_constant_coefficients():
    shape = (9, 10, 11)
    spacing = (3, 4, 5)
    controls = tuple(
        (size - 1) // step + 4 for size, step in zip(shape, spacing)
    )
    values = torch.tensor([1.25, -2.5, 0.75]).reshape(1, 3, 1, 1, 1)
    coefficients = values.expand(1, 3, *controls).clone()

    field = cubic_bspline_field(coefficients, shape, spacing)

    assert field.shape == (1, 3, *shape)
    torch.testing.assert_close(
        field, values.expand_as(field), atol=2e-6, rtol=2e-6
    )


def test_fsl_residual_sampling_and_jacobian_have_expected_axes_and_signs():
    shape = (5, 6, 7)
    positions = tuple(torch.arange(size, dtype=torch.float32) for size in shape)
    fixed_voxel_to_fsl = torch.tensor(
        [[-2.0, 0.0, 0.0, 8.0], [0.0, 1.5, 0.0, 0.0],
         [0.0, 0.0, 2.5, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )
    moving_voxel_to_fsl = torch.tensor(
        [[0.8, 0.0, 0.0, 0.0], [0.0, 1.2, 0.0, 0.0],
         [0.0, 0.0, 1.6, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )
    affine_pull = torch.tensor(
        [[0.9, 0.0, 0.0, 0.2], [0.0, 1.1, 0.0, 0.3],
         [0.0, 0.0, 1.0, 0.4], [0.0, 0.0, 0.0, 1.0]]
    )
    slopes = torch.tensor([0.04, -0.03, 0.02])
    target_fsl = _coordinate_grid(fixed_voxel_to_fsl, positions)
    residual = target_fsl * slopes[None, :, None, None, None]
    moving_axes = torch.meshgrid(
        *(torch.arange(20, dtype=torch.float32) for _ in range(3)),
        indexing="ij",
    )
    moving = (
        moving_axes[0] + 10 * moving_axes[1] + 100 * moving_axes[2]
    )[None, None]

    warped, valid, source_fsl = _sample(
        moving,
        torch.linalg.inv(moving_voxel_to_fsl),
        target_fsl,
        affine_pull,
        residual,
    )
    expected_source_fsl = (
        torch.einsum(
            "ij,bjxyz->bixyz", affine_pull[:3, :3], target_fsl
        )
        + affine_pull[:3, 3][None, :, None, None, None]
        + residual
    )
    expected_source_voxels = torch.einsum(
        "ij,bjxyz->bixyz",
        torch.linalg.inv(moving_voxel_to_fsl)[:3, :3],
        expected_source_fsl,
    )
    actual_source_voxels = torch.einsum(
        "ij,bjxyz->bixyz",
        torch.linalg.inv(moving_voxel_to_fsl)[:3, :3],
        source_fsl,
    )

    torch.testing.assert_close(source_fsl, expected_source_fsl)
    torch.testing.assert_close(actual_source_voxels, expected_source_voxels)
    assert bool(valid.all())
    expected_warped = (
        expected_source_voxels[:, 0]
        + 10 * expected_source_voxels[:, 1]
        + 100 * expected_source_voxels[:, 2]
    )[:, None]
    torch.testing.assert_close(warped, expected_warped, atol=3e-4, rtol=2e-6)
    jacobian = pull_jacobian(
        torch.eye(3), residual, fixed_voxel_to_fsl[:3, :3]
    )
    expected_jacobian = float(torch.prod(1 + slopes))
    torch.testing.assert_close(
        jacobian,
        torch.full_like(jacobian, expected_jacobian),
        atol=2e-6,
        rtol=2e-6,
    )


def test_fnirt_nonlinear_jacobian_is_not_full_jacobian_divided_by_affine():
    shape = (5, 6, 7)
    positions = tuple(torch.arange(size, dtype=torch.float32) for size in shape)
    fixed_voxel_to_fsl = torch.tensor(
        [[-2.0, 0.0, 0.0, 8.0], [0.0, 1.5, 0.0, 0.0],
         [0.0, 0.0, 2.5, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )
    affine_pull = torch.diag(torch.tensor([0.8, 1.2, 1.1, 1.0]))
    slopes = torch.tensor([0.10, -0.05, 0.08])
    target_fsl = _coordinate_grid(fixed_voxel_to_fsl, positions)
    residual = target_fsl * slopes[None, :, None, None, None]

    constrained, full, nonlinear, qc = _constrain_fnirt_field(
        affine_pull,
        residual,
        fixed_voxel_to_fsl,
        jacobian_range=(0.2, 5.0),
    )

    expected_full = float(torch.prod(torch.diag(affine_pull)[:3] + slopes))
    expected_nonlinear = float(torch.prod(1 + slopes))
    affine_determinant = float(torch.det(affine_pull[:3, :3]))
    divided_by_affine = expected_full / affine_determinant
    torch.testing.assert_close(constrained, residual)
    torch.testing.assert_close(
        full, torch.full_like(full, expected_full), atol=2e-6, rtol=2e-6
    )
    torch.testing.assert_close(
        nonlinear,
        torch.full_like(nonlinear, expected_nonlinear),
        atol=2e-6,
        rtol=2e-6,
    )
    assert expected_nonlinear != pytest.approx(divided_by_affine, rel=1e-3)
    assert qc["deformation_scale"] == 1.0


def test_zero_field_preserves_reference_grid_and_has_unit_nonlinear_jacobian():
    moving = _volume()
    fixed_affine = np.array(
        [[0, -1.2, 0, 20], [1.0, 0, 0, -10],
         [0, 0, 1.5, 5], [0, 0, 0, 1]],
        dtype=float,
    )
    fixed = _volume(vox2world=fixed_affine)
    model = PyTorchFNIRTRegistration(
        device="cpu",
        strides=(1,),
        steps=(0,),
        learning_rates=(0.1,),
        input_fwhm_mm=(0,),
        reference_fwhm_mm=(0,),
        regularization=(0,),
        jacobian_penalty=0,
    )

    result = model(moving, fixed, _identity(moving, fixed))

    assert isinstance(result, FNIRTVBMResult)
    assert freesurfer_torch.PyTorchFNIRTRegistration is PyTorchFNIRTRegistration
    assert tuple(result.moved.shape) == tuple(fixed.shape)
    np.testing.assert_allclose(
        result.moved.geom.vox2world.matrix, fixed.geom.vox2world.matrix
    )
    np.testing.assert_allclose(result.nonlinear_jacobian.data, 1, atol=2e-5)
    np.testing.assert_allclose(
        result.modulated_gm.data, result.moved.data, atol=2e-5
    )
    assert sf.transform.image_geometry_equal(
        result.pull_transform.source, moving.geom, tol=1e-5
    )
    assert sf.transform.image_geometry_equal(
        result.pull_transform.target, fixed.geom, tol=1e-5
    )
    assert result.qc["fast_vbm_output_role_compatible"] is True
    assert result.qc["fsl_fnirt_numerically_equivalent"] is False
    assert result.qc["accepts_fsl_coefficient_file"] is False
    assert result.qc["output_jacobian_convention"].startswith(
        "FSL FNIRT nonlinear-only"
    )


def test_affine_only_result_removes_affine_factor_from_modulation_jacobian():
    moving = _volume()
    fixed = moving.copy()
    forward = np.eye(4)
    forward[0, 0] = 1.1
    forward[1, 1] = 0.9
    initial = sf.Affine(forward, source=moving, target=fixed, space="world")
    model = PyTorchFNIRTRegistration(
        device="cpu",
        strides=(1,),
        steps=(0,),
        learning_rates=(0.1,),
        input_fwhm_mm=(0,),
        reference_fwhm_mm=(0,),
        regularization=(0,),
        jacobian_penalty=0,
    )

    result = model(moving, fixed, initial)

    np.testing.assert_allclose(result.nonlinear_jacobian.data, 1, atol=2e-5)
    assert result.affine_pull_determinant == pytest.approx(
        1 / np.linalg.det(forward[:3, :3]), rel=1e-5
    )
    assert result.qc["nonpositive_nonlinear_jacobian_voxels"] == 0


def test_full_jacobian_comparison_accounts_for_sheared_image_bases():
    moving_affine = np.array(
        [[1.0, 0.25, 0.0, 0.0], [0.0, 1.0, 0.1, 0.0],
         [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=float,
    )
    fixed_affine = np.array(
        [[-1.2, 0.15, 0.0, 4.0], [0.0, 1.1, 0.2, -3.0],
         [0.0, 0.0, 0.9, 2.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=float,
    )
    moving = _volume(vox2world=moving_affine)
    fixed = _volume(vox2world=fixed_affine)
    model = PyTorchFNIRTRegistration(
        device="cpu",
        strides=(1,),
        steps=(0,),
        learning_rates=(0.1,),
        input_fwhm_mm=(0,),
        reference_fwhm_mm=(0,),
        regularization=(0,),
        jacobian_penalty=0,
    )

    result = model(moving, fixed, _identity(moving, fixed))

    np.testing.assert_allclose(result.full_pull_jacobian.data, 1, atol=2e-5)
    np.testing.assert_allclose(result.nonlinear_jacobian.data, 1, atol=2e-5)


def test_register_gm_fnirt_branch_returns_fsl_vbm_output_roles():
    moving = _volume()
    fixed = moving.copy()

    result = register_gm(
        moving,
        fixed,
        device="cpu",
        initial_pull=np.eye(4),
        initial_pull_convention="fixed-to-moving-world-ras",
        registration_backend="fnirt",
        fnirt_strides=(1,),
        fnirt_steps=(0,),
        fnirt_learning_rates=(0.1,),
        fnirt_input_fwhm_mm=(0,),
        fnirt_reference_fwhm_mm=(0,),
        fnirt_regularization=(0,),
        fnirt_jacobian_penalty=0,
    )

    assert result.qc["nonlinear_backend"] == "pytorch-fnirt-gm-config"
    assert result.qc["fnirt_style"] is False
    assert result.qc["fsl_fnirt_numerically_equivalent"] is False
    for output in (result.warped_gm, result.jacobian, result.modulated_gm):
        assert output.data.dtype == np.float32
        assert tuple(output.shape) == tuple(fixed.shape)
        np.testing.assert_allclose(
            output.geom.vox2world.matrix, fixed.geom.vox2world.matrix
        )
        assert np.isfinite(output.data).all()


@pytest.mark.parametrize(
    "arguments,message",
    [
        ({"strides": (0,)}, "positive"),
        ({
            "steps": (-1,), "strides": (1,), "learning_rates": 0.1,
            "input_fwhm_mm": 0, "reference_fwhm_mm": 0,
            "regularization": 0,
        }, "non-negative"),
        ({"jacobian_range": (1, 0)}, "increasing"),
        ({"warp_resolution_mm": 0}, "positive"),
    ],
)
def test_fnirt_options_are_validated(arguments, message):
    with pytest.raises(ValueError, match=message):
        PyTorchFNIRTRegistration(**arguments)


def test_fnirt_rejects_single_voxel_spatial_dimension():
    moving = _volume(shape=(1, 9, 10))
    fixed = _volume(shape=(9, 10, 11))

    with pytest.raises(ValueError, match="spatial dimensions must be at least 2"):
        PyTorchFNIRTRegistration(device="cpu")(
            moving, fixed, _identity(moving, fixed)
        )
