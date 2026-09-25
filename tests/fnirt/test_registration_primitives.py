import numpy as np
import pytest
import surfa as sf
import torch

import fnit.fnirt.registration as registration_module
from fnit.fnirt.registration import (
    GMFNIRTConfig,
    TorchFNIRT,
    _fsl_affine_grid,
    _fsl_displacement_coordinates,
    _spline_jacobian,
    _subsampled_size,
    _trilinear_sample,
    spm_like_mean,
)
from fnit.fnirt.spline import fsl_control_shape


def _volume(shape=(8, 8, 8)):
    coordinates = np.indices(shape, dtype=np.float32)
    center = (np.asarray(shape, dtype=np.float32) - 1) / 2
    data = np.exp(
        -sum((axis - center[index]) ** 2 for index, axis in enumerate(coordinates))
        / 8
    ).astype(np.float32)
    return sf.Volume(data, geometry=sf.ImageGeometry(shape, vox2world=np.eye(4)))


def _single_level_config():
    return GMFNIRTConfig(
        subsampling=(1,),
        maximum_iterations=(0,),
        input_fwhm_mm=(0.0,),
        reference_fwhm_mm=(0.0,),
        regularization=(0.0,),
        estimate_intensity=(False,),
        apply_reference_mask=(False,),
        warp_resolution_mm=(4.0, 4.0, 4.0),
    )


def _failed_topology_projection(coefficients, shape, *_):
    jacobian = torch.ones(
        shape, dtype=coefficients.dtype, device=coefficients.device
    )
    qc = {
        "required": True,
        "calls": [],
        "range": [0.204623, 5.04025],
        "succeeded": False,
    }
    return coefficients, jacobian, qc


def test_spm_like_mean_uses_one_eighth_of_whole_image_mean():
    values = np.array([0.0, 1.0, 2.0, 20.0], dtype=np.float32)
    # Whole-volume mean is 5.75; values above 0.71875 are 1, 2 and 20.
    assert spm_like_mean(values) == np.mean([1.0, 2.0, 20.0])


def test_fsl_recursive_subsampling_keeps_endpoint_coverage():
    assert _subsampled_size(91, 4) == 24
    assert _subsampled_size(109, 4) == 28
    assert _subsampled_size(90, 2) == 46


def test_trilinear_sampler_returns_piecewise_analytic_gradient():
    axes = torch.meshgrid(
        torch.arange(5.0), torch.arange(6.0), torch.arange(7.0), indexing="ij"
    )
    volume = 2 * axes[0] - 3 * axes[1] + 0.5 * axes[2] + 7
    coordinates = torch.tensor([2.25, 3.5, 1.75])[:, None, None, None]
    value, valid, gradient = _trilinear_sample(volume, coordinates)
    assert bool(valid)
    torch.testing.assert_close(value.squeeze(), torch.tensor(1.875))
    torch.testing.assert_close(
        gradient.squeeze(), torch.tensor([2.0, -3.0, 0.5])
    )


def test_warpfns_coordinate_arithmetic_uses_scalar_float_order():
    affine = torch.tensor(
        [
            [1.0000001, 0.1000003, -0.2000002, 0.3000004],
            [-0.0500002, 0.9999998, 0.0700001, -0.4000003],
            [0.0300001, -0.0900002, 1.1000003, 0.2000001],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    field = torch.linspace(
        -0.015, 0.021, 3 * 3 * 2 * 2, dtype=torch.float32
    ).reshape(3, 3, 2, 2)
    mm_to_voxel = torch.tensor(
        [
            [0.7, 0.02, -0.01, 0.1],
            [0.01, 0.8, 0.03, -0.2],
            [-0.02, 0.01, 0.9, 0.05],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )

    expected_mm = np.empty_like(field.numpy())
    matrix = affine.numpy().astype(np.float32)
    for i in range(3):
        for j in range(2):
            for k in range(2):
                for row in range(3):
                    value = np.float32(np.float32(i) * matrix[row, 0])
                    value = np.float32(
                        value + np.float32(j) * matrix[row, 1]
                    )
                    value = np.float32(
                        value + np.float32(k) * matrix[row, 2]
                    )
                    expected_mm[row, i, j, k] = np.float32(
                        value + matrix[row, 3]
                    )
    np.testing.assert_array_equal(
        _fsl_affine_grid(affine, (3, 2, 2)).numpy(), expected_mm
    )

    expected_voxels = np.empty_like(expected_mm)
    source_mm = (expected_mm + field.numpy()).astype(np.float32)
    matrix = mm_to_voxel.numpy().astype(np.float32)
    for i in range(3):
        for j in range(2):
            for k in range(2):
                for row in range(3):
                    value = np.float32(source_mm[0, i, j, k] * matrix[row, 0])
                    value = np.float32(
                        value + source_mm[1, i, j, k] * matrix[row, 1]
                    )
                    value = np.float32(
                        value + source_mm[2, i, j, k] * matrix[row, 2]
                    )
                    expected_voxels[row, i, j, k] = np.float32(
                        value + matrix[row, 3]
                    )
    np.testing.assert_array_equal(
        _fsl_displacement_coordinates(
            field, affine, mm_to_voxel
        ).numpy(),
        expected_voxels,
    )


def test_gm_config_rejects_mismatched_schedules():
    try:
        GMFNIRTConfig(maximum_iterations=(1, 2))
    except ValueError as error:
        assert "same" in str(error)
    else:
        raise AssertionError("invalid schedule was accepted")


def test_failed_topology_projection_warns_and_continues_by_default(monkeypatch):
    monkeypatch.setattr(
        registration_module,
        "_force_jacobian_range",
        _failed_topology_projection,
    )
    moving = _volume()
    fixed = moving.copy()
    initial = sf.Affine(
        np.eye(4), source=moving, target=fixed, space="world"
    )

    with pytest.warns(
        RuntimeWarning,
        match=r"Jacobian range was 0\.204623--5\.04025.*continuing as FSL FNIRT",
    ):
        result = TorchFNIRT(device="cpu", config=_single_level_config())(
            moving, fixed, initial
        )

    assert result.qc["strict_topology"] is False
    topology_qc = result.qc["levels"][0]["topology_projection"]
    assert topology_qc["succeeded"] is False
    assert topology_qc["range"] == [0.204623, 5.04025]


def test_failed_topology_projection_raises_in_explicit_strict_mode(monkeypatch):
    monkeypatch.setattr(
        registration_module,
        "_force_jacobian_range",
        _failed_topology_projection,
    )
    moving = _volume()
    fixed = moving.copy()
    initial = sf.Affine(
        np.eye(4), source=moving, target=fixed, space="world"
    )

    with pytest.raises(
        RuntimeError,
        match=r"Jacobian range was 0\.204623--5\.04025",
    ):
        TorchFNIRT(
            device="cpu",
            config=_single_level_config(),
            strict_topology=True,
        )(moving, fixed, initial)


def test_spline_jacobian_matches_reproduced_linear_field():
    shape = (12, 11, 10)
    spacing = (3, 3, 3)
    voxel_sizes = (2.0, 2.5, 3.0)
    control_shape = fsl_control_shape(shape, spacing)
    coefficients = torch.zeros((3, *control_shape), dtype=torch.float64)
    centres = tuple(
        (torch.arange(count, dtype=torch.float64) - 1)
        * knot_spacing
        * voxel_size
        for count, knot_spacing, voxel_size in zip(
            control_shape, spacing, voxel_sizes
        )
    )
    coefficients[0] = 0.1 * centres[0][:, None, None]
    coefficients[1] = -0.05 * centres[1][None, :, None]
    coefficients[2] = 0.2 * centres[2][None, None, :]
    determinant = _spline_jacobian(
        coefficients, shape, spacing, voxel_sizes
    )
    expected = torch.full_like(determinant, 1.1 * 0.95 * 1.2)
    # The retired cubic NCoef rule truncates one still-nonzero spline when
    # (size + 1) is exactly divisible by the knot spacing.  FSL therefore
    # reproduces the linear field on the interior but not on that final plane.
    torch.testing.assert_close(
        determinant[:, :-1], expected[:, :-1], atol=2e-12, rtol=2e-12
    )
    assert bool(torch.isfinite(determinant).all())


def test_full_spline_jacobian_includes_affine_pull():
    shape = (8, 7, 6)
    spacing = (3, 3, 3)
    coefficients = torch.zeros(
        (3, *fsl_control_shape(shape, spacing)), dtype=torch.float64
    )
    affine_pull = torch.tensor(
        [[1.2, 0.1, 0.0], [0.0, 0.9, 0.2], [0.0, 0.0, 1.1]],
        dtype=torch.float64,
    )
    determinant = _spline_jacobian(
        coefficients,
        shape,
        spacing,
        (2.0, 2.0, 2.0),
        affine_pull_linear=affine_pull,
    )
    torch.testing.assert_close(
        determinant,
        torch.full_like(determinant, torch.linalg.det(affine_pull)),
    )
