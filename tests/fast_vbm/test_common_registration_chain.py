"""Tests for the backend-independent FastVBM registration chain."""

from types import SimpleNamespace

import numpy as np
import surfa as sf
import torch

import freesurfer_torch.fast_vbm.registration as registration_module
from freesurfer_torch.fast_vbm.synthmorph_backend import (
    SynthMorphDeformRegistration,
)
from freesurfer_torch.fnirt import TorchFNIRT
from freesurfer_torch.fast_vbm.linear import (
    WORLD_PULL_CONVENTION,
    voxel_to_fsl_scaled_mm,
    world_to_flirt_affine,
)
from freesurfer_torch.fast_vbm.registration import (
    _fsl_dense_nonlinear_jacobian,
    _pull_ras_to_fsl_fields,
    register_gm,
)


def _volume(shape=(7, 8, 9), affine=None):
    if affine is None:
        affine = np.array(
            [
                [1.2, 0.1, 0.0, -8.0],
                [0.0, 1.5, 0.2, 3.0],
                [0.0, 0.0, 1.8, -4.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
    grid = np.stack(
        np.meshgrid(
            *(np.arange(size, dtype=np.float32) for size in shape),
            indexing="ij",
        ),
        axis=-1,
    )
    centre = (np.asarray(shape, dtype=np.float32) - 1) / 2
    data = np.exp(-np.square(grid - centre).sum(-1) / 12).astype(np.float32)
    return sf.Volume(data, geometry=sf.ImageGeometry(shape, vox2world=affine))


def _pull_from_fsl_residual(moving, fixed, forward_fsl, residual):
    shape = tuple(fixed.shape[:3])
    target_voxels = np.stack(
        np.meshgrid(*(np.arange(size) for size in shape), indexing="ij"),
        axis=-1,
    )
    moving_world = np.asarray(moving.geom.vox2world.matrix)
    fixed_world = np.asarray(fixed.geom.vox2world.matrix)
    moving_fsl = voxel_to_fsl_scaled_mm(
        moving_world, moving.shape[:3], moving.geom.voxsize
    )
    fixed_fsl = voxel_to_fsl_scaled_mm(
        fixed_world, fixed.shape[:3], fixed.geom.voxsize
    )
    target_fsl = np.einsum(
        "ab,...b->...a", fixed_fsl[:3, :3], target_voxels
    ) + fixed_fsl[:3, 3]
    affine_pull = np.linalg.inv(forward_fsl)
    source_fsl = np.einsum(
        "ab,...b->...a", affine_pull[:3, :3], target_fsl
    ) + affine_pull[:3, 3]
    source_fsl += residual
    source_voxels = np.einsum(
        "ab,...b->...a", np.linalg.inv(moving_fsl)[:3, :3], source_fsl
    ) + np.linalg.inv(moving_fsl)[:3, 3]
    source_world = np.einsum(
        "ab,...b->...a", moving_world[:3, :3], source_voxels
    ) + moving_world[:3, 3]
    target_world = np.einsum(
        "ab,...b->...a", fixed_world[:3, :3], target_voxels
    ) + fixed_world[:3, 3]
    return sf.Warp(
        (source_world - target_world).astype(np.float32),
        source=moving,
        target=fixed,
        format=sf.Warp.Format.disp_ras,
    )


def test_synthmorph_ras_pull_conversion_recovers_fsl_residual():
    moving = _volume()
    fixed = _volume()
    forward = np.array(
        [
            [1.03, 0.02, 0.00, 1.2],
            [0.01, 0.97, 0.03, -0.7],
            [0.00, 0.01, 1.04, 0.4],
            [0.00, 0.00, 0.00, 1.0],
        ]
    )
    shape = tuple(fixed.shape[:3])
    target_voxels = np.stack(
        np.meshgrid(*(np.arange(size) for size in shape), indexing="ij"),
        axis=-1,
    )
    fixed_fsl = voxel_to_fsl_scaled_mm(
        fixed.geom.vox2world.matrix, shape, fixed.geom.voxsize
    )
    target_fsl = np.einsum(
        "ab,...b->...a", fixed_fsl[:3, :3], target_voxels
    ) + fixed_fsl[:3, 3]
    linear = np.array(
        [[0.015, -0.004, 0.002], [0.003, -0.010, 0.001], [0.0, 0.002, 0.008]]
    )
    residual = np.einsum("ab,...b->...a", linear, target_fsl)
    residual += np.array([0.2, -0.1, 0.05])
    pull = _pull_from_fsl_residual(moving, fixed, forward, residual)

    recovered, dense, recovered_fsl = _pull_ras_to_fsl_fields(
        pull, moving, fixed, forward, device=torch.device("cpu")
    )

    np.testing.assert_allclose(recovered.numpy(), residual, atol=7e-6, rtol=0)
    np.testing.assert_allclose(recovered_fsl, fixed_fsl, atol=0, rtol=0)
    assert dense.shape == recovered.shape


def test_common_dense_jacobian_uses_fsl_residual_and_excludes_affine():
    shape = (7, 8, 9)
    fixed_fsl = np.diag([-1.2, 1.5, 1.8, 1.0])
    fixed_fsl[0, 3] = 1.2 * (shape[0] - 1)
    axes = torch.meshgrid(
        *(torch.arange(size, dtype=torch.float32) for size in shape),
        indexing="ij",
    )
    voxels = torch.stack(axes).reshape(3, -1)
    fsl = torch.as_tensor(fixed_fsl, dtype=torch.float32)
    coordinates = (fsl[:3, :3] @ voxels + fsl[:3, 3:4]).reshape(3, *shape)
    derivative = torch.tensor(
        [[0.02, -0.01, 0.00], [0.01, 0.03, 0.01], [0.00, -0.02, -0.01]],
        dtype=torch.float32,
    )
    residual = torch.einsum("ab,bxyz->axyz", derivative, coordinates)
    jacobian = _fsl_dense_nonlinear_jacobian(
        residual.movedim(0, -1), fixed_fsl
    )
    expected = torch.linalg.det(torch.eye(3) + derivative)

    torch.testing.assert_close(
        jacobian, torch.full(shape, expected), atol=2e-6, rtol=0
    )


class _CaptureEstimator:
    def __init__(self):
        self.calls = []

    def __call__(self, moving, fixed, initial):
        self.calls.append(
            {
                "moving": np.asarray(moving.data).copy(),
                "fixed": np.asarray(fixed.data).copy(),
                "initial": np.asarray(initial.matrix).copy(),
                "moving_geometry": np.asarray(
                    moving.geom.vox2world.matrix
                ).copy(),
                "fixed_geometry": np.asarray(fixed.geom.vox2world.matrix).copy(),
            }
        )
        forward_fsl = world_to_flirt_affine(
            initial.matrix,
            moving.geom.vox2world.matrix,
            fixed.geom.vox2world.matrix,
            moving.shape[:3],
            fixed.shape[:3],
            moving.geom.voxsize,
            fixed.geom.voxsize,
        )
        residual = np.zeros((*fixed.shape[:3], 3), dtype=np.float32)
        pull = _pull_from_fsl_residual(moving, fixed, forward_fsl, residual)
        return SimpleNamespace(pull_transform=pull, qc={"capture": True})


def test_synthmorph_and_fnirt_enter_estimator_with_identical_preparation():
    moving = _volume()
    fixed = _volume()
    pull_world = np.array(
        [
            [0.98, 0.01, 0.00, -0.4],
            [0.00, 1.02, -0.01, 0.3],
            [0.01, 0.00, 1.01, -0.2],
            [0.00, 0.00, 0.00, 1.0],
        ]
    )
    synthmorph = _CaptureEstimator()
    fnirt = _CaptureEstimator()
    reference_mask = fixed.new(
        (np.asarray(fixed.data) > 0.15).astype(np.uint8)
    )
    common = {
        "device": "cpu",
        "initial_pull": pull_world,
        "initial_pull_convention": WORLD_PULL_CONVENTION,
        "reference_mask": reference_mask,
    }

    synthmorph_result = register_gm(
        moving,
        fixed,
        registration_backend="synthmorph",
        deform_model=synthmorph,
        **common,
    )
    fnirt_result = register_gm(
        moving,
        fixed,
        registration_backend="fnirt",
        deform_model=fnirt,
        **common,
    )

    for name in synthmorph.calls[0]:
        np.testing.assert_array_equal(
            synthmorph.calls[0][name], fnirt.calls[0][name]
        )
    assert (
        synthmorph_result.qc["pre_nonlinear_signature"]
        == fnirt_result.qc["pre_nonlinear_signature"]
    )
    assert synthmorph_result.qc["only_backend_specific_stage"] == (
        "nonlinear pull-field estimation, including estimator-specific "
        "objective and mask use"
    )
    assert synthmorph_result.qc["reference_mask_source"] == "explicit"
    assert "passed to FNIRT" in synthmorph_result.qc["reference_mask_role"]
    assert synthmorph_result.qc["fsl_reference_mask_exact"] is None
    assert synthmorph_result.qc["fsl_fnirt_numerically_equivalent"] is None
    assert synthmorph_result.qc["resampling"] == (
        "freesurfer_torch.applywarp.TorchApplyWarp"
    )
    np.testing.assert_allclose(synthmorph_result.jacobian.data, 1, atol=3e-6)
    np.testing.assert_array_equal(
        synthmorph_result.warped_gm.data, fnirt_result.warped_gm.data
    )
    np.testing.assert_array_equal(
        synthmorph_result.modulated_gm.data, fnirt_result.modulated_gm.data
    )


def test_real_backend_dispatch_uses_the_same_default_flirt_and_common_tail(
    monkeypatch,
):
    moving = _volume()
    fixed = _volume()
    reference_mask = fixed.new(
        (np.asarray(fixed.data) > 0.15).astype(np.uint8)
    )
    flirt_calls = []
    synth_calls = []
    fnirt_calls = []

    class FakeFLIRT:
        def __init__(self, *, device):
            self.device = device

        def __call__(self, moving_value, fixed_value):
            flirt_calls.append(
                (
                    np.asarray(moving_value.data).copy(),
                    np.asarray(fixed_value.data).copy(),
                )
            )
            matrix = world_to_flirt_affine(
                np.eye(4),
                moving_value.geom.vox2world.matrix,
                fixed_value.geom.vox2world.matrix,
                moving_value.shape[:3],
                fixed_value.shape[:3],
                moving_value.geom.voxsize,
                fixed_value.geom.voxsize,
            )
            return SimpleNamespace(
                matrix=matrix,
                moving_to_fixed_world=np.eye(4),
                fixed_to_moving_world=np.eye(4),
                qc={"validated_fsl_equivalent": False},
            )

    class FakeSynthMorph(SynthMorphDeformRegistration):
        def __init__(self):
            def model(moving_value, fixed_value, *, init, mid_space):
                synth_calls.append(
                    {
                        "moving": np.asarray(moving_value.data).copy(),
                        "fixed": np.asarray(fixed_value.data).copy(),
                        "initial": np.asarray(init.matrix).copy(),
                        "mid_space": mid_space,
                    }
                )
                forward = world_to_flirt_affine(
                    init.matrix,
                    moving_value.geom.vox2world.matrix,
                    fixed_value.geom.vox2world.matrix,
                    moving_value.shape[:3],
                    fixed_value.shape[:3],
                    moving_value.geom.voxsize,
                    fixed_value.geom.voxsize,
                )
                residual = np.zeros(
                    (*fixed_value.shape[:3], 3), dtype=np.float32
                )
                return SimpleNamespace(
                    transform=_pull_from_fsl_residual(
                        moving_value, fixed_value, forward, residual
                    )
                )

            self.synthmorph = model

    class FakeFNIRT(TorchFNIRT):
        def __init__(self):
            pass

        def __call__(
            self, moving_value, fixed_value, initial,
            *, reference_mask,
        ):
            fnirt_calls.append(
                {
                    "moving": np.asarray(moving_value.data).copy(),
                    "fixed": np.asarray(fixed_value.data).copy(),
                    "initial": np.asarray(initial.matrix).copy(),
                    "reference_mask": np.asarray(reference_mask.data).copy(),
                }
            )
            forward = world_to_flirt_affine(
                initial.matrix,
                moving_value.geom.vox2world.matrix,
                fixed_value.geom.vox2world.matrix,
                moving_value.shape[:3],
                fixed_value.shape[:3],
                moving_value.geom.voxsize,
                fixed_value.geom.voxsize,
            )
            residual = np.zeros(
                (*fixed_value.shape[:3], 3), dtype=np.float32
            )
            return SimpleNamespace(
                pull_transform=_pull_from_fsl_residual(
                    moving_value, fixed_value, forward, residual
                ),
                nonlinear_jacobian=fixed_value.new(
                    np.ones(fixed_value.shape[:3], dtype=np.float32)
                ),
                qc={"fsl_fnirt_numerically_equivalent": False},
            )

    monkeypatch.setattr(registration_module, "FSLFLIRT", FakeFLIRT)
    synth_result = register_gm(
        moving,
        fixed,
        device="cpu",
        reference_mask=reference_mask,
        registration_backend="synthmorph",
        deform_model=FakeSynthMorph(),
    )
    fnirt_result = register_gm(
        moving,
        fixed,
        device="cpu",
        reference_mask=reference_mask,
        registration_backend="fnirt",
        deform_model=FakeFNIRT(),
    )

    assert len(flirt_calls) == 2
    for index in (0, 1):
        np.testing.assert_array_equal(flirt_calls[0][index], flirt_calls[1][index])
    np.testing.assert_array_equal(synth_calls[0]["moving"], fnirt_calls[0]["moving"])
    np.testing.assert_array_equal(synth_calls[0]["fixed"], fnirt_calls[0]["fixed"])
    np.testing.assert_array_equal(synth_calls[0]["initial"], fnirt_calls[0]["initial"])
    np.testing.assert_array_equal(
        fnirt_calls[0]["reference_mask"], np.asarray(reference_mask.data)
    )
    assert synth_calls[0]["mid_space"] is False
    assert (
        synth_result.qc["pre_nonlinear_signature"]
        == fnirt_result.qc["pre_nonlinear_signature"]
    )
    np.testing.assert_array_equal(
        synth_result.warped_gm.data, fnirt_result.warped_gm.data
    )
    np.testing.assert_array_equal(
        synth_result.jacobian.data, fnirt_result.jacobian.data
    )
    np.testing.assert_array_equal(
        synth_result.modulated_gm.data, fnirt_result.modulated_gm.data
    )
