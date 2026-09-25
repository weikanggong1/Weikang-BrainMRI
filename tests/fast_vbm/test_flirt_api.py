"""FSL coordinate and image I/O contracts for the PyTorch FLIRT API."""

import numpy as np
import pytest
import surfa as sf

import freesurfer_torch
from freesurfer_torch.flirt import FLIRTResult, LegacyTorchFLIRT, TorchFLIRT
from freesurfer_torch.fast_vbm.linear import (
    flirt_to_world_affine,
    world_to_flirt_affine,
)


def _volume(shape=(7, 8, 9), vox2world=None):
    if vox2world is None:
        vox2world = np.eye(4)
    axes = np.meshgrid(
        *[np.arange(size, dtype=np.float32) for size in shape], indexing="ij"
    )
    center = (np.asarray(shape, dtype=np.float32) - 1) / 2
    data = np.exp(
        -sum((axis - center[index]) ** 2 for index, axis in enumerate(axes)) / 6
    ).astype(np.float32)
    return sf.Volume(data, geometry=sf.ImageGeometry(shape, vox2world=vox2world))


def test_world_and_flirt_matrix_conversions_are_exact_inverses():
    moving_affine = np.array(
        [[-2, 0, 0, 30], [0, 3, 0, -12], [0, 0, 4, 8], [0, 0, 0, 1]],
        dtype=float,
    )
    fixed_affine = np.array(
        [[1.5, 0, 0, -20], [0, 2.5, 0, 5], [0, 0, 3.5, -9], [0, 0, 0, 1]],
        dtype=float,
    )
    world = np.array(
        [[1.02, 0.03, 0, 7], [-0.01, 0.98, 0.02, -4],
         [0, 0.01, 1.04, 3], [0, 0, 0, 1]],
        dtype=float,
    )
    moving_shape, fixed_shape = (11, 12, 13), (21, 22, 23)

    flirt = world_to_flirt_affine(
        world, moving_affine, fixed_affine, moving_shape, fixed_shape
    )
    recovered = flirt_to_world_affine(
        flirt, moving_affine, fixed_affine, moving_shape, fixed_shape
    )

    np.testing.assert_allclose(recovered, world, atol=1e-12, rtol=1e-12)


def test_legacy_torch_flirt_returns_reference_grid_and_fsl_matrix():
    moving = _volume()
    fixed_affine = np.array(
        [[0, -1.2, 0, 20], [1.0, 0, 0, -10],
         [0, 0, 1.5, 5], [0, 0, 0, 1]],
        dtype=float,
    )
    fixed = _volume((8, 9, 10), fixed_affine)
    result = LegacyTorchFLIRT(
        device="cpu", strides=(1,), steps=(0,), learning_rates=(0.01,)
    )(moving, fixed)

    assert isinstance(result, FLIRTResult)
    assert freesurfer_torch.LegacyTorchFLIRT is LegacyTorchFLIRT
    assert freesurfer_torch.TorchFLIRT is TorchFLIRT
    assert freesurfer_torch.FLIRTResult is FLIRTResult
    assert freesurfer_torch.world_to_flirt_affine is world_to_flirt_affine
    assert tuple(result.moved.shape) == tuple(fixed.shape)
    np.testing.assert_allclose(
        result.moved.geom.vox2world.matrix, fixed.geom.vox2world.matrix
    )
    np.testing.assert_allclose(
        flirt_to_world_affine(
            result.matrix,
            moving.geom.vox2world.matrix,
            fixed.geom.vox2world.matrix,
            moving.shape[:3],
            fixed.shape[:3],
        ),
        result.moving_to_fixed_world,
        atol=1e-12,
        rtol=1e-12,
    )
    assert result.qc["image_output_grid"] == "fixed/reference"
    assert result.qc["matrix_coordinate_system"] == "FSL scaled-mm"
    assert result.qc["matrix_direction"] == "moving/input-to-fixed/reference"
    assert result.qc["equivalent_to_fsl_flirt_algorithm"] is False


def test_run_accepts_paths_and_saves_flirt_style_outputs(tmp_path):
    moving = _volume()
    fixed = _volume()
    moving_path = tmp_path / "moving.nii.gz"
    fixed_path = tmp_path / "reference.nii.gz"
    output_path = tmp_path / "moved.nii.gz"
    matrix_path = tmp_path / "moving_to_reference.mat"
    moving.save(moving_path)
    fixed.save(fixed_path)

    result = LegacyTorchFLIRT(
        device="cpu", strides=(1,), steps=(0,), learning_rates=(0.01,)
    ).run(
        moving_path,
        fixed_path,
        output=output_path,
        omat=matrix_path,
    )

    saved = sf.load_volume(output_path)
    saved_matrix = np.loadtxt(matrix_path)
    np.testing.assert_allclose(
        saved.geom.vox2world.matrix, fixed.geom.vox2world.matrix
    )
    np.testing.assert_allclose(saved.data, result.moved.data, atol=0, rtol=0)
    np.testing.assert_allclose(saved_matrix, result.matrix, atol=1e-11, rtol=1e-11)


def test_legacy_torch_flirt_rejects_non_volume_inputs():
    model = LegacyTorchFLIRT(
        device="cpu", strides=(1,), steps=(0,), learning_rates=(0.01,)
    )

    try:
        model(np.zeros((4, 4, 4)), _volume())
    except TypeError as error:
        assert str(error) == "moving must be a path or surfa.Volume"
    else:
        raise AssertionError("non-volume input was accepted")


def test_flirt_result_rejects_one_path_for_image_and_matrix(tmp_path):
    volume = _volume()
    result = FLIRTResult(
        moved=volume,
        matrix=np.eye(4),
        moving_to_fixed_world=np.eye(4),
        fixed_to_moving_world=np.eye(4),
        qc={},
    )
    destination = tmp_path / "output.nii.gz"

    with pytest.raises(ValueError, match="different paths"):
        result.save(output=destination, omat=destination)

    assert not destination.exists()
