"""FastVBM registration input and displacement-QC regression tests."""

from types import SimpleNamespace

import numpy as np
import pytest
import surfa as sf

from fnit.fast_vbm.registration import (
    _displacement_qc,
    _register_gm,
)


def _volume(shape=(6, 7, 8), affine=None):
    if affine is None:
        affine = np.eye(4)
    geometry = sf.ImageGeometry(shape, vox2world=affine)
    return sf.Volume(np.ones(shape, dtype=np.float32), geometry=geometry)


def _identity_deform(moving, fixed, _initial):
    shape = tuple(fixed.shape[:3])
    moved = fixed.new(np.asarray(fixed.data, dtype=np.float32).copy())
    jacobian = fixed.new(np.ones(shape, dtype=np.float32))
    pull = sf.Warp(
        np.zeros((*shape, 3), dtype=np.float32),
        source=moving,
        target=fixed,
        format=sf.Warp.Format.disp_ras,
    )
    return SimpleNamespace(
        moved=moved,
        pull_transform=pull,
        nonlinear_jacobian=jacobian,
        modulated_gm=moved.copy(),
        qc={},
    )


def test_internal_registration_rejects_empty_volume():
    shape = (6, 7, 8)
    geometry = sf.ImageGeometry(shape, vox2world=np.eye(4))
    empty = sf.Volume(np.zeros(shape, dtype=np.float32), geometry=geometry)
    positive = sf.Volume(np.ones(shape, dtype=np.float32), geometry=geometry)

    with pytest.raises(ValueError, match="GM image is empty"):
        _register_gm(empty, positive)


def test_internal_affine_requires_geometry_tagged_surfa_affine():
    volume = _volume()

    with pytest.raises(TypeError, match="geometry-tagged surfa.Affine"):
        _register_gm(volume, volume, initial_pull=np.eye(4))


def test_geometry_tagged_world_pull_is_accepted():
    moving = _volume()
    fixed = moving.copy()
    tagged = sf.Affine(
        np.eye(4), source=fixed, target=moving, space="world"
    )

    result = _register_gm(
        moving,
        fixed,
        initial_pull=tagged,
        deform_model=_identity_deform,
    )

    np.testing.assert_allclose(result.pull_world_affine, np.eye(4))
    assert result.qc["linear"]["pull_transform_convention"] == (
        "fixed-to-moving-world-ras"
    )
    assert result.qc["nonlinear_backend"] == "custom-deform-model"


def test_geometry_tagged_forward_affine_is_not_accepted_as_a_pull():
    moving = _volume()
    fixed_affine = np.eye(4)
    fixed_affine[0, 3] = 5
    fixed = _volume(affine=fixed_affine)
    forward = sf.Affine(
        np.eye(4), source=moving, target=fixed, space="world"
    )

    with pytest.raises(ValueError, match="source geometry must match fixed"):
        _register_gm(
            moving,
            fixed,
            initial_pull=forward,
            deform_model=_identity_deform,
        )


def test_displacement_qc_separates_affine_and_nonlinear_components():
    shape = (9, 7, 5)
    vox2world = np.array(
        [
            [1.5, 0.2, 0, -4],
            [0, 2, 0.1, 3],
            [0, 0, -2.5, 8],
            [0, 0, 0, 1],
        ],
        dtype=np.float64,
    )
    geometry = sf.ImageGeometry(shape, vox2world=vox2world)
    fixed = sf.Volume(np.ones(shape, dtype=np.float32), geometry=geometry)
    pull_affine = np.array(
        [
            [1.1, 0.05, 0, 2],
            [0, 0.9, 0.02, -1],
            [0, 0, 1.05, 0.5],
            [0, 0, 0, 1],
        ],
        dtype=np.float64,
    )
    indices = np.stack(
        np.meshgrid(*[np.arange(size) for size in shape], indexing="ij"), axis=-1
    )
    world = np.einsum("ab,...b->...a", vox2world[:3, :3], indices)
    world += vox2world[:3, 3]
    affine_source = np.einsum("ab,...b->...a", pull_affine[:3, :3], world)
    affine_source += pull_affine[:3, 3]
    residual = np.array([0.25, -0.5, 0.75])
    displacement = affine_source - world + residual
    pull = SimpleNamespace(data=displacement.astype(np.float32))

    qc = _displacement_qc(pull, fixed, pull_affine)

    expected_total = np.linalg.norm(displacement, axis=-1).max()
    assert qc["maximum_total_pull_displacement_mm"] == pytest.approx(expected_total)
    assert qc["maximum_nonlinear_displacement_mm"] == pytest.approx(
        np.linalg.norm(residual), abs=1e-6
    )
