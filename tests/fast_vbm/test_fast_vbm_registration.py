"""Registration math and compatibility-wrapper regression tests."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import surfa as sf
import torch

from freesurfer_torch.fast_vbm.registration import (
    _displacement_qc,
    pull_jacobian,
    register,
    register_gm,
)
from freesurfer_torch.fast_vbm.linear import WORLD_PULL_CONVENTION


TOOLS = Path(__file__).resolve().parents[2] / "tools" / "experimental" / "ukb_vbm"


def _load_wrapper():
    path = TOOLS / "gpu_register.py"
    spec = importlib.util.spec_from_file_location("gpu_register_wrapper", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pull_jacobian_identity_and_affine_scale():
    identity = torch.eye(3)
    field = torch.zeros((1, 3, 7, 8, 9))
    np.testing.assert_allclose(
        pull_jacobian(identity, field, identity).numpy(), 1, atol=1e-6
    )
    scaled = identity.clone()
    scaled[0, 0] = 1.2
    np.testing.assert_allclose(
        pull_jacobian(scaled, field, identity).numpy(), 1.2, atol=1e-6
    )


def test_legacy_wrapper_uses_package_backend_and_preserves_arrays():
    wrapper = _load_wrapper()
    assert wrapper.register is register
    shape = (7, 8, 9)
    moving = torch.arange(np.prod(shape), dtype=torch.float32).reshape(
        1, 1, *shape
    )
    fixed = moving.clone()
    identity = torch.eye(4)
    pull = identity.clone()
    pull[0, 3] = 1

    warped, jacobian, field, affine, _ = wrapper.register(
        moving,
        fixed,
        identity,
        identity,
        initial_pull=pull,
        scales=(1,),
        affine_steps=0,
        deform_steps=0,
    )

    expected = torch.zeros(shape)
    expected[:-1] = moving[0, 0, 1:]
    torch.testing.assert_close(warped, expected, atol=2e-4, rtol=0)
    torch.testing.assert_close(jacobian, torch.ones_like(jacobian))
    torch.testing.assert_close(field, torch.zeros_like(field))
    torch.testing.assert_close(affine, pull)


def test_legacy_wrapper_self_test_runs(capsys):
    wrapper = _load_wrapper()
    wrapper.self_test(torch.device("cpu"))
    assert '"jacobian_identity": 1.0' in capsys.readouterr().out


def test_register_gm_rejects_empty_volume():
    shape = (6, 7, 8)
    geometry = sf.ImageGeometry(shape, vox2world=np.eye(4))
    empty = sf.Volume(np.zeros(shape, dtype=np.float32), geometry=geometry)
    positive = sf.Volume(np.ones(shape, dtype=np.float32), geometry=geometry)
    with pytest.raises(ValueError, match="GM image is empty"):
        register_gm(empty, positive)


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


def test_bare_initial_pull_requires_explicit_world_ras_convention():
    shape = (6, 7, 8)
    geometry = sf.ImageGeometry(shape, vox2world=np.eye(4))
    volume = sf.Volume(np.ones(shape, dtype=np.float32), geometry=geometry)

    with pytest.raises(ValueError, match="coordinate-ambiguous"):
        register_gm(volume, volume, initial_pull=np.eye(4))
    with pytest.raises(TypeError, match="FLIRT .mat"):
        register_gm(
            volume,
            volume,
            initial_pull="transform.mat",
            initial_pull_convention=WORLD_PULL_CONVENTION,
        )


def test_explicit_world_pull_and_geometry_tagged_pull_are_accepted():
    shape = (6, 7, 8)
    geometry = sf.ImageGeometry(shape, vox2world=np.eye(4))
    moving = sf.Volume(np.ones(shape, dtype=np.float32), geometry=geometry)
    fixed = moving.copy()

    explicit = register_gm(
        moving,
        fixed,
        initial_pull=np.eye(4),
        initial_pull_convention=WORLD_PULL_CONVENTION,
        deform_model=_identity_deform,
    )
    tagged = register_gm(
        moving,
        fixed,
        initial_pull=sf.Affine(
            np.eye(4), source=fixed, target=moving, space="world"
        ),
        deform_model=_identity_deform,
    )

    np.testing.assert_allclose(explicit.pull_world_affine, np.eye(4))
    np.testing.assert_allclose(tagged.pull_world_affine, np.eye(4))
    assert explicit.qc["linear"]["pull_transform_convention"] == (
        WORLD_PULL_CONVENTION
    )
    assert explicit.qc["nonlinear_backend"] == "custom-deform-model"
    assert explicit.qc["synthmorph_implementation"] is None


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


def test_geometry_tagged_forward_affine_is_not_accepted_as_a_pull():
    shape = (6, 7, 8)
    moving_geometry = sf.ImageGeometry(shape, vox2world=np.eye(4))
    fixed_affine = np.eye(4)
    fixed_affine[0, 3] = 5
    fixed_geometry = sf.ImageGeometry(shape, vox2world=fixed_affine)
    moving = sf.Volume(np.ones(shape, dtype=np.float32), geometry=moving_geometry)
    fixed = sf.Volume(np.ones(shape, dtype=np.float32), geometry=fixed_geometry)
    forward = sf.Affine(
        np.eye(4), source=moving, target=fixed, space="world"
    )

    with pytest.raises(ValueError, match="source geometry must match fixed"):
        register_gm(
            moving,
            fixed,
            initial_pull=forward,
            deform_model=_identity_deform,
        )
