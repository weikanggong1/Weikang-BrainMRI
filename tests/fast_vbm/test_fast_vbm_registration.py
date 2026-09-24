"""Registration math and compatibility-wrapper regression tests."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import surfa as sf
import torch

from freesurfer_torch.fast_vbm.registration import (
    pull_jacobian,
    register,
    register_gm,
)


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
        register_gm(empty, positive, affine_steps=0, deform_steps=0, scales=(1,))
