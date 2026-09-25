"""Focused Talairach affine conversion and standalone-stage tests."""

import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import surfa as sf


MODULE = Path(os.environ.get(
    "TALAIRACH_MODULE_PATH",
    Path(__file__).resolve().parents[2] / "src/fnit/recon_all/talairach_synthmorph.py"))
spec = importlib.util.spec_from_file_location("talairach_synthmorph", MODULE)
talairach = importlib.util.module_from_spec(spec)
spec.loader.exec_module(talairach)


def _affine():
    source = np.eye(4)
    source[:3, :3] = [[-1, 0, 0], [0, 0, 1], [0, 1, 0]]
    source[:3, 3] = [4, -6, 2]
    target = np.eye(4)
    target[:3, :3] = [[-1, 0, 0], [0, 0, -1], [0, 1, 0]]
    moving = sf.Volume(np.zeros((16, 18, 20), np.float32),
                       geometry=sf.ImageGeometry((16, 18, 20), vox2world=source))
    fixed = sf.Volume(np.zeros((16, 18, 20), np.float32),
                      geometry=sf.ImageGeometry((16, 18, 20), vox2world=target))
    matrix = np.array([[1.1, 0.03, 0.01, 3.5],
                       [-0.02, 0.95, 0.04, -5.25],
                       [0.01, -0.01, 1.2, 2.75],
                       [0, 0, 0, 1]], np.float32)
    return sf.Affine(matrix, source=moving, target=fixed, space="world")


def test_talairach_conversion_preserves_world_coordinate_mapping():
    affine = _affine()
    matrix = talairach.talairach_matrix(affine)
    np.testing.assert_allclose(matrix[:3], affine.matrix[:3], atol=1e-5, rtol=0)


def test_register_talairach_writes_readable_xfm_without_native_program(tmp_path, monkeypatch):
    affine = _affine()
    called = {}

    class FakeModel:
        def __init__(self, **kwargs):
            called.update(kwargs)

        def __call__(self, moving, template):
            called["input"] = (moving, template)
            return SimpleNamespace(transform=affine)

    monkeypatch.setattr(talairach, "SynthMorph", FakeModel)
    path = tmp_path / "transforms/talairach.xfm"
    matrix = talairach.register_talairach(
        "synthstrip.mgz", "mni305.cor.stripped.mgz", "weights", path, threads=1)
    lines = path.read_text().splitlines()
    assert lines[0] == "MNI Transform File"
    assert lines[4] == "Linear_Transform ="
    assert lines[7].endswith(";")
    parsed = np.array([[float(value) for value in line.rstrip(";").split()]
                       for line in lines[5:8]])
    np.testing.assert_allclose(parsed, matrix[:3], atol=5e-9, rtol=0)
    assert called == {"weights": "weights", "device": "cpu", "model": "affine",
                      "extent": 256,
                      "input": ("synthstrip.mgz", "mni305.cor.stripped.mgz")}
