"""Single-subject CLI checks for the PyTorch FLIRT interface."""

from pathlib import Path

import numpy as np
import pytest

from freesurfer_torch import cli
from freesurfer_torch import fast_vbm as fast_vbm_module


class _Moved:
    def save(self, path):
        Path(path).write_text("moved")


def test_flirt_cli_uses_fsl_argument_names_and_writes_both_outputs(
    tmp_path, monkeypatch, capsys
):
    captured = {}

    class Model:
        def __init__(self, **options):
            captured["options"] = options

        def __call__(self, moving, fixed):
            captured["call"] = (moving, fixed)
            return type(
                "Result",
                (),
                {"moved": _Moved(), "matrix": np.eye(4)},
            )()

    monkeypatch.setattr(fast_vbm_module, "TorchFLIRT", Model)
    output = tmp_path / "moved.nii.gz"
    matrix = tmp_path / "transform.mat"

    cli.main([
        "flirt", "-in", "moving.nii.gz", "-ref", "fixed.nii.gz",
        "-out", str(output), "-omat", str(matrix), "-dof", "12",
        "-cost", "normcorr", "--device", "cuda:1", "--threads", "2",
    ])

    assert captured["call"] == ("moving.nii.gz", "fixed.nii.gz")
    assert captured["options"] == {
        "device": "cuda:1",
        "strides": (4, 2, 1),
        "steps": (80, 60, 50),
        "learning_rates": (0.05, 0.025, 0.0125),
        "cost": "normcorr",
    }
    assert output.read_text() == "moved"
    np.testing.assert_allclose(np.loadtxt(matrix), np.eye(4))
    assert capsys.readouterr().out.splitlines() == [str(output), str(matrix)]


def test_flirt_cli_requires_an_output():
    try:
        cli.main(["flirt", "-in", "moving.nii.gz", "-ref", "fixed.nii.gz"])
    except ValueError as error:
        assert str(error) == "provide -out and/or -omat"
    else:
        raise AssertionError("FLIRT CLI accepted a command without output")


def test_flirt_cli_rejects_one_path_for_image_and_matrix(tmp_path):
    destination = tmp_path / "same-output"

    with pytest.raises(ValueError, match="different paths"):
        cli.main([
            "flirt", "-in", "moving.nii.gz", "-ref", "fixed.nii.gz",
            "-out", str(destination), "-omat", str(destination),
        ])

    assert not destination.exists()
