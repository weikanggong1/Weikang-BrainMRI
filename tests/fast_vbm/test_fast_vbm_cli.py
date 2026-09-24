"""The public FastVBM CLI is a single-subject entry point."""

from pathlib import Path

import pytest

from freesurfer_torch import cli
from freesurfer_torch import fast_vbm as fast_vbm_module


def _fake_pipeline(captured):
    class Result:
        def save(self, output_dir, overwrite=False):
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            paths = {}
            for name, filename in fast_vbm_module.OUTPUT_FILENAMES.items():
                path = output_dir / filename
                path.write_text(name)
                paths[name] = str(path)
            (output_dir / "fast_vbm_report.json").write_text("{}")
            captured["save"] = (output_dir, overwrite)
            return paths

    class Pipeline:
        def __init__(self, **options):
            captured["options"] = options

        def __call__(self, image, template, brain_mask=None):
            captured["call"] = (image, template, brain_mask)
            return Result()

    return Pipeline


def test_fast_vbm_cli_runs_one_subject_and_prints_all_outputs(
        tmp_path, monkeypatch, capsys):
    captured = {}
    monkeypatch.setattr(fast_vbm_module, "FastVBM", _fake_pipeline(captured))
    output = tmp_path / "subject"

    cli.main([
        "fast-vbm", "-i", "T1w.nii.gz", "--template", "template.nii.gz",
        "-o", str(output), "--brain-mask", "mask.nii.gz",
        "--synthstrip-weights", "weights", "--device", "cuda:2",
        "--threads", "3", "--affine-steps", "7", "--deform-steps", "8",
        "--smoothness", "9", "--no-bias", "--overwrite",
    ])

    assert captured == {
        "options": {
            "device": "cuda:2", "threads": 3,
            "synthstrip_weights": "weights", "bias_correction": False,
            "affine_steps": 7, "deform_steps": 8, "smoothness": 9.0,
        },
        "call": ("T1w.nii.gz", "template.nii.gz", "mask.nii.gz"),
        "save": (output, True),
    }
    expected = [output / name for name in fast_vbm_module.OUTPUT_FILENAMES.values()]
    expected.append(output / "fast_vbm_report.json")
    assert [Path(line) for line in capsys.readouterr().out.splitlines()] == expected
    assert all(path.is_file() for path in expected)


def test_fast_vbm_cli_rejects_existing_output_before_model_load(
        tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setattr(fast_vbm_module, "FastVBM", _fake_pipeline(captured))
    output = tmp_path / "subject"
    output.mkdir()
    existing = output / "T1_GM_JAC_nl.nii.gz"
    existing.write_text("old")

    with pytest.raises(FileExistsError, match="use --overwrite"):
        cli.main([
            "fast-vbm", "-i", "T1w.nii.gz", "--template", "template.nii.gz",
            "-o", str(output),
        ])

    assert captured == {}
    assert existing.read_text() == "old"


def test_multi_subject_cli_is_not_registered(capsys):
    with pytest.raises(SystemExit) as error:
        cli.main(["batch"])

    assert error.value.code == 2
    assert "invalid choice: 'batch'" in capsys.readouterr().err
