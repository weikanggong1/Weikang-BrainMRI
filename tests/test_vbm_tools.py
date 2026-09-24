"""Argument and private-report checks for the experimental VBM wrappers."""

import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest


TOOLS = Path(__file__).resolve().parents[1] / "tools" / "experimental" / "ukb_vbm"


def _load_tool(name, monkeypatch):
    gpu_gm = ModuleType("gpu_gm")
    gpu_gm.SynthSegGM = object
    monkeypatch.setitem(sys.modules, "gpu_gm", gpu_gm)
    path = TOOLS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{name}_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def test_raw_torch_fast_rejects_non_input_grid(tmp_path, monkeypatch, capsys):
    tool, path = _load_tool("run_gpu_raw", monkeypatch)
    monkeypatch.setattr(sys, "argv", [
        str(path), "--manifest", str(tmp_path / "manifest.json"),
        "--subjects-root", str(tmp_path / "subjects"), "--cases", "case1",
        "--gm-method", "torch-fast", "--grid", "native-1mm",
    ])

    with pytest.raises(SystemExit) as error:
        tool.main()

    assert error.value.code == 2
    assert "torch-fast outputs use the input grid" in capsys.readouterr().err


def test_raw_torch_fast_private_timing_report_is_overwrite_guarded(
        tmp_path, monkeypatch, capsys):
    tool, path = _load_tool("run_gpu_raw", monkeypatch)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"cases": [{"case_id": "case1", "path": "unused"}]}))
    subjects = tmp_path / "subjects"
    report = subjects / "case1" / "gpu_fast_timing.private.json"
    report.parent.mkdir(parents=True)
    report.write_text("old")
    monkeypatch.setattr(sys, "argv", [
        str(path), "--manifest", str(manifest), "--subjects-root", str(subjects),
        "--cases", "case1", "--gm-method", "torch-fast",
    ])

    with pytest.raises(SystemExit) as error:
        tool.main()

    assert error.value.code == 2
    assert "gpu_fast_timing.private.json" in capsys.readouterr().err
    assert report.read_text() == "old"


def test_single_case_vbm_private_report_name_is_overwrite_guarded(
        tmp_path, monkeypatch, capsys):
    tool, path = _load_tool("run_gpu_vbm", monkeypatch)
    image = tmp_path / "input.nii.gz"
    template = tmp_path / "template.nii.gz"
    image.touch()
    template.touch()
    output = tmp_path / "vbm"
    output.mkdir()
    report = output / "report.private.json"
    report.write_text("old")
    monkeypatch.setattr(sys, "argv", [
        str(path), "--input", str(image), "--template", str(template),
        "--output-dir", str(output),
    ])

    with pytest.raises(SystemExit) as error:
        tool.main()

    assert error.value.code == 2
    assert "report.private.json" in capsys.readouterr().err
    assert report.read_text() == "old"


def test_batch_accepts_torch_fast_hcp_arm_and_guards_private_report(
        tmp_path, monkeypatch):
    tool, path = _load_tool("run_gpu_batch", monkeypatch)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"cases": [{"case_id": "case1"}]}))
    subjects = tmp_path / "subjects"
    gm = subjects / "case1" / "T1" / "T1_gpu_fast" / "GM_prob.nii.gz"
    gm.parent.mkdir(parents=True)
    gm.touch()
    template = tmp_path / "hcp_template.nii.gz"
    template.touch()
    report = (subjects / "case1" / "T1" / "T1_vbm" / "gpu_fast_hcp" /
              "gpu_register.report.private.json")
    report.parent.mkdir(parents=True)
    report.write_text("old")
    monkeypatch.setattr(sys, "argv", [
        str(path), "--manifest", str(manifest), "--subjects-root", str(subjects),
        "--cases", "case1", "--arms", "gpu_fast_hcp",
        "--hcp-template", str(template),
    ])

    with pytest.raises(FileExistsError, match="gpu_register.report.private.json"):
        tool.main()

    assert report.read_text() == "old"
