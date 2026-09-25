import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest


SCRIPT = Path(__file__).parents[2] / "tools" / "validate_fnirt_fsl.py"
SPEC = importlib.util.spec_from_file_location("validate_fnirt_fsl", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _save(path, data, affine):
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(
        nib.Nifti1Image(np.asarray(data, dtype=np.float32), affine),
        str(path),
    )


def _synthetic_tree(tmp_path, monkeypatch, case_count=2):
    study = tmp_path / "study"
    work = tmp_path / "work"
    fsl_dir = tmp_path / "fsl"
    shape = (4, 5, 6)
    affine = np.array(
        [[-2, 0, 0, 8], [0, 2, 0, -10], [0, 0, 2, -6], [0, 0, 0, 1]],
        dtype=np.float64,
    )
    grid = np.indices(shape, dtype=np.float32)
    template_data = 0.1 + (grid[0] + 2 * grid[1] + 3 * grid[2]) / 50
    template = study / "assets" / "template_GM_v1.nii.gz"
    mask = fsl_dir / "data" / "standard" / MODULE.DEFAULT_REFERENCE_MASK
    _save(template, template_data, affine)
    _save(mask, np.ones(shape, dtype=np.uint8), affine)

    (fsl_dir / "etc").mkdir(parents=True, exist_ok=True)
    (fsl_dir / "etc" / "fslversion").write_text("6.0.7.4\n")
    config = fsl_dir / "etc" / "flirtsch" / MODULE.SUPPORTED_CONFIG
    config.parent.mkdir(parents=True, exist_ok=True)
    config_bytes = b"synthetic matched FNIRT config\n"
    config.write_bytes(config_bytes)
    monkeypatch.setattr(
        MODULE,
        "SUPPORTED_CONFIG_SHA256",
        hashlib.sha256(config_bytes).hexdigest(),
    )
    for executable in ("fnirt", "fnirtfileutils"):
        path = fsl_dir / "bin" / executable
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#!/bin/sh\nexit 0\n")
        path.chmod(0o755)

    for index in range(1, case_count + 1):
        case = MODULE.case_inputs(study, f"case{index:02d}")
        case_data = template_data + 0.01 * index * grid[0]
        _save(case.gm, case_data, affine)
        case.affine.parent.mkdir(parents=True, exist_ok=True)
        matrix = np.eye(4)
        matrix[0, 3] = 0.1 * index
        np.savetxt(case.affine, matrix)

    args = SimpleNamespace(
        study_root=study,
        work_dir=work,
        template=None,
        reference_mask=mask,
        fsl_dir=fsl_dir,
        config=None,
        case_count=case_count,
        device="cpu",
        threads=1,
        overwrite=False,
        output_json=work / "private" / "dry_run.private.json",
        private_json=work / "private" / "summary.private.json",
        public_json=work / "summary.public.json",
    )
    return args, affine, template_data


def _install_fake_runners(monkeypatch, template_data, affine):
    calls = {"fsl": 0, "torch": 0, "expand": 0}

    def write_primary(outputs, offset):
        coefficients = np.zeros((3, 3, 3, 3), dtype=np.float32)
        _save(outputs["cout"], coefficients, np.eye(4))
        iout = template_data + offset + 0.03 * np.indices(template_data.shape)[0]
        jout = 0.8 + template_data / 10 + offset / 10
        _save(outputs["iout"], iout, affine)
        _save(outputs["jout"], jout, affine)

    def fake_fsl(case, inputs, outputs, environment, log_path):
        calls["fsl"] += 1
        write_primary(outputs, 0.0)
        return 2.0

    def fake_torch(args, case, inputs, outputs):
        calls["torch"] += 1
        write_primary(outputs, 0.01)
        return 1.0

    def fake_expand(inputs, coefficients, output, environment, log_path):
        calls["expand"] += 1
        grid = np.indices(template_data.shape, dtype=np.float32)
        residual = np.stack(
            (0.1 * grid[0], -0.05 * grid[1], 0.02 * grid[2]), axis=-1
        )
        if coefficients.parent.name == "torch":
            residual = residual + 0.005
        _save(output, residual, affine)
        return 0.2

    monkeypatch.setattr(MODULE, "run_fsl_fnirt", fake_fsl)
    monkeypatch.setattr(MODULE, "run_torch_fnirt", fake_torch)
    monkeypatch.setattr(MODULE, "expand_coefficients", fake_expand)
    return calls


def test_metric_contract_handles_defined_and_undefined_pearson():
    reference = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    candidate = reference + 0.5
    metrics = MODULE.comparison_metrics(reference, candidate)
    assert np.isclose(metrics["pearson"], 1.0)
    assert metrics["pearson_defined"]
    assert np.isclose(metrics["mae"], 0.5)
    assert np.isclose(metrics["rmse"], 0.5)
    assert np.isclose(metrics["maxabs"], 0.5)

    constant = MODULE.comparison_metrics(np.ones(5), np.ones(5))
    assert constant["pearson"] is None
    assert not constant["pearson_defined"]


def test_run_cache_summary_and_public_privacy(tmp_path, monkeypatch, capsys):
    args, affine, template_data = _synthetic_tree(tmp_path, monkeypatch)
    inputs = MODULE.validate_inputs(args, check_device=True)
    calls = _install_fake_runners(monkeypatch, template_data, affine)

    for case in inputs["cases"]:
        record, cached = MODULE.run_case(args, inputs, case)
        assert record["status"] == "success"
        assert not cached
    assert calls == {"fsl": 2, "torch": 2, "expand": 4}

    cached_record, cached = MODULE.run_case(args, inputs, inputs["cases"][0])
    assert cached
    assert cached_record["status"] == "success"
    assert calls == {"fsl": 2, "torch": 2, "expand": 4}

    assert MODULE.summarize(args) == 0
    public = json.loads(args.public_json.read_text())
    private = json.loads(args.private_json.read_text())
    public_text = args.public_json.read_text()
    assert public["cohort"]["case_count"] == 2
    assert public["timing"]["distributions_seconds_or_ratio"][
        "fsl_cpu_fnirt_wall"
    ]["median"] == 2.0
    assert public["timing"]["distributions_seconds_or_ratio"][
        "torch_fnirt_synchronized_wall"
    ]["median"] == 1.0
    assert public["metrics"]["results"]["iout"]["mae"]["count"] == 2
    assert len(private["cases"]) == 2
    assert "case01" in args.private_json.read_text()
    assert "case01" not in public_text
    assert str(args.study_root.resolve()) not in public_text
    assert str(args.work_dir.resolve()) not in public_text
    assert "run_signature" not in public_text
    assert "/cwStorage/" not in public_text

    first_case = inputs["cases"][0]
    image = nib.load(first_case.gm)
    changed = np.asarray(image.dataobj, dtype=np.float32) + 0.02
    _save(first_case.gm, changed, image.affine)
    assert MODULE.dry_run(args) == 0
    dry = json.loads(args.output_json.read_text())
    assert dry["cases"][0]["cache_status"] == "stale_or_corrupt"
    assert dry["cases"][1]["cache_status"] == "valid"
    capsys.readouterr()


def test_fsl_commands_use_matched_inputs_and_expand_without_affine(
    tmp_path, monkeypatch
):
    args, _, _ = _synthetic_tree(tmp_path, monkeypatch, case_count=1)
    inputs = MODULE.validate_inputs(args, check_device=True)
    case = inputs["cases"][0]
    paths = MODULE.output_paths(args.work_dir, case.case_id)
    commands = []

    def capture(command, *, environment, log_path):
        commands.append(command)
        return 0.5

    monkeypatch.setattr(MODULE, "run_command", capture)
    environment = MODULE._fsl_environment(inputs["tools"], 1)
    MODULE.run_fsl_fnirt(
        case, inputs, paths["fsl"], environment, paths["log"]
    )
    MODULE.expand_coefficients(
        inputs,
        paths["fsl"]["cout"],
        paths["fsl"]["nonlinear_residual"],
        environment,
        paths["log"],
    )

    fnirt = commands[0]
    assert f"--in={case.gm}" in fnirt
    assert f"--ref={inputs['template']}" in fnirt
    assert f"--aff={case.affine}" in fnirt
    assert f"--refmask={inputs['reference_mask']}" in fnirt
    assert f"--config={inputs['tools'].config}" in fnirt
    expansion = commands[1]
    assert "--outformat=field" in expansion
    assert "--withaff" not in expansion


def test_running_record_cannot_be_resumed_or_overwritten_implicitly(
    tmp_path, monkeypatch
):
    args, _, _ = _synthetic_tree(tmp_path, monkeypatch, case_count=1)
    inputs = MODULE.validate_inputs(args, check_device=True)
    case = inputs["cases"][0]
    paths = MODULE.output_paths(args.work_dir, case.case_id)
    paths["record"].parent.mkdir(parents=True, exist_ok=True)
    paths["record"].write_text(json.dumps({"status": "running"}))
    sentinel = paths["fsl"]["cout"]
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    sentinel.write_bytes(b"active output")

    with pytest.raises(RuntimeError, match="already has a running record"):
        MODULE.run_case(args, inputs, case)
    assert sentinel.read_bytes() == b"active output"


def test_validation_rejects_nonbinary_reference_mask(tmp_path, monkeypatch):
    args, affine, template_data = _synthetic_tree(
        tmp_path, monkeypatch, case_count=1
    )
    mask = np.ones(template_data.shape, dtype=np.float32)
    mask[0, 0, 0] = 0.5
    _save(args.reference_mask, mask, affine)

    with pytest.raises(ValueError, match="binary 0/1"):
        MODULE.validate_inputs(args, check_device=True)


def test_aggregate_metrics_reports_all_undefined_pearson():
    record = {
        output: {
            "pearson": None,
            "mae": 0.0,
            "rmse": 0.0,
            "maxabs": 0.0,
        }
        for output in MODULE.OUTPUT_KEYS
    }
    aggregate = MODULE.aggregate_metrics([record])
    for output in MODULE.OUTPUT_KEYS:
        pearson = aggregate[output]["pearson"]
        assert pearson == {
            "count": 0,
            "minimum": None,
            "q25": None,
            "median": None,
            "mean": None,
            "q75": None,
            "maximum": None,
            "undefined_count": 1,
        }
