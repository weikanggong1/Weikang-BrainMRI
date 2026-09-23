"""The published WMH summary must be complete and contain no source identifiers."""

import json
from pathlib import Path
import subprocess
import sys


SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "summarize_wmh_validation.py"
CASES = ("patient_A_private", "patient_B_private")
ARMS = ("official-cpu", "official-cuda", "torch-cpu", "torch-cuda")


def _inputs(directory):
    files = {}
    for arm, seconds in zip(ARMS, ((10, 20), (5, 7), (8, 12), (2, 4))):
        path = directory / f"{arm}.json"
        path.write_text(json.dumps([
            {"case": case, "arm": arm, "seconds": elapsed, "ok": True,
             "returncode": 0, "command_private": "/private/host/patient"}
            for case, elapsed in zip(CASES, seconds)
        ]))
        files[arm] = path
    for device in ("cpu", "cuda"):
        path = directory / f"comparison-{device}.json"
        path.write_text(json.dumps([
            {"case": case, "segmentation_voxel_agreement": 1.0 - i * 0.1,
             "wmh_hard_dice": 1.0 - i * 0.2,
             "lesion_probability_nrmse": i * 0.2,
             "lesion_probability_mae": i * 0.01,
             "lesion_probability_max_abs": i * 0.03,
             "segmentation_affine_max_abs": i * 0.1,
             "probability_affine_max_abs": i * 0.2,
             "max_csv_volume_abs_error_mm3": i * 0.04,
             "reference_path_private": "/private/host/patient"}
            for i, case in enumerate(CASES)
        ]))
        files[f"comparison-{device}"] = path
    return files


def _command(files, output):
    command = [sys.executable, str(SCRIPT)]
    for arm in ARMS:
        command += [f"--{arm}", str(files[arm])]
    for device in ("cpu", "cuda"):
        command += [f"--comparison-{device}", str(files[f"comparison-{device}"])]
    return command + ["--expected-cases", "2", "--output", str(output)]


def test_summary_aggregates_matched_arms_without_private_fields(tmp_path):
    files = _inputs(tmp_path)
    output = tmp_path / "report.public.json"
    result = subprocess.run(_command(files, output), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text())
    assert report["n"] == 2
    assert report["case_order"] == ["case01", "case02"]
    assert report["all_success"] is True
    assert report["arm_seconds"]["official-cpu"] == {
        "median": 15.0, "mean": 15.0, "range": [10.0, 20.0]
    }
    assert report["comparison"]["cuda"]["wmh_hard_dice"]["range"] == [0.8, 1.0]
    assert report["comparison"]["cpu"]["output_affine_max_abs"]["range"] == [0.0, 0.2]
    assert report["case_records"][0]["seconds"]["official-cpu"] == 10.0
    assert report["case_records"][1]["comparison"]["cuda"]["wmh_hard_dice"] == 0.8
    text = output.read_text()
    assert "private" not in text and "/host/" not in text
    assert not any(case in text for case in CASES)


def test_summary_rejects_failed_arm_before_publishing(tmp_path):
    files = _inputs(tmp_path)
    rows = json.loads(files["torch-cuda"].read_text())
    rows[1]["ok"] = False
    files["torch-cuda"].write_text(json.dumps(rows))
    output = tmp_path / "report.public.json"
    result = subprocess.run(_command(files, output), capture_output=True, text=True)
    assert result.returncode != 0
    assert not output.exists()
