"""Aggregate matched WMH-SynthSeg runs without publishing source paths or case IDs."""

import argparse
import json
import math
from pathlib import Path
from statistics import fmean, median


ARMS = ("official-cpu", "official-cuda", "torch-cpu", "torch-cuda")
METRICS = ("segmentation_voxel_agreement", "wmh_hard_dice",
           "lesion_probability_mae", "lesion_probability_nrmse",
           "lesion_probability_max_abs", "segmentation_affine_max_abs",
           "probability_affine_max_abs", "max_csv_volume_abs_error_mm3")


def _read_rows(path):
    rows = json.loads(Path(path).read_text())
    if not isinstance(rows, list) or not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError("input must be a nonempty JSON array of records")
    return rows


def _cases(rows):
    cases = [row.get("case") for row in rows]
    if any(not isinstance(case, str) or not case for case in cases) or len(set(cases)) != len(cases):
        raise ValueError("records need unique nonempty case IDs")
    return cases


def _finite(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return float(value)


def _distribution(values):
    return {"median": median(values), "mean": fmean(values),
            "range": [min(values), max(values)]}


def summarize(execution_paths, comparison_paths, expected_cases=12):
    if expected_cases < 1:
        raise ValueError("expected_cases must be positive")
    execution = {arm: _read_rows(execution_paths[arm]) for arm in ARMS}
    case_order = _cases(execution[ARMS[0]])
    if len(case_order) != expected_cases:
        raise ValueError("run count does not match expected_cases")

    arm_seconds = {}
    execution_seconds = {}
    for arm, rows in execution.items():
        if _cases(rows) != case_order or any(row.get("arm") != arm for row in rows):
            raise ValueError("execution arms have different case order or arm labels")
        if any(row.get("ok") is not True or row.get("returncode") != 0 for row in rows):
            raise ValueError("not all benchmark executions succeeded")
        seconds = [_finite(row.get("seconds"), "seconds") for row in rows]
        if any(value <= 0 for value in seconds):
            raise ValueError("elapsed seconds must be positive")
        arm_seconds[arm] = _distribution(seconds)
        execution_seconds[arm] = seconds

    comparison = {}
    comparison_rows = {}
    for device in ("cpu", "cuda"):
        rows = _read_rows(comparison_paths[device])
        if _cases(rows) != case_order:
            raise ValueError("comparison cases differ from execution cases")
        metrics = {}
        for name in METRICS:
            values = [_finite(row.get(name), name) for row in rows]
            if name in ("segmentation_voxel_agreement", "wmh_hard_dice"):
                if any(not 0 <= value <= 1 for value in values):
                    raise ValueError(f"{name} must be in [0, 1]")
            elif any(value < 0 for value in values):
                raise ValueError(f"{name} cannot be negative")
            metrics[name] = _distribution(values)
        affine_max = [max(row["segmentation_affine_max_abs"],
                          row["probability_affine_max_abs"]) for row in rows]
        metrics["output_affine_max_abs"] = _distribution(affine_max)
        comparison[device] = metrics
        comparison_rows[device] = rows

    case_records = [
        {"case": f"case{i:02d}",
         "seconds": {arm: execution_seconds[arm][i - 1] for arm in ARMS},
         "comparison": {device: {name: float(comparison_rows[device][i - 1][name])
                                 for name in METRICS}
                        for device in ("cpu", "cuda")}}
        for i in range(1, expected_cases + 1)
    ]

    return {"n": expected_cases,
            "case_order": [f"case{i:02d}" for i in range(1, expected_cases + 1)],
            "case_ids_anonymized": True,
            "arm_seconds": arm_seconds,
            "comparison": comparison,
            "case_records": case_records,
            "all_success": True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for arm in ARMS:
        parser.add_argument(f"--{arm}", type=Path, required=True,
                            help=f"{arm} execution.json")
    parser.add_argument("--comparison-cpu", type=Path, required=True)
    parser.add_argument("--comparison-cuda", type=Path, required=True)
    parser.add_argument("--expected-cases", type=int, default=12)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = summarize({arm: getattr(args, arm.replace("-", "_")) for arm in ARMS},
                       {device: getattr(args, f"comparison_{device}")
                        for device in ("cpu", "cuda")}, args.expected_cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Summarized {report['n']} anonymous cases")


if __name__ == "__main__":
    main()
