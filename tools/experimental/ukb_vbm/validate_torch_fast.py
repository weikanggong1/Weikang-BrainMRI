#!/usr/bin/env python3
"""Compare TorchFAST with existing FSL FAST outputs on anonymized cases."""

import argparse
import csv
import json
import math
import re
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from freesurfer_torch.fast import TorchFAST


TISSUES = ("csf", "gm", "wm")
CASE_RE = re.compile(r"case(\d+)$")


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def nonnegative_int(value):
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return value


def discover_cases(root, limit):
    cases = []
    if not root.is_dir():
        raise FileNotFoundError("subjects root does not exist or is not a directory")
    for path in root.iterdir():
        match = CASE_RE.fullmatch(path.name)
        if path.is_dir() and match:
            cases.append((int(match.group(1)), path.name, path))
    cases.sort(key=lambda item: (item[0], item[1]))
    if not cases:
        raise FileNotFoundError("subjects root contains no case<number> directories")
    return [path for _, _, path in cases[:limit]]


def load_nifti(path, reference=None):
    image = nib.load(path)
    data = np.asarray(image.dataobj, dtype=np.float32)
    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"expected a 3D image: {path.name}")
    if reference is not None:
        shape, affine = reference
        if data.shape != shape or not np.allclose(image.affine, affine, atol=1e-5, rtol=0):
            raise ValueError(f"image geometry differs from T1_brain: {path.name}")
    if not np.isfinite(data).all():
        raise ValueError(f"image contains NaN or infinity: {path.name}")
    return data, (data.shape, np.asarray(image.affine, dtype=float))


def volume_array(volume, shape, name):
    data = np.asarray(volume.data)
    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]
    if data.shape != shape:
        raise ValueError(f"TorchFAST {name} shape differs from T1_brain")
    return data


def pearson(first, second):
    first = np.array(first, dtype=np.float64, copy=True)
    second = np.array(second, dtype=np.float64, copy=True)
    first -= first.mean()
    second -= second.mean()
    denominator = math.sqrt(float(np.dot(first, first)) * float(np.dot(second, second)))
    if denominator == 0:
        return None
    return float(np.dot(first, second) / denominator)


def agreement(first, second):
    first64 = np.asarray(first, dtype=np.float64)
    second64 = np.asarray(second, dtype=np.float64)
    difference = first64 - second64
    return {
        "pearson": pearson(first64, second64),
        "mae": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(difference * difference))),
    }


def pve_agreement(predicted, reference, mask):
    values = agreement(predicted[mask], reference[mask])
    predicted_binary = predicted[mask] >= 0.5
    reference_binary = reference[mask] >= 0.5
    denominator = int(predicted_binary.sum() + reference_binary.sum())
    reference_volume = float(reference[mask].sum(dtype=np.float64))
    values.update({
        "dice_at_0_5": (2.0 * float(np.logical_and(
            predicted_binary, reference_binary).sum()) / denominator
            if denominator else None),
        "volume_ratio": (float(predicted[mask].sum(dtype=np.float64))
                         / reference_volume if reference_volume > 0 else None),
    })
    return values


def read_fsl_fast_time(case):
    path = case / "timings.private.json"
    try:
        payload = json.loads(path.read_text())
        value = payload.get("fast")
        if value is None and isinstance(payload.get("timings_sec"), dict):
            value = payload["timings_sec"].get("fast")
        value = float(value)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None
    return value if math.isfinite(value) and value >= 0 else None


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def numeric_constraints(result, image, mask):
    arrays = {
        "pve_csf": volume_array(result.pve_csf, image.shape, "pve_csf"),
        "pve_gm": volume_array(result.pve_gm, image.shape, "pve_gm"),
        "pve_wm": volume_array(result.pve_wm, image.shape, "pve_wm"),
        "hard_segmentation": volume_array(
            result.hard_segmentation, image.shape, "hard_segmentation"),
        "pve_segmentation": volume_array(
            result.pve_segmentation, image.shape, "pve_segmentation"),
        "mixel_type": volume_array(result.mixel_type, image.shape, "mixel_type"),
        "bias_field": volume_array(result.bias_field, image.shape, "bias_field"),
        "restored": volume_array(result.restored, image.shape, "restored"),
    }
    finite = all(np.isfinite(array).all() for array in arrays.values())
    pves = np.stack([arrays[f"pve_{name}"] for name in TISSUES])
    pve_min = float(pves[:, mask].min())
    pve_max = float(pves[:, mask].max())
    pve_sum_error = float(np.max(np.abs(pves[:, mask].sum(axis=0) - 1.0)))
    pve_outside = float(np.max(np.abs(pves[:, ~mask]))) if np.any(~mask) else 0.0
    bias = arrays["bias_field"]
    restored = arrays["restored"]
    bias_outside = float(np.max(np.abs(bias[~mask] - 1.0))) if np.any(~mask) else 0.0
    restored_outside = (float(np.max(np.abs(restored[~mask])))
                        if np.any(~mask) else 0.0)
    identity = np.divide(image, bias, out=np.zeros_like(image), where=mask)
    identity_error = float(np.max(np.abs(restored[mask] - identity[mask])))
    means = np.asarray(result.tissue_means, dtype=float)
    variances = np.asarray(result.tissue_variances, dtype=float)
    labels_valid = (
        arrays["hard_segmentation"].min() >= 0
        and arrays["hard_segmentation"].max() <= 3
        and arrays["pve_segmentation"].min() >= 0
        and arrays["pve_segmentation"].max() <= 3
        and arrays["mixel_type"].min() >= 0
        and arrays["mixel_type"].max() <= 5
    )
    scale = max(1.0, float(np.max(np.abs(restored[mask]))))
    passed = bool(
        finite and pve_min >= -1e-6 and pve_max <= 1.0 + 1e-6
        and pve_sum_error <= 1e-5 and pve_outside <= 1e-6
        and float(bias[mask].min()) > 0 and bias_outside <= 1e-6
        and restored_outside <= 1e-6 and identity_error <= 1e-6 * scale
        and np.isfinite(means).all() and np.all(np.diff(means) > 0)
        and np.isfinite(variances).all() and np.all(variances > 0)
        and labels_valid
    )
    return arrays, {
        "passed": passed,
        "all_outputs_finite": bool(finite),
        "pve_min_inside_mask": pve_min,
        "pve_max_inside_mask": pve_max,
        "pve_sum_max_abs_error_inside_mask": pve_sum_error,
        "pve_max_abs_outside_mask": pve_outside,
        "bias_min_inside_mask": float(bias[mask].min()),
        "bias_max_abs_error_from_one_outside_mask": bias_outside,
        "restored_max_abs_outside_mask": restored_outside,
        "restore_identity_max_abs_error": identity_error,
        "tissue_means_strictly_increasing": bool(np.all(np.diff(means) > 0)),
        "tissue_variances_positive": bool(np.all(variances > 0)),
        "label_ranges_valid": bool(labels_valid),
    }


def evaluate_case(case, label, estimator, device):
    t1 = case / "T1"
    image_path = t1 / "T1_brain.nii.gz"
    fast_dir = t1 / "T1_fast"
    required = [image_path, fast_dir / "T1_brain_bias.nii.gz"] + [
        fast_dir / f"T1_brain_pve_{index}.nii.gz" for index in range(3)
    ]
    if not all(path.is_file() for path in required):
        raise FileNotFoundError(f"{label} is missing T1_brain or required FSL FAST outputs")

    image, geometry = load_nifti(image_path)
    mask = image > 0
    if int(mask.sum()) < 3:
        raise ValueError(f"{label} has fewer than three positive brain voxels")

    synchronize(device)
    started = time.perf_counter()
    result = estimator(image_path)
    synchronize(device)
    torch_seconds = time.perf_counter() - started

    arrays, constraints = numeric_constraints(result, image, mask)
    pve_metrics = {}
    for index, tissue in enumerate(TISSUES):
        reference, _ = load_nifti(fast_dir / f"T1_brain_pve_{index}.nii.gz", geometry)
        pve_metrics[tissue] = pve_agreement(arrays[f"pve_{tissue}"], reference, mask)

    reference_bias, _ = load_nifti(fast_dir / "T1_brain_bias.nii.gz", geometry)
    if np.any(reference_bias[mask] <= 0):
        raise ValueError(f"{label} FSL bias field is not positive inside the brain mask")
    bias_metrics = agreement(
        np.log(arrays["bias_field"][mask].astype(np.float64)),
        np.log(reference_bias[mask].astype(np.float64)),
    )
    reference_restore = np.divide(
        image, reference_bias, out=np.zeros_like(image), where=mask)
    restore_metrics = agreement(arrays["restored"][mask], reference_restore[mask])
    restore_denominator = float(np.mean(np.abs(reference_restore[mask]), dtype=np.float64))
    restore_metrics["relative_mae"] = (
        restore_metrics["mae"] / restore_denominator if restore_denominator > 0 else None)

    fsl_seconds = read_fsl_fast_time(case)
    return {
        "case": label,
        "timing": {
            "torch_fast_sec": float(torch_seconds),
            "fsl_fast_cpu_sec": fsl_seconds,
            "fsl_over_torch_ratio": (fsl_seconds / torch_seconds
                                     if fsl_seconds is not None else None),
        },
        "pve": pve_metrics,
        "bias_log_domain": bias_metrics,
        "restore": restore_metrics,
        "numeric_constraints": constraints,
    }


def finite_values(cases, getter):
    values = [getter(case) for case in cases]
    return [float(value) for value in values
            if value is not None and math.isfinite(float(value))]


def describe(values):
    if not values:
        return {"count": 0, "median": None, "min": None, "max": None}
    return {
        "count": len(values),
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def summarize(cases):
    summary = {
        "timing": {
            name: describe(finite_values(cases, lambda case, key=name:
                           case["timing"][key]))
            for name in ("torch_fast_sec", "fsl_fast_cpu_sec", "fsl_over_torch_ratio")
        },
        "pve_median": {},
        "bias_log_domain_median": {},
        "restore_median": {},
        "numeric_constraints_passed": sum(
            case["numeric_constraints"]["passed"] for case in cases),
        "numeric_constraints_total": len(cases),
    }
    for tissue in TISSUES:
        summary["pve_median"][tissue] = {
            metric: (float(np.median(values)) if values else None)
            for metric in ("pearson", "mae", "rmse", "dice_at_0_5", "volume_ratio")
            for values in [finite_values(
                cases, lambda case, t=tissue, key=metric: case["pve"][t][key])]
        }
    for section in ("bias_log_domain", "restore"):
        target = summary[f"{section}_median"]
        metrics = ("pearson", "mae", "rmse") + (("relative_mae",)
                                                   if section == "restore" else ())
        for metric in metrics:
            values = finite_values(cases, lambda case, s=section, key=metric:
                                   case[s][key])
            target[metric] = float(np.median(values)) if values else None
    return summary


def csv_rows(cases):
    rows = []
    for case in cases:
        row = {"case": case["case"], **case["timing"]}
        for tissue in TISSUES:
            for metric, value in case["pve"][tissue].items():
                row[f"pve_{tissue}_{metric}"] = value
        for metric, value in case["bias_log_domain"].items():
            row[f"bias_log_{metric}"] = value
        for metric, value in case["restore"].items():
            row[f"restore_{metric}"] = value
        row["numeric_constraints_passed"] = case["numeric_constraints"]["passed"]
        rows.append(row)
    return rows


def parse_args():
    parser = argparse.ArgumentParser(
        description=("Compare TorchFAST with existing FSL FAST PVE and bias outputs "
                     "without publishing source paths or subject identifiers."))
    parser.add_argument("--subjects-root", required=True, type=Path,
                        help="root containing case*/T1/T1_brain.nii.gz and T1_fast outputs")
    parser.add_argument("--output", required=True, type=Path,
                        help="private anonymized case-level JSON report; do not publish")
    parser.add_argument("--csv", type=Path,
                        help="optional private anonymized case-level CSV; do not publish")
    parser.add_argument("--limit", type=positive_int, default=10,
                        help="number of numerically sorted cases to evaluate (default: 10)")
    parser.add_argument("--device", default="cuda",
                        help="PyTorch device, for example cuda, cuda:0, or cpu (default: cuda)")
    parser.add_argument("--threads", type=positive_int,
                        help="PyTorch CPU thread count")
    parser.add_argument("--warmup", type=nonnegative_int, default=1,
                        help="untimed TorchFAST calls on the first case (default: 1)")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace existing JSON and CSV reports")
    return parser.parse_args()


def main():
    args = parse_args()
    destinations = [args.output] + ([args.csv] if args.csv else [])
    if len({path.resolve() for path in destinations}) != len(destinations):
        raise ValueError("--output and --csv must be different files")
    existing = [path for path in destinations if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError("report already exists; use --overwrite to replace it")

    cases = discover_cases(args.subjects_root, args.limit)
    estimator = TorchFAST(device=args.device, threads=args.threads)
    device = estimator.device
    first_image = cases[0] / "T1" / "T1_brain.nii.gz"
    for _ in range(args.warmup):
        estimator(first_image)
    synchronize(device)

    results = [evaluate_case(case, f"case{index:02d}", estimator, device)
               for index, case in enumerate(cases, start=1)]
    report = {
        "schema": "torch_fast_validation/v1",
        "case_count": len(results),
        "configuration": {
            "device": str(device),
            "threads": int(torch.get_num_threads()),
            "warmup_calls": args.warmup,
            "case_selection": "first numeric case directories",
        },
        "methods": {
            "torch_timing": ("wall time for TorchFAST(path), including input loading and "
                             "CPU materialization, excluding output serialization"),
            "fsl_timing": ("existing timings.private.json fast wall time, including the FSL "
                           "subprocess and output serialization; null when unavailable"),
            "mask": "finite positive voxels in T1_brain",
            "pve_order": ["CSF", "GM", "WM"],
            "bias_comparison": "Pearson, MAE, and RMSE of log bias fields inside the mask",
            "fsl_restore_reference": "T1_brain divided by the positive FSL bias field",
            "timing_caveat": ("FSL and Torch timings have different output-serialization "
                              "boundaries and are not kernel-only measurements"),
        },
        "cases": results,
        "summary": summarize(results),
    }
    for path in destinations:
        path.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if args.csv:
        rows = csv_rows(results)
        with args.csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
