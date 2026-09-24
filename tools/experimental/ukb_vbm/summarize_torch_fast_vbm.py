#!/usr/bin/env python3
"""Summarize the TorchFAST VBM arm against existing FSL and GPU arms."""

import argparse
import json
import math
import os
import re
from pathlib import Path
import uuid

import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np


FILES = {
    "warped": "T1_GM_to_template_GM.nii.gz",
    "modulated": "T1_GM_to_template_GM_mod.nii.gz",
    "jacobian": "T1_GM_JAC_nl.nii.gz",
}
FSL_STAGES = (
    "robustfov", "crop", "bet", "standard_space_roi", "flirt_xyztrans",
    "xfm_inverse", "xfm_concat", "fnirt_t1", "invwarp", "mask_to_native",
    "brain", "fast", "fsl_reg_ukb", "modulate_ukb",
)


def _load(path, target=None, *, resample=False):
    image = nib.load(path)
    if target is not None:
        different = image.shape != target.shape or not np.allclose(
            image.affine, target.affine, atol=1e-5, rtol=0)
        if different and not resample:
            raise ValueError(f"registered image is not on the template grid: {path}")
        if different:
            image = resample_from_to(image, target, order=1)
    data = np.asarray(image.dataobj, dtype=np.float32)
    if data.ndim != 3 or not np.isfinite(data).all():
        raise ValueError(f"expected a finite 3D image: {path}")
    return image, data


def _metrics(candidate, reference, mask):
    candidate = candidate[mask].astype(np.float64)
    reference = reference[mask].astype(np.float64)
    if candidate.size < 3:
        raise ValueError("comparison mask has fewer than three voxels")
    candidate_binary, reference_binary = candidate >= 0.5, reference >= 0.5
    denominator = int(candidate_binary.sum() + reference_binary.sum())
    reference_volume = float(reference.sum())
    return {
        "pearson": float(np.corrcoef(candidate, reference)[0, 1]),
        "mae": float(np.mean(np.abs(candidate - reference))),
        "dice_0_5": (2 * int((candidate_binary & reference_binary).sum()) / denominator
                       if denominator else None),
        "volume_ratio": (float(candidate.sum()) / reference_volume
                         if reference_volume else None),
    }


def _describe(values):
    values = [float(value) for value in values
              if value is not None and math.isfinite(float(value))]
    return {
        "count": len(values),
        "median": float(np.median(values)) if values else None,
        "min": float(np.min(values)) if values else None,
        "max": float(np.max(values)) if values else None,
    }


def _summarize(rows):
    summary = {
        "n": len(rows),
        "all_jacobians_positive": all(
            row["jacobian"]["nonpositive_voxels"] == 0 for row in rows),
    }
    comparisons = (
        "raw_gm_vs_fsl", "fsl_warped", "fsl_modulated",
        "same_registration_warped", "same_registration_modulated",
    )
    for comparison in comparisons:
        summary[comparison] = {
            metric: _describe(row[comparison][metric] for row in rows)
            for metric in ("pearson", "mae", "dice_0_5", "volume_ratio")
        }
    summary["timing_sec"] = {
        metric: _describe(row["timing_sec"][metric] for row in rows)
        for metric in (
            "synthstrip_fast_outputs", "gpu_registration_modulation",
            "gpu_raw_to_vbm", "fsl_raw_to_vbm",
        )
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects-root", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="anonymous case-level private JSON")
    parser.add_argument("--public-output", type=Path,
                        help="aggregate-only JSON suitable for publication")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    destinations = [args.output] + ([args.public_output] if args.public_output else [])
    if len({path.resolve() for path in destinations}) != len(destinations):
        parser.error("--output and --public-output must be different files")
    if any(path.exists() for path in destinations) and not args.overwrite:
        parser.error("output exists; use --overwrite")

    cases = sorted(
        (path for path in args.subjects_root.iterdir()
         if path.is_dir() and re.fullmatch(r"case\d+", path.name)),
        key=lambda path: int(path.name[4:]),
    )[:args.limit]
    if not cases:
        raise FileNotFoundError("no case<number> directories found")
    template_image, template = _load(args.template)
    template_mask = template > 0.01
    if not template_mask.any():
        raise ValueError("template mask is empty")

    rows = []
    for index, case in enumerate(cases, 1):
        t1 = case / "T1"
        fsl_gm_image, fsl_gm = _load(t1 / "T1_fast" / "T1_brain_pve_1.nii.gz")
        _, torch_gm = _load(t1 / "T1_gpu_fast" / "GM_prob.nii.gz",
                            fsl_gm_image, resample=True)
        native_mask = (fsl_gm > 0.01) | (torch_gm > 0.01)
        row = {
            "case": f"case{index:02d}",
            "raw_gm_vs_fsl": _metrics(torch_gm, fsl_gm, native_mask),
        }
        candidate = t1 / "T1_vbm" / "gpu_fast_ukb"
        comparisons = {
            "fsl": t1 / "T1_vbm" / "ukb",
            "same_registration": t1 / "T1_vbm" / "gpu_ukb",
        }
        candidate_report = json.loads(
            (candidate / "gpu_register.report.private.json").read_text())
        same_report = json.loads(
            (comparisons["same_registration"] /
             "gpu_register.report.private.json").read_text())
        for label, registration_report in (("candidate", candidate_report),
                                           ("same-registration", same_report)):
            fixed = Path(registration_report["fixed"]).resolve()
            if fixed != args.template.resolve():
                raise ValueError(f"{case.name}: {label} arm used another template")
        registration_keys = ("affine_steps_per_scale", "deform_steps_per_scale",
                             "smoothness", "scales")
        candidate_settings = candidate_report.get("settings", {})
        same_settings = same_report.get("settings", {})
        if any(candidate_settings.get(key) != same_settings.get(key)
               for key in registration_keys):
            raise ValueError(f"{case.name}: GPU arms used different registration settings")
        for label, reference in comparisons.items():
            for image_type in ("warped", "modulated"):
                _, candidate_data = _load(candidate / FILES[image_type], template_image)
                _, reference_data = _load(reference / FILES[image_type], template_image)
                row[f"{label}_{image_type}"] = _metrics(
                    candidate_data, reference_data, template_mask)
        _, jacobian = _load(candidate / FILES["jacobian"], template_image)
        row["jacobian"] = {
            "min": float(jacobian[template_mask].min()),
            "max": float(jacobian[template_mask].max()),
            "nonpositive_voxels": int((jacobian[template_mask] <= 0).sum()),
        }
        fast_timing = json.loads((case / "gpu_fast_timing.private.json").read_text())
        registration_timing = json.loads(
            (candidate / "timings.private.json").read_text())
        fsl_timing = json.loads((case / "timings.private.json").read_text())
        upstream = float(fast_timing["gpu_gm_raw_seconds"])
        registration = float(registration_timing["total_sec"])
        stage_values = [fsl_timing.get(stage) for stage in FSL_STAGES]
        complete_fsl_time = (sum(float(value) for value in stage_values)
                             if all(isinstance(value, (int, float))
                                    and math.isfinite(float(value))
                                    for value in stage_values) else None)
        row["timing_sec"] = {
            "synthstrip_fast_outputs": upstream,
            "gpu_registration_modulation": registration,
            "gpu_raw_to_vbm": upstream + registration,
            "fsl_raw_to_vbm": complete_fsl_time,
        }
        rows.append(row)

    methods = {
        "gpu": ("raw T1 -> SynthStrip -> TorchFAST with bias correction -> "
                "PyTorch registration, Jacobian and modulation"),
        "fsl_reference": "existing UKB v1.5-method FSL FAST and FNIRT arm",
        "registered_mask": "official UKB GM template > 0.01",
        "native_mask": ("union of FSL and resampled TorchFAST GM > 0.01 on the "
                        "FSL FAST grid"),
        "timing": ("GPU values are persistent-model warm per-case saved-stage wall "
                   "times and exclude the one-time model load; FSL values sum all "
                   "available raw-to-VBM stages. Methods ran in different windows "
                   "on a shared node."),
    }
    report = {
        "schema": "torch_fast_vbm_validation/v1",
        "methods": methods,
        "cases": rows,
        "summary": _summarize(rows),
    }
    public = {
        "schema": report["schema"],
        "contains_case_level_records": False,
        "contains_subject_paths": False,
        "methods": methods,
        "summary": report["summary"],
    }
    for path in destinations:
        path.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(
        f".{args.output.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}.json")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, args.output)
    if args.public_output:
        temporary = args.public_output.with_name(
            f".{args.public_output.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}.json")
        temporary.write_text(json.dumps(public, indent=2, allow_nan=False) + "\n")
        os.replace(temporary, args.public_output)


if __name__ == "__main__":
    main()
