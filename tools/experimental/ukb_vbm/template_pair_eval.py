#!/usr/bin/env python3
"""Evaluate two template-registration arms with a fixed independent mask.

The mask is the symmetric union of the two input templates. Subject outputs do
not influence it. All templates and outputs must already share one voxel grid;
this script never resamples data.

For each arm, a subject is compared with the mean of all other subjects from
that arm. The exact paired-label permutation test swaps the two arm labels
within subjects and recomputes every leave-one-out (LOO) reference. Pearson
permutations use algebraically equivalent image sufficient statistics. Dice
permutations rebuild the selected-arm sums voxel by voxel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import tempfile

import nibabel as nib
import numpy as np


IMAGE_FILES = {
    "warped": "T1_GM_to_template_GM.nii.gz",
    "modulated": "T1_GM_to_template_GM_mod.nii.gz",
    "jacobian": "T1_GM_JAC_nl.nii.gz",
}
GPU_REPORT = "gpu_register.report.private.json"


def _number(value):
    if value is None or not np.isfinite(value):
        return None
    return float(value)


def _hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _check_3d(image, path):
    if len(image.shape) != 3:
        raise ValueError(f"{path}: expected a 3D image, got shape {image.shape}")
    if not np.isfinite(image.affine).all():
        raise ValueError(f"{path}: affine contains NaN or infinity")


def _check_grid(image, path, shape, affine, affine_atol):
    _check_3d(image, path)
    if image.shape != shape:
        raise ValueError(
            f"{path}: shape {image.shape} does not match reference shape {shape}; "
            "resample explicitly before evaluation"
        )
    if not np.allclose(image.affine, affine, rtol=0, atol=affine_atol):
        difference = float(np.max(np.abs(image.affine - affine)))
        raise ValueError(
            f"{path}: affine does not match reference (maximum absolute "
            f"difference {difference:.6g}, tolerance {affine_atol}); resample "
            "explicitly before evaluation"
        )


def _load_template(path, reference=None, affine_atol=1e-5):
    image = nib.load(str(path))
    _check_3d(image, path)
    if reference is not None:
        _check_grid(image, path, reference.shape, reference.affine, affine_atol)
    data = np.asarray(image.dataobj, dtype=np.float32)
    if not np.isfinite(data).all():
        raise ValueError(f"{path}: template contains NaN or infinity")
    return image, data


def _load_masked(path, shape, affine, mask, affine_atol, allow_nonfinite=False):
    image = nib.load(str(path))
    _check_grid(image, path, shape, affine, affine_atol)
    values = np.asarray(image.dataobj, dtype=np.float32)[mask]
    if not allow_nonfinite and not np.isfinite(values).all():
        count = int((~np.isfinite(values)).sum())
        raise ValueError(f"{path}: {count} non-finite voxels inside the fixed mask")
    return values


def _corr(left, right):
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < 3:
        return None
    left = left[valid].astype(np.float64, copy=False)
    right = right[valid].astype(np.float64, copy=False)
    left = left - left.mean()
    right = right - right.mean()
    denominator = np.sqrt(np.dot(left, left) * np.dot(right, right))
    if denominator <= 0:
        return None
    return _number(np.dot(left, right) / denominator)


def _dice(left, right, threshold):
    valid = np.isfinite(left) & np.isfinite(right)
    left_binary = valid & (left >= threshold)
    right_binary = valid & (right >= threshold)
    denominator = int(left_binary.sum() + right_binary.sum())
    if denominator == 0:
        return None
    return _number(2 * np.logical_and(left_binary, right_binary).sum() / denominator)


def _loo_scores(stack, dice_threshold):
    count = stack.shape[0]
    total = stack.astype(np.float64).sum(axis=0)
    scores = {"pearson": [], "dice_0_5": []}
    for index in range(count):
        reference = (total - stack[index]) / (count - 1)
        scores["pearson"].append(_corr(stack[index], reference))
        scores["dice_0_5"].append(_dice(stack[index], reference, dice_threshold))
    return scores


def _choice_bits(n):
    assignments = np.arange(1 << n, dtype=np.uint32)[:, None]
    return ((assignments >> np.arange(n, dtype=np.uint32)) & 1).astype(bool)


def _pearson_permutation_statistics(arm_a, arm_b, bits):
    """Mean paired LOO differences for all label assignments.

    The Gram matrix contains all inner products required to recompute the LOO
    mean for each relabelled arm. This is algebraically identical to creating
    each LOO reference image, but avoids repeated full-volume reads.
    """
    n, voxels = arm_a.shape
    all_images = np.concatenate((arm_a, arm_b), axis=0).astype(np.float64)
    sums = all_images.sum(axis=1)
    gram = all_images @ all_images.T
    statistics = np.full(len(bits), np.nan, dtype=np.float64)
    valid_n = np.zeros(len(bits), dtype=np.int16)

    def score(indices, subject):
        selected = int(indices[subject])
        others = np.delete(indices, subject)
        sum_x = sums[selected]
        sum_y = sums[others].sum()
        dot_xx = gram[selected, selected]
        dot_xy = gram[selected, others].sum()
        dot_yy = gram[np.ix_(others, others)].sum()
        centered_xx = dot_xx - sum_x * sum_x / voxels
        centered_yy = dot_yy - sum_y * sum_y / voxels
        if centered_xx <= 0 or centered_yy <= 0:
            return np.nan
        return (dot_xy - sum_x * sum_y / voxels) / np.sqrt(centered_xx * centered_yy)

    base = np.arange(n)
    for assignment, swap in enumerate(bits):
        selected_a = base + swap.astype(np.int64) * n
        selected_b = base + (~swap).astype(np.int64) * n
        differences = []
        for subject in range(n):
            left = score(selected_a, subject)
            right = score(selected_b, subject)
            if np.isfinite(left) and np.isfinite(right):
                differences.append(left - right)
        if differences:
            statistics[assignment] = np.mean(differences)
            valid_n[assignment] = len(differences)
    return statistics, valid_n


def _dice_permutation_statistics(arm_a, arm_b, bits, threshold, device, batch_size):
    """Mean paired LOO Dice differences after exact within-subject relabelling."""
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("exact Dice permutation requires PyTorch") from error

    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"permutation device {device} requested but CUDA is unavailable")
    # Match _loo_scores: its leave-one-out sum is float64. This also makes
    # threshold decisions deterministic for values very close to the cutoff.
    dtype = torch.float64
    a = torch.as_tensor(arm_a, dtype=dtype, device=torch_device)
    b = torch.as_tensor(arm_b, dtype=dtype, device=torch_device)
    pair_sum = a + b
    both_arm_total = pair_sum.sum(dim=0)
    n = arm_a.shape[0]
    cutoff = threshold * (n - 1)
    statistics = np.full(len(bits), np.nan, dtype=np.float64)
    valid_n = np.zeros(len(bits), dtype=np.int16)

    with torch.inference_mode():
        for start in range(0, len(bits), batch_size):
            stop = min(start + batch_size, len(bits))
            swap = torch.as_tensor(bits[start:stop], device=torch_device)
            selected_a = torch.where(swap[:, :, None], b[None], a[None])
            sum_a = selected_a.sum(dim=1)
            sum_b = both_arm_total[None] - sum_a
            differences = []
            valid_pairs = []
            for subject in range(n):
                image_a = selected_a[:, subject]
                image_b = pair_sum[subject][None] - image_a
                binary_a = image_a >= threshold
                binary_b = image_b >= threshold
                reference_a = (sum_a - image_a) >= cutoff
                reference_b = (sum_b - image_b) >= cutoff
                denominator_a = binary_a.sum(1) + reference_a.sum(1)
                denominator_b = binary_b.sum(1) + reference_b.sum(1)
                dice_a = (2 * (binary_a & reference_a).sum(1).to(torch.float64)
                          / denominator_a.clamp_min(1))
                dice_b = (2 * (binary_b & reference_b).sum(1).to(torch.float64)
                          / denominator_b.clamp_min(1))
                valid = (denominator_a > 0) & (denominator_b > 0)
                differences.append(torch.where(valid, dice_a - dice_b,
                                                torch.zeros_like(dice_a)))
                valid_pairs.append(valid)
            differences = torch.stack(differences, dim=1)
            valid_pairs = torch.stack(valid_pairs, dim=1)
            counts = valid_pairs.sum(dim=1)
            means = differences.sum(dim=1) / counts.clamp_min(1)
            means = torch.where(counts > 0, means,
                                torch.full_like(means, float("nan")))
            statistics[start:stop] = means.cpu().numpy()
            valid_n[start:stop] = counts.cpu().numpy()
    return statistics, valid_n


def _descriptive(values):
    values = np.asarray([value for value in values if value is not None], dtype=float)
    return {
        "n": int(len(values)),
        "mean": _number(values.mean()) if len(values) else None,
        "median": _number(np.median(values)) if len(values) else None,
    }


def _metric_summary(left, right, permutation_statistics, permutation_n):
    paired = [(a, b) for a, b in zip(left, right) if a is not None and b is not None]
    differences = np.asarray([a - b for a, b in paired], dtype=float)
    observed = _number(differences.mean()) if len(differences) else None
    finite = np.isfinite(permutation_statistics)
    if observed is None or not finite.any():
        p_value = None
    else:
        tolerance = 1e-12
        p_value = _number(np.mean(
            np.abs(permutation_statistics[finite]) >= abs(observed) - tolerance
        ))
    if observed is not None and np.isfinite(permutation_statistics[0]):
        if not np.isclose(observed, permutation_statistics[0], rtol=2e-5, atol=2e-7):
            raise RuntimeError(
                "permutation assignment zero does not reproduce the observed LOO statistic: "
                f"{permutation_statistics[0]} versus {observed}"
            )
    tolerance = 1e-12
    return {
        "arm_a": _descriptive(left),
        "arm_b": _descriptive(right),
        "paired": {
            "n": int(len(differences)),
            "arm_a_minus_arm_b_mean": observed,
            "arm_a_minus_arm_b_median": _number(np.median(differences))
            if len(differences) else None,
            "arm_a_higher": int((differences > tolerance).sum()),
            "arm_b_higher": int((differences < -tolerance).sum()),
            "ties": int((np.abs(differences) <= tolerance).sum()),
        },
        "exact_paired_label_permutation": {
            "statistic": "mean paired arm_a_minus_arm_b LOO score",
            "observed": observed,
            "p_two_sided": p_value,
            "assignments_total": int(len(permutation_statistics)),
            "assignments_with_defined_statistic": int(finite.sum()),
            "paired_case_n_min": int(permutation_n[finite].min()) if finite.any() else 0,
            "paired_case_n_max": int(permutation_n[finite].max()) if finite.any() else 0,
            "loo_reference_recomputed_after_each_label_assignment": True,
        },
    }


def _jacobian_metrics(values):
    finite = np.isfinite(values)
    positive = values[finite & (values > 0)]
    return {
        "fixed_mask_voxels": int(len(values)),
        "finite_fraction": _number(finite.mean()),
        "invalid_or_nonpositive_fraction": _number((~finite | (values <= 0)).mean()),
        "below_0_2_fraction": _number((finite & (values < 0.2)).mean()),
        "above_5_fraction": _number((finite & (values > 5)).mean()),
        "minimum": _number(values[finite].min()) if finite.any() else None,
        "maximum": _number(values[finite].max()) if finite.any() else None,
        "log_sd_positive": _number(np.std(np.log(positive), dtype=np.float64))
        if len(positive) else None,
    }


def _gpu_report(path):
    if not path.exists():
        return None
    report = json.loads(path.read_text())
    fields = (
        "raw_jacobian_min",
        "raw_jacobian_max",
        "raw_nonpositive_jacobian_voxels",
        "deformation_scale",
        "fit_score_before_jacobian_constraint",
        "fit_score_1_minus_loss",
        "jacobian_min",
        "jacobian_max",
        "nonpositive_jacobian_voxels",
    )
    result = {field: _number(report.get(field)) for field in fields}
    before = result["fit_score_before_jacobian_constraint"]
    after = result["fit_score_1_minus_loss"]
    result["fit_score_after_minus_before_constraint"] = (
        _number(after - before) if before is not None and after is not None else None
    )
    result.update({
        "raw_jacobian_scope": "full output grid before whole-field deformation scaling",
        "final_jacobian_is_constraint_output": "deformation_scale" in report,
        "constraint_changed_deformation": (
            result["deformation_scale"] is not None
            and result["deformation_scale"] < 1.0 - 1e-8
        ),
    })
    return result


def _aggregate_jacobian(cases, arm_a, arm_b):
    final_fields = (
        "finite_fraction", "invalid_or_nonpositive_fraction", "below_0_2_fraction",
        "above_5_fraction", "minimum", "maximum", "log_sd_positive",
    )
    report_fields = (
        "raw_jacobian_min", "raw_jacobian_max",
        "raw_nonpositive_jacobian_voxels", "deformation_scale",
        "fit_score_before_jacobian_constraint", "fit_score_1_minus_loss",
        "fit_score_after_minus_before_constraint",
    )

    def field_summary(section, field):
        left, right, paired = [], [], []
        for case in cases:
            left_section = case["jacobian"][arm_a].get(section)
            right_section = case["jacobian"][arm_b].get(section)
            left_value = left_section.get(field) if left_section else None
            right_value = right_section.get(field) if right_section else None
            if left_value is not None:
                left.append(left_value)
            if right_value is not None:
                right.append(right_value)
            if left_value is not None and right_value is not None:
                paired.append((left_value, right_value))
        return {
            "arm_a": _descriptive(left),
            "arm_b": _descriptive(right),
            "paired_n": len(paired),
            "arm_a_minus_arm_b_mean": _number(np.mean([a - b for a, b in paired]))
            if paired else None,
            "arm_a_minus_arm_b_median": _number(np.median([a - b for a, b in paired]))
            if paired else None,
        }

    result = {
        "saved_final_jacobian_on_fixed_mask": {
            field: field_summary("saved_final", field) for field in final_fields
        },
        "gpu_registration_report_full_grid": {},
        "interpretation": (
            "Saved Jacobian metrics use the same template-derived fixed mask for both arms. "
            "When a GPU registration report is present, the saved Jacobian follows whole-field "
            "deformation scaling to satisfy the configured range. Its valid range is therefore a "
            "construction constraint, not independent evidence of registration quality. Raw "
            "Jacobian fields are not saved; raw extrema and nonpositive counts come from the "
            "registration report and cover the full output grid."
        ),
    }
    if any(case["jacobian"][arm_a].get("gpu_report") or
           case["jacobian"][arm_b].get("gpu_report") for case in cases):
        result["gpu_registration_report_full_grid"] = {
            field: field_summary("gpu_report", field) for field in report_fields
        }
    return result


def evaluate_pair(subjects_root, template_a, template_b, arm_a, arm_b,
                  exclude_cases=(), template_threshold=0.01, dice_threshold=0.5,
                  affine_atol=1e-5, permutation_device="cpu", permutation_batch_size=4,
                  require_gpu_reports=False):
    if arm_a == arm_b:
        raise ValueError("--arm-a and --arm-b must differ")
    if (arm_a in (".", "..") or arm_b in (".", "..") or
            not re.fullmatch(r"[A-Za-z0-9_.-]+", arm_a) or not re.fullmatch(
                r"[A-Za-z0-9_.-]+", arm_b)):
        raise ValueError(
            "arm names may contain only letters, digits, '.', '_' and '-', and cannot be '.' or '..'"
        )
    if permutation_batch_size < 1:
        raise ValueError("permutation batch size must be positive")
    if (not np.isfinite(dice_threshold) or not np.isfinite(template_threshold) or
            dice_threshold < 0 or template_threshold < 0):
        raise ValueError("thresholds must be non-negative")
    if not np.isfinite(affine_atol) or affine_atol < 0:
        raise ValueError("affine tolerance must be finite and non-negative")

    template_a_image, template_a_data = _load_template(template_a)
    template_b_image, template_b_data = _load_template(
        template_b, reference=template_a_image, affine_atol=affine_atol
    )
    del template_b_image
    mask = (template_a_data > template_threshold) | (template_b_data > template_threshold)
    if not mask.any():
        raise ValueError("the symmetric template-union mask is empty")

    discovered = sorted(
        path for path in subjects_root.iterdir()
        if path.is_dir() and re.fullmatch(r"case\d+", path.name)
    )
    unknown = set(exclude_cases) - {path.name for path in discovered}
    if unknown:
        raise ValueError(f"excluded case IDs not found: {', '.join(sorted(unknown))}")
    subjects = [path for path in discovered if path.name not in set(exclude_cases)]
    if not 2 <= len(subjects) <= 15:
        raise ValueError(
            f"exact paired-label enumeration requires 2 to 15 included cases; got {len(subjects)}"
        )

    arrays = {
        image_kind: {arm_a: [], arm_b: []}
        for image_kind in ("warped", "modulated")
    }
    cases = []
    for subject in subjects:
        record = {"case_id": subject.name, "loo": {}, "jacobian": {}}
        for arm in (arm_a, arm_b):
            directory = subject / "T1" / "T1_vbm" / arm
            missing = [filename for filename in IMAGE_FILES.values()
                       if not (directory / filename).is_file()]
            if missing:
                raise FileNotFoundError(
                    f"{subject.name}/{arm}: missing {', '.join(missing)}"
                )
            for image_kind in ("warped", "modulated"):
                arrays[image_kind][arm].append(_load_masked(
                    directory / IMAGE_FILES[image_kind],
                    template_a_image.shape, template_a_image.affine, mask,
                    affine_atol, allow_nonfinite=False,
                ))
            jacobian = _load_masked(
                directory / IMAGE_FILES["jacobian"],
                template_a_image.shape, template_a_image.affine, mask,
                affine_atol, allow_nonfinite=True,
            )
            report = _gpu_report(directory / GPU_REPORT)
            if require_gpu_reports and report is None:
                raise FileNotFoundError(f"{subject.name}/{arm}: missing {GPU_REPORT}")
            record["jacobian"][arm] = {
                "saved_final": _jacobian_metrics(jacobian),
                "gpu_report": report,
            }
        cases.append(record)

    for image_kind in arrays:
        for arm in (arm_a, arm_b):
            arrays[image_kind][arm] = np.stack(arrays[image_kind][arm], axis=0)

    bits = _choice_bits(len(cases))
    summary = {}
    for image_kind in ("warped", "modulated"):
        left_stack = arrays[image_kind][arm_a]
        right_stack = arrays[image_kind][arm_b]
        left_scores = _loo_scores(left_stack, dice_threshold)
        right_scores = _loo_scores(right_stack, dice_threshold)
        pearson_permutations, pearson_n = _pearson_permutation_statistics(
            left_stack, right_stack, bits
        )
        dice_permutations, dice_n = _dice_permutation_statistics(
            left_stack, right_stack, bits, dice_threshold,
            permutation_device, permutation_batch_size,
        )
        summary[image_kind] = {
            "loo_pearson": _metric_summary(
                left_scores["pearson"], right_scores["pearson"],
                pearson_permutations, pearson_n,
            ),
            "loo_dice": _metric_summary(
                left_scores["dice_0_5"], right_scores["dice_0_5"],
                dice_permutations, dice_n,
            ),
        }
        for index, record in enumerate(cases):
            record["loo"][image_kind] = {
                "pearson": {
                    "arm_a": left_scores["pearson"][index],
                    "arm_b": right_scores["pearson"][index],
                    "arm_a_minus_arm_b": (
                        _number(left_scores["pearson"][index] -
                                right_scores["pearson"][index])
                        if left_scores["pearson"][index] is not None and
                        right_scores["pearson"][index] is not None else None
                    ),
                },
                "dice": {
                    "arm_a": left_scores["dice_0_5"][index],
                    "arm_b": right_scores["dice_0_5"][index],
                    "arm_a_minus_arm_b": (
                        _number(left_scores["dice_0_5"][index] -
                                right_scores["dice_0_5"][index])
                        if left_scores["dice_0_5"][index] is not None and
                        right_scores["dice_0_5"][index] is not None else None
                    ),
                },
            }

    return {
        "schema_version": 1,
        "private_case_level_output": True,
        "method": {
            "mask": (
                f"symmetric union: (template_a > {template_threshold}) OR "
                f"(template_b > {template_threshold}); subjects do not influence the mask"
            ),
            "grid_policy": "strict shape and affine equality; no implicit resampling",
            "loo": "each subject is compared with the mean of all other subjects in its arm",
            "permutation": (
                "all 2^n within-subject arm-label assignments; each assignment rebuilds both "
                "arm-specific LOO references; two-sided p is the exact tail proportion"
            ),
            "bootstrap_confidence_interval": "not computed",
            "dice_threshold": dice_threshold,
        },
        "arms": {"arm_a": arm_a, "arm_b": arm_b},
        "templates": {
            "arm_a": {"path": str(template_a.resolve()), "sha256": _hash(template_a)},
            "arm_b": {"path": str(template_b.resolve()), "sha256": _hash(template_b)},
        },
        "grid": {
            "shape": list(template_a_image.shape),
            "affine": template_a_image.affine.tolist(),
            "affine_absolute_tolerance": affine_atol,
        },
        "fixed_mask_voxels": int(mask.sum()),
        "included_cases": len(cases),
        "excluded_cases": sorted(set(exclude_cases)),
        "summary": summary,
        "jacobian_summary": _aggregate_jacobian(cases, arm_a, arm_b),
        "cases": cases,
    }


def _save_nifti(path, data, affine):
    nib.save(nib.Nifti1Image(np.asarray(data, dtype=np.float32), affine), str(path))


def self_test():
    shape = (6, 7, 8)
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    grid = np.stack(np.meshgrid(*[np.arange(size) for size in shape], indexing="ij"))
    template_a = np.exp(-np.square(grid - np.array([2.5, 3, 3.5])[:, None, None, None]).sum(0) / 7)
    template_b = np.exp(-np.square(grid - np.array([2.7, 3, 3.5])[:, None, None, None]).sum(0) / 7)
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        path_a, path_b = root / "template_a.nii.gz", root / "template_b.nii.gz"
        _save_nifti(path_a, template_a, affine)
        _save_nifti(path_b, template_b, affine)
        synthetic_warped = {"left": [], "right": []}
        for index in range(3):
            for arm, shift in (("left", 0.03 * index), ("right", 0.12 * index)):
                directory = root / "subjects" / f"case{index + 1:02d}" / "T1" / "T1_vbm" / arm
                directory.mkdir(parents=True)
                warped = np.clip(template_a + shift * (grid[0] - 2.5), 0, 1)
                synthetic_warped[arm].append(warped.astype(np.float32))
                jacobian = np.full(shape, 1 + (0.01 if arm == "left" else 0.02) * index)
                _save_nifti(directory / IMAGE_FILES["warped"], warped, affine)
                _save_nifti(directory / IMAGE_FILES["modulated"], warped * jacobian, affine)
                _save_nifti(directory / IMAGE_FILES["jacobian"], jacobian, affine)
                report = {
                    "raw_jacobian_min": -0.1 if index == 0 else 0.1,
                    "raw_jacobian_max": 5.4,
                    "raw_nonpositive_jacobian_voxels": 2,
                    "deformation_scale": 0.75,
                    "fit_score_before_jacobian_constraint": 0.8,
                    "fit_score_1_minus_loss": 0.78,
                    "jacobian_min": float(jacobian.min()),
                    "jacobian_max": float(jacobian.max()),
                    "nonpositive_jacobian_voxels": 0,
                }
                (directory / GPU_REPORT).write_text(json.dumps(report))
        result = evaluate_pair(
            root / "subjects", path_a, path_b, "left", "right",
            permutation_device="cpu", permutation_batch_size=2,
            require_gpu_reports=True,
        )
        assert result["included_cases"] == 3
        assert result["fixed_mask_voxels"] == int(((template_a > 0.01) |
                                                    (template_b > 0.01)).sum())
        for image_kind in ("warped", "modulated"):
            for metric in ("loo_pearson", "loo_dice"):
                test = result["summary"][image_kind][metric][
                    "exact_paired_label_permutation"]
                assert test["assignments_total"] == 8
                assert test["loo_reference_recomputed_after_each_label_assignment"]
                assert test["p_two_sided"] is None or 0 <= test["p_two_sided"] <= 1
        assert result["cases"][0]["jacobian"]["left"]["gpu_report"][
            "final_jacobian_is_constraint_output"]

        # Compare the optimized permutation engines with literal LOO rebuilding
        # for every assignment in this small example.
        fixed_mask = (template_a > 0.01) | (template_b > 0.01)
        left = np.stack([volume[fixed_mask] for volume in synthetic_warped["left"]])
        right = np.stack([volume[fixed_mask] for volume in synthetic_warped["right"]])
        bits = _choice_bits(3)
        pearson_statistics, _ = _pearson_permutation_statistics(left, right, bits)
        dice_statistics, _ = _dice_permutation_statistics(
            left, right, bits, 0.5, "cpu", 2
        )
        for assignment, swap in enumerate(bits):
            selected_left = np.where(swap[:, None], right, left)
            selected_right = np.where(swap[:, None], left, right)
            literal_left = _loo_scores(selected_left, 0.5)
            literal_right = _loo_scores(selected_right, 0.5)
            for metric, exact in (("pearson", pearson_statistics),
                                  ("dice_0_5", dice_statistics)):
                literal = np.mean([
                    a - b for a, b in zip(literal_left[metric], literal_right[metric])
                    if a is not None and b is not None
                ])
                assert np.isclose(exact[assignment], literal, rtol=1e-7, atol=1e-9)

        bad = root / "bad_grid.nii.gz"
        bad_affine = affine.copy()
        bad_affine[0, 3] = 1
        _save_nifti(bad, template_a, bad_affine)
        try:
            _load_masked(bad, shape, affine, template_a > 0.01, 1e-5)
        except ValueError as error:
            assert "affine does not match" in str(error)
        else:
            raise AssertionError("grid mismatch was not rejected")
    print(json.dumps({"self_test": "passed", "cases": 3, "assignments": 8}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects-root", type=Path)
    parser.add_argument("--template-a", type=Path)
    parser.add_argument("--template-b", type=Path)
    parser.add_argument("--arm-a")
    parser.add_argument("--arm-b")
    parser.add_argument("--out", type=Path,
                        help="private JSON output; contains case IDs and template paths")
    parser.add_argument("--exclude-cases", nargs="*", default=())
    parser.add_argument("--template-threshold", type=float, default=0.01)
    parser.add_argument("--dice-threshold", type=float, default=0.5)
    parser.add_argument("--affine-atol", type=float, default=1e-5)
    parser.add_argument("--permutation-device", default="cpu",
                        help="PyTorch device used for exact Dice permutations, e.g. cuda:0")
    parser.add_argument("--permutation-batch-size", type=int, default=4)
    parser.add_argument("--require-gpu-reports", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    required = {
        "--subjects-root": args.subjects_root,
        "--template-a": args.template_a,
        "--template-b": args.template_b,
        "--arm-a": args.arm_a,
        "--arm-b": args.arm_b,
        "--out": args.out,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        parser.error(f"required arguments missing: {', '.join(missing)}")
    for name, path in (("subjects root", args.subjects_root),
                       ("template A", args.template_a),
                       ("template B", args.template_b)):
        if not path.exists():
            parser.error(f"{name} does not exist: {path}")
    result = evaluate_pair(
        args.subjects_root, args.template_a, args.template_b,
        args.arm_a, args.arm_b, exclude_cases=args.exclude_cases,
        template_threshold=args.template_threshold,
        dice_threshold=args.dice_threshold, affine_atol=args.affine_atol,
        permutation_device=args.permutation_device,
        permutation_batch_size=args.permutation_batch_size,
        require_gpu_reports=args.require_gpu_reports,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    compact = {
        "included_cases": result["included_cases"],
        "fixed_mask_voxels": result["fixed_mask_voxels"],
        "arms": result["arms"],
        "summary": result["summary"],
    }
    print(json.dumps(compact, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
