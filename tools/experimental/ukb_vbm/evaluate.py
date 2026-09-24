"""Evaluate paired UKB/local-HCP-derived VBM registrations without source paths.

Example:
    python evaluate.py --subjects-root subjects --out-prefix reports/vbm.private

This writes vbm.private.json and vbm.private.csv with anonymous case-level
records. Treat both files as private intermediates and publish only aggregate
output from build_public_report.py. Both FSL arms must use the same FAST grey
matter image. Optional gpu_ukb, gpu_raw_ukb and gpu_raw_hcp directories are
included when present under each subject's T1/T1_vbm directory. Cross-template
image comparisons resample local HCP-derived outputs onto the UKB grid.
Template fit scores are not used to rank methods.
"""

import argparse
import csv
import json
from pathlib import Path
import re

import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np


FILES = {
    "warped": "T1_GM_to_template_GM.nii.gz",
    "modulated": "T1_GM_to_template_GM_mod.nii.gz",
    "jacobian": "T1_GM_JAC_nl.nii.gz",
}
BASE_ARMS = ("ukb", "hcp")
GPU_ARMS = ("gpu_ukb", "gpu_raw_ukb", "gpu_raw_hcp")
PREPROCESS_STAGES = ("robustfov", "crop", "bet", "standard_space_roi",
                     "flirt_xyztrans", "xfm_inverse", "xfm_concat", "fnirt_t1",
                     "invwarp", "mask_to_native", "brain", "fast")


def number(value):
    if value is None or not np.isfinite(value):
        return None
    return float(value)


def mean(values):
    values = [value for value in values if value is not None]
    return number(np.mean(values)) if values else None


def difference(left, right):
    return number(left - right) if left is not None and right is not None else None


def corr(a, b, mask):
    a, b = a[mask], b[mask]
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 3 or np.std(a[valid]) == 0 or np.std(b[valid]) == 0:
        return None
    return number(np.corrcoef(a[valid], b[valid])[0, 1])


def dice(a, b, mask, threshold=0.5):
    a = (a >= threshold) & mask
    b = (b >= threshold) & mask
    denominator = int(a.sum() + b.sum())
    return number(2 * (a & b).sum() / denominator) if denominator else None


def distribution(data, mask):
    values = data[mask]
    finite = values[np.isfinite(values)]
    if not len(values) or not len(finite):
        return {"finite_fraction": 0.0, "p01": None, "p50": None,
                "p99": None, "mean": None}
    return {
        "finite_fraction": number(len(finite) / len(values)),
        "p01": number(np.percentile(finite, 1)),
        "p50": number(np.percentile(finite, 50)),
        "p99": number(np.percentile(finite, 99)),
        "mean": number(np.mean(finite, dtype=np.float64)),
    }


def image(path, target=None):
    img = nib.load(path)
    resampled = target is not None and (
        img.shape != target.shape or not np.allclose(img.affine, target.affine, atol=1e-4)
    )
    if resampled:
        img = resample_from_to(img, target, order=1)
    return img, np.asarray(img.dataobj, dtype=np.float32), resampled


def compare(reference, candidate, mask):
    a, b = reference[mask], candidate[mask]
    valid = np.isfinite(a) & np.isfinite(b)
    error = np.abs(a[valid] - b[valid])
    return {
        "mae": number(np.mean(error)) if len(error) else None,
        "rmse": number(np.sqrt(np.mean(error * error))) if len(error) else None,
        "pearson": corr(reference, candidate, mask),
        "dice_0_5": dice(reference, candidate, mask),
    }


def timing(subject, arm):
    paths = [subject / "timings.private.json",
             subject / "T1" / "T1_vbm" / arm / "timings.private.json"]
    data = {}
    arm_timing = {}
    for path in paths:
        if path.exists():
            loaded = json.loads(path.read_text())
            data.update(loaded)
            if path == paths[1]:
                arm_timing = loaded
    registration = arm_timing.get("registration_sec", data.get(f"fsl_reg_{arm}"))
    modulation = arm_timing.get("modulation_sec", data.get(f"modulate_{arm}"))
    total = arm_timing.get("total_sec", arm_timing.get("elapsed_sec", data.get(arm)))
    if isinstance(total, dict):
        registration = total.get("registration_sec", registration)
        modulation = total.get("modulation_sec", modulation)
        total = total.get("total_sec", total.get("elapsed_sec"))
    if total is None and registration is not None and modulation is not None:
        total = registration + modulation
    return {"registration_sec": number(registration),
            "modulation_sec": number(modulation), "total_sec": number(total)}


def summarize_arm(warped, modulated, jacobian, mask, voxel_mm3):
    # Use the same template-derived mask for every arm. Arm-specific warped-GM
    # support would make deformation QC incomparable across methods.
    support = mask
    j = jacobian[mask]
    finite_j = j[np.isfinite(j)]
    positive = finite_j[finite_j > 0]
    result = {
        "warped": distribution(warped, mask),
        "modulated": distribution(modulated, mask),
        "jacobian": distribution(jacobian, support),
        "warped_gm_mm3": number(np.nansum(warped[mask], dtype=np.float64) * voxel_mm3),
        "modulated_gm_mm3": number(np.nansum(modulated[mask], dtype=np.float64) * voxel_mm3),
        "warped_outside_0_1_fraction": number(np.mean((warped[mask] < -1e-4) |
                                                       (warped[mask] > 1.0001))),
        "jacobian_invalid_or_nonpositive_fraction": number(np.mean(
            ~np.isfinite(j) | (j <= 0))) if len(j) else None,
        "jacobian_below_0_2_fraction": number(np.mean(j < 0.2)) if len(j) else None,
        "jacobian_above_5_fraction": number(np.mean(j > 5)) if len(j) else None,
        "log_jacobian_sd": number(np.std(np.log(positive))) if len(positive) else None,
        "support_voxels": int(support.sum()),
    }
    return result


def flatten(prefix, item, row):
    for key, value in item.items():
        name = f"{prefix}_{key}" if prefix else key
        if isinstance(value, dict):
            flatten(name, value, row)
        else:
            row[name] = value


def paired_summary(cases, path):
    values = [case["comparisons"][path] for case in cases
              if path in case["comparisons"]]
    metrics = sorted({key for value in values for key in value})
    result = {"n": len(values)}
    for key in metrics:
        numeric = [value[key] for value in values if isinstance(value[key], (int, float))]
        result[key + "_n"] = len(numeric)
        if numeric:
            result[key + "_mean"] = number(np.mean(numeric))
            result[key + "_median"] = number(np.median(numeric))
    return result


def matched_loo(entries, mask):
    """LOO scores for two arms restricted to the same subjects."""
    result = {}
    for arm in ("gpu_raw_ukb", "gpu_raw_hcp"):
        stack = np.stack([arm_data[arm][0] for _, arm_data in entries]).astype(np.float64)
        finite = np.isfinite(stack)
        values = np.where(finite, stack, 0)
        total = values.sum(axis=0)
        count = finite.sum(axis=0)
        for index, (case, _) in enumerate(entries):
            other_count = count - finite[index]
            other_mean = np.divide(total - values[index], other_count,
                                   out=np.zeros_like(total), where=other_count > 0)
            roi = mask & (other_count > 0)
            result[(case["case_id"], arm)] = {
                "loo_pearson": corr(stack[index], other_mean, roi),
                "loo_dice_0_5": dice(stack[index], other_mean, roi),
            }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects-root", type=Path, required=True)
    parser.add_argument("--out-prefix", type=Path, required=True)
    parser.add_argument("--ukb-template", type=Path, required=True,
                        help="UKB GM template used to define the independent evaluation mask")
    parser.add_argument("--hcp-template", type=Path, required=True,
                        help="comparison GM template; must share the UKB template grid")
    parser.add_argument("--exclude-cases", nargs="*", default=(),
                        help="e.g. case01, used for tuning; excluded from all LOO and summary statistics")
    args = parser.parse_args()

    discovered = sorted(path for path in args.subjects_root.iterdir()
                        if path.is_dir() and re.fullmatch(r"case\d+", path.name))
    unknown = set(args.exclude_cases) - {path.name for path in discovered}
    if unknown:
        parser.error(f"excluded case IDs not found: {', '.join(sorted(unknown))}")
    subjects = [path for path in discovered if path.name not in args.exclude_cases]
    if not subjects:
        raise ValueError("No caseXX directories found")

    ukb_template_img, ukb_template, _ = image(args.ukb_template)
    hcp_template_img = nib.load(args.hcp_template)
    if (hcp_template_img.shape != ukb_template_img.shape or
            not np.allclose(hcp_template_img.affine, ukb_template_img.affine,
                            rtol=0, atol=1e-5)):
        raise ValueError("UKB and comparison templates must share shape and affine")
    hcp_template = np.asarray(hcp_template_img.dataobj, dtype=np.float32)
    if not np.isfinite(ukb_template).all() or not np.isfinite(hcp_template).all():
        raise ValueError("templates contain NaN or infinity")
    mask = (ukb_template > 0.01) | (hcp_template > 0.01)
    if not np.any(mask):
        raise ValueError("The symmetric template-union evaluation mask is empty")

    cases = []
    warped_by_arm = {arm: [] for arm in BASE_ARMS + GPU_ARMS}
    reference_grid = ukb_template_img
    for subject in subjects:
        native_path = subject / "T1" / "T1_fast" / "T1_brain_pve_1.nii.gz"
        native_img, native, _ = image(native_path)
        native_mm3 = abs(np.linalg.det(native_img.affine[:3, :3]))
        result = {"case_id": subject.name,
                  "native_gm_mm3": number(np.nansum(native, dtype=np.float64) * native_mm3),
                  "arms": {}, "comparisons": {}}
        gpu_native_path = subject / "T1" / "T1_gpu_raw" / "GM_prob.nii.gz"
        if gpu_native_path.exists():
            _, gpu_native, _ = image(gpu_native_path, native_img)
            native_mask = np.isfinite(native) & np.isfinite(gpu_native) & (
                (native > 0.05) | (gpu_native > 0.05))
            result["gpu_raw_gm_vs_fast"] = compare(native, gpu_native, native_mask)
            result["gpu_raw_gm_vs_fast"]["volume_ratio"] = number(
                np.nansum(gpu_native, dtype=np.float64) /
                np.nansum(native, dtype=np.float64))
        private_timing = subject / "timings.private.json"
        if private_timing.exists():
            stage_times = json.loads(private_timing.read_text())
            result["shared_preprocessing_sec"] = number(sum(
                stage_times.get(key, 0) for key in PREPROCESS_STAGES
                if isinstance(stage_times.get(key, 0), (int, float))))
        raw_gpu_timing = subject / "gpu_gm_timing.private.json"
        if raw_gpu_timing.exists():
            result["gpu_raw_gm_sec"] = number(json.loads(
                raw_gpu_timing.read_text()).get("gpu_gm_raw_seconds"))
        arm_data = {}
        for arm in BASE_ARMS + GPU_ARMS:
            directory = subject / "T1" / "T1_vbm" / arm
            if arm in GPU_ARMS and not directory.exists():
                continue
            missing = [name for name in FILES.values() if not (directory / name).exists()]
            if missing:
                raise FileNotFoundError(f"{subject.name}/{arm}: missing {', '.join(missing)}")
            target = reference_grid
            warped_img, warped, resampled = image(directory / FILES["warped"], target)
            _, modulated, resampled_mod = image(directory / FILES["modulated"], target)
            _, jacobian, resampled_jac = image(directory / FILES["jacobian"], target)
            arm_data[arm] = (warped, modulated, jacobian)
            warped_by_arm[arm].append((subject.name, warped))
            result["arms"][arm] = {
                "resampled_to_ukb_grid": bool(resampled or resampled_mod or resampled_jac),
                "timing": timing(subject, arm),
            }
        if any(arm not in arm_data for arm in BASE_ARMS):
            raise ValueError(f"{subject.name}: both FSL arms are required")
        shared = result.get("shared_preprocessing_sec")
        for arm in BASE_ARMS:
            reg = result["arms"][arm]["timing"]["total_sec"]
            result["arms"][arm]["raw_to_vbm_sec"] = number(shared + reg) if (
                shared is not None and reg is not None) else None
        for raw_arm in ("gpu_raw_ukb", "gpu_raw_hcp"):
            if raw_arm not in result["arms"]:
                continue
            gm_sec = result.get("gpu_raw_gm_sec")
            reg = result["arms"][raw_arm]["timing"]["total_sec"]
            result["arms"][raw_arm]["raw_to_vbm_sec"] = number(
                gm_sec + reg) if (gm_sec is not None and reg is not None) else None
        cases.append((result, arm_data))

    voxel_mm3 = abs(np.linalg.det(reference_grid.affine[:3, :3]))

    for result, arm_data in cases:
        for arm, (warped, modulated, jacobian) in arm_data.items():
            result["arms"][arm].update(summarize_arm(warped, modulated, jacobian,
                                                       mask, voxel_mm3))
        ukb_warp, ukb_mod, _ = arm_data["ukb"]
        for arm in ("hcp",) + GPU_ARMS:
            if arm not in arm_data:
                continue
            warped, modulated, _ = arm_data[arm]
            pair = {"warped_" + key: value for key, value in
                    compare(ukb_warp, warped, mask).items()}
            pair.update({"modulated_" + key: value for key, value in
                         compare(ukb_mod, modulated, mask).items()})
            reference_sec = result["arms"]["ukb"]["timing"]["total_sec"]
            candidate_sec = result["arms"][arm]["timing"]["total_sec"]
            pair["ukb_time_divided_by_candidate_time"] = number(
                reference_sec / candidate_sec) if (reference_sec is not None and
                                                candidate_sec is not None and
                                                candidate_sec > 0) else None
            reference_raw_sec = result["arms"]["ukb"].get("raw_to_vbm_sec")
            candidate_raw_sec = result["arms"][arm].get("raw_to_vbm_sec")
            pair["ukb_raw_time_divided_by_candidate_raw_time"] = number(
                reference_raw_sec / candidate_raw_sec) if (
                    reference_raw_sec is not None and candidate_raw_sec is not None
                    and candidate_raw_sec > 0) else None
            result["comparisons"]["ukb_vs_" + arm] = pair

    # Leave-one-subject-out similarity measures between subjects, not between
    # an output and the template it was optimised to match.
    for arm, entries in warped_by_arm.items():
        if len(entries) < 2:
            continue
        stack = np.stack([data for _, data in entries]).astype(np.float64)
        finite = np.isfinite(stack)
        stack = np.where(finite, stack, 0)
        sum_stack = stack.sum(axis=0)
        count_stack = finite.sum(axis=0)
        for case_id, warped in entries:
            i = next(index for index, pair in enumerate(entries) if pair[0] == case_id)
            count = count_stack - finite[i]
            leave_one_out = np.divide(sum_stack - stack[i], count,
                                      out=np.zeros_like(sum_stack), where=count > 0)
            evaluation_mask = mask & (count > 0)
            result = next(item[0] for item in cases if item[0]["case_id"] == case_id)
            result["arms"][arm]["loo_pearson"] = corr(warped, leave_one_out,
                                                         evaluation_mask)
            result["arms"][arm]["loo_dice_0_5"] = dice(warped, leave_one_out,
                                                         evaluation_mask)

    for result, _ in cases:
        pair = result["comparisons"]["ukb_vs_hcp"]
        for metric in ("loo_pearson", "loo_dice_0_5", "log_jacobian_sd",
                       "jacobian_invalid_or_nonpositive_fraction"):
            ukb = result["arms"]["ukb"].get(metric)
            hcp = result["arms"]["hcp"].get(metric)
            pair[metric + "_ukb_minus_hcp"] = number(ukb - hcp) if (
                ukb is not None and hcp is not None) else None

    paired_raw = [(result, arm_data) for result, arm_data in cases
                  if "gpu_raw_ukb" in arm_data and "gpu_raw_hcp" in arm_data]
    matched_scores = matched_loo(paired_raw, mask) if len(paired_raw) >= 2 else {}
    for result, arm_data in paired_raw:
        ukb_warp, ukb_mod, _ = arm_data["gpu_raw_ukb"]
        hcp_warp, hcp_mod, _ = arm_data["gpu_raw_hcp"]
        pair = {"warped_" + key: value for key, value in
                compare(ukb_warp, hcp_warp, mask).items()}
        pair.update({"modulated_" + key: value for key, value in
                     compare(ukb_mod, hcp_mod, mask).items()})
        ukb_arm, hcp_arm = result["arms"]["gpu_raw_ukb"], result["arms"]["gpu_raw_hcp"]
        ukb_sec, hcp_sec = ukb_arm["timing"]["total_sec"], hcp_arm["timing"]["total_sec"]
        pair["ukb_time_divided_by_hcp_time"] = number(
            ukb_sec / hcp_sec) if ukb_sec is not None and hcp_sec else None
        ukb_raw, hcp_raw = ukb_arm.get("raw_to_vbm_sec"), hcp_arm.get("raw_to_vbm_sec")
        pair["ukb_raw_time_divided_by_hcp_raw_time"] = number(
            ukb_raw / hcp_raw) if ukb_raw is not None and hcp_raw else None
        for metric in ("log_jacobian_sd", "jacobian_invalid_or_nonpositive_fraction",
                       "jacobian_below_0_2_fraction", "jacobian_above_5_fraction"):
            pair[metric + "_ukb_minus_hcp"] = difference(ukb_arm.get(metric),
                                                          hcp_arm.get(metric))
        for metric in ("loo_pearson", "loo_dice_0_5"):
            left = matched_scores.get((result["case_id"], "gpu_raw_ukb"), {}).get(metric)
            right = matched_scores.get((result["case_id"], "gpu_raw_hcp"), {}).get(metric)
            pair["paired_" + metric + "_ukb"] = left
            pair["paired_" + metric + "_hcp"] = right
            pair["paired_" + metric + "_ukb_minus_hcp"] = difference(left, right)
        result["comparisons"]["gpu_raw_ukb_vs_gpu_raw_hcp"] = pair

    public_cases = [item[0] for item in cases]
    comparisons = sorted({name for case in public_cases for name in case["comparisons"]})
    summary = {
        "n_cases": len(public_cases),
        "mask_voxels": int(mask.sum()),
        "mask_provenance": (
            "fixed symmetric union of the two input templates at intensity > 0.01; "
            "subject outputs do not influence the mask"
        ),
        "voxel_mm3": number(voxel_mm3),
        "arms": {},
        "comparisons": {name: paired_summary(public_cases, name) for name in comparisons},
    }
    if args.exclude_cases:
        summary["excluded_cases"] = sorted(set(args.exclude_cases))
    raw_gm_cases = [case["gpu_raw_gm_vs_fast"] for case in public_cases
                    if "gpu_raw_gm_vs_fast" in case]
    if raw_gm_cases:
        summary["gpu_raw_gm_vs_fast"] = {"n": len(raw_gm_cases)}
        for metric in ("mae", "rmse", "pearson", "dice_0_5", "volume_ratio"):
            values = [case[metric] for case in raw_gm_cases if case[metric] is not None]
            summary["gpu_raw_gm_vs_fast"][metric + "_median"] = number(
                np.median(values)) if values else None
    for arm in BASE_ARMS + GPU_ARMS:
        members = [case["arms"][arm] for case in public_cases if arm in case["arms"]]
        if members:
            summary["arms"][arm] = {"n": len(members)}
            for metric in ("loo_pearson", "loo_dice_0_5", "log_jacobian_sd",
                           "jacobian_invalid_or_nonpositive_fraction"):
                summary["arms"][arm][metric + "_mean"] = mean(
                    [member.get(metric) for member in members])
            summary["arms"][arm]["total_sec_median"] = number(np.median([
                member["timing"]["total_sec"] for member in members
                if member["timing"]["total_sec"] is not None
            ])) if any(member["timing"]["total_sec"] is not None for member in members) else None
            raw_times = [member["raw_to_vbm_sec"] for member in members
                         if member.get("raw_to_vbm_sec") is not None]
            if raw_times:
                summary["arms"][arm]["raw_to_vbm_sec_median"] = number(
                    np.median(raw_times))
    if all("loo_pearson" in case["arms"]["ukb"] and
           "loo_pearson" in case["arms"]["hcp"] for case in public_cases):
        deltas = [case["arms"]["ukb"]["loo_pearson"] -
                  case["arms"]["hcp"]["loo_pearson"] for case in public_cases
                  if case["arms"]["ukb"]["loo_pearson"] is not None and
                  case["arms"]["hcp"]["loo_pearson"] is not None]
        if deltas:
            summary["ukb_minus_hcp_loo_pearson_mean"] = number(np.mean(deltas))

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    args.out_prefix.with_name(args.out_prefix.name + ".json").write_text(json.dumps(
        {"summary": summary, "cases": public_cases}, indent=2, allow_nan=False))
    rows = []
    for case in public_cases:
        row = {"case_id": case["case_id"], "native_gm_mm3": case["native_gm_mm3"]}
        flatten("", case["arms"], row)
        flatten("comparison", case["comparisons"], row)
        rows.append(row)
    fields = ["case_id"] + sorted({key for row in rows for key in row if key != "case_id"})
    with args.out_prefix.with_name(args.out_prefix.name + ".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"cases": len(public_cases), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
