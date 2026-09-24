"""Create privacy-safe cohort QC figures from evaluate.py output.

python qc_figures.py --evaluation reports/vbm.private.json \
    --subjects-root subjects --out-dir reports/figures

Figures contain cohort means or paired aggregate summaries. A comparison is
drawn only when at least ten matched cases are available.
"""

import argparse
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np


MIN_PUBLIC_N = 10
FILES = {
    "warped": "T1_GM_to_template_GM.nii.gz",
    "modulated": "T1_GM_to_template_GM_mod.nii.gz",
}
ARM_META = {
    "ukb": ("FSL · UKB", "#1769aa"),
    "hcp": ("FSL · local HCP-derived", "#d17b21"),
    "gpu_ukb": ("GPU · FAST → UKB", "#388554"),
    "gpu_raw_ukb": ("GPU · raw T1 → UKB", "#934ca2"),
    "gpu_raw_hcp": ("GPU · raw T1 → local HCP-derived", "#c94f9d"),
}
MAP_PAIRS = (
    ("fsl_ukb_hcp", "ukb", "hcp", "FSL template comparison"),
    ("gpu_raw_ukb_hcp", "gpu_raw_ukb", "gpu_raw_hcp",
     "GPU template comparison"),
    ("fsl_gpu_fast_ukb", "ukb", "gpu_ukb",
     "FSL–GPU comparison · matched FAST input and UKB template"),
)


def finite_number(value):
    return isinstance(value, (int, float)) and np.isfinite(value)


def matched_cases(cases, *arms):
    return [case for case in cases
            if all(arm in case.get("arms", {}) for arm in arms)]


def load_image(path, grid=None):
    image = nib.load(path)
    if grid is not None and (image.shape != grid.shape or
                             not np.allclose(image.affine, grid.affine, atol=1e-4)):
        image = resample_from_to(image, grid, order=1)
    return image, np.asarray(image.dataobj, dtype=np.float32)


def cohort_maps(cases, subjects_root, arms, filename):
    grid = None
    totals = {}
    counts = {}
    for case in cases:
        for arm in arms:
            path = subjects_root / case["case_id"] / "T1" / "T1_vbm" / arm / filename
            image, data = load_image(path, grid)
            if grid is None:
                grid = image
                totals = {name: np.zeros(grid.shape, dtype=np.float64) for name in arms}
                counts = {name: np.zeros(grid.shape, dtype=np.uint16) for name in arms}
            valid = np.isfinite(data)
            totals[arm] += np.where(valid, data, 0)
            counts[arm] += valid
    return {arm: np.divide(totals[arm], counts[arm],
                           out=np.zeros_like(totals[arm]), where=counts[arm] > 0)
            for arm in arms}


def axial(data, index):
    return np.rot90(data[:, :, index])


def draw_maps(maps, arms, title, image_kind, count, destination):
    left, right = (maps[arm] for arm in arms)
    difference = left - right
    occupied = (left > 0.02) | (right > 0.02)
    z = np.where(occupied.any(axis=(0, 1)))[0]
    if len(z) < 3:
        raise ValueError("Too few occupied slices for a QC map")
    slices = np.unique(np.quantile(z, [0.35, 0.50, 0.65]).astype(int))
    if len(slices) < 3:
        slices = np.linspace(z[0], z[-1], 3).astype(int)
    intensity_max = max(0.5, float(np.percentile(
        np.concatenate((left[occupied], right[occupied])), 99)))
    difference_max = max(0.01, float(np.percentile(np.abs(difference[occupied]), 99)))
    left_label, right_label = (ARM_META[arm][0] for arm in arms)

    fig, axes = plt.subplots(3, 3, figsize=(10.5, 9), constrained_layout=True)
    rows = ((left, left_label), (right, right_label),
            (difference, f"{left_label} − {right_label}"))
    for column, index in enumerate(slices):
        for row, (data, label) in enumerate(rows):
            if row == 2:
                picture = axes[row, column].imshow(
                    axial(data, index), cmap="coolwarm",
                    vmin=-difference_max, vmax=difference_max)
            else:
                picture = axes[row, column].imshow(
                    axial(data, index), cmap="gray", vmin=0, vmax=intensity_max)
            if row == 0:
                axes[row, column].set_title(f"Axial slice {column + 1}")
            axes[row, column].axis("off")
    row_labels = (left_label.replace("local ", ""),
                  right_label.replace("local ", ""),
                  "Difference (first − second)")
    for row, label in enumerate(row_labels):
        axes[row, 0].annotate(
            label, xy=(-0.10, 0.5), xycoords="axes fraction", rotation=90,
            ha="center", va="center", fontsize=9, annotation_clip=False)
    fig.colorbar(axes[0, 2].images[0],
                 ax=axes[:2, :].ravel().tolist(), shrink=0.72,
                 label=f"Mean {image_kind} GM")
    fig.colorbar(axes[2, 2].images[0], ax=axes[2, :].ravel().tolist(),
                 shrink=0.72, label="Mean difference")
    fig.suptitle(f"{title} · cohort mean {image_kind} GM · matched N={count}")
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)


def paired_values(cases, left_arm, right_arm, getter, positive=False):
    values = []
    for case in matched_cases(cases, left_arm, right_arm):
        left = getter(case, left_arm)
        right = getter(case, right_arm)
        if not finite_number(left) or not finite_number(right):
            continue
        if positive and (left <= 0 or right <= 0):
            continue
        values.append((float(left), float(right)))
    return values


def draw_paired(axis, values, arms, title, ylabel, log=False):
    if len(values) < MIN_PUBLIC_N:
        axis.text(0.5, 0.5, f"Not shown: matched N={len(values)} < {MIN_PUBLIC_N}",
                  ha="center", va="center", transform=axis.transAxes)
        axis.set_title(title)
        axis.axis("off")
        return False
    array = np.asarray(values)
    x = np.arange(2)
    labels = []
    for column, arm in enumerate(arms):
        label, color = ARM_META[arm]
        q25, median, q75 = np.quantile(array[:, column], (0.25, 0.5, 0.75))
        axis.vlines(x[column], q25, q75, color=color, linewidth=8,
                    alpha=0.45, zorder=1)
        axis.scatter(x[column], median, s=55, color=color, zorder=2)
        axis.annotate(f"median {median:.3g}", (x[column], median),
                      xytext=(0, 9), textcoords="offset points",
                      ha="center", fontsize=8)
        labels.append(label.replace(" · ", "\n"))
    if log:
        axis.set_yscale("log")
    axis.set_ylabel(ylabel)
    axis.set_xticks(x, labels)
    axis.set_xlim(-0.6, 1.6)
    axis.grid(axis="y", alpha=0.25)
    if np.allclose(array, 0):
        axis.text(0.5, 0.88, f"All {len(values)} values are zero",
                  ha="center", transform=axis.transAxes, fontsize=8)
    axis.set_title(f"{title} · matched N={len(values)}")
    return True


def draw_times(cases, destination):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    registration = paired_values(
        cases, "ukb", "gpu_ukb",
        lambda case, arm: case["arms"][arm].get("timing", {}).get("total_sec"),
        positive=True)
    raw_to_vbm = paired_values(
        cases, "ukb", "gpu_raw_ukb",
        lambda case, arm: case["arms"][arm].get("raw_to_vbm_sec"), positive=True)
    shown = draw_paired(
        axes[0], registration, ("ukb", "gpu_ukb"),
        "Registration, Jacobian, modulation and I/O", "Time (s, log scale)", log=True)
    shown |= draw_paired(
        axes[1], raw_to_vbm, ("ukb", "gpu_raw_ukb"),
        "Raw T1 to VBM", "Time (s, log scale)", log=True)
    if shown:
        fig.suptitle("Paired processing time")
        fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return shown


def metric_pairs(cases, comparison, left_key, right_key):
    values = []
    for case in cases:
        item = case.get("comparisons", {}).get(comparison, {})
        left, right = item.get(left_key), item.get(right_key)
        if finite_number(left) and finite_number(right):
            values.append((float(left), float(right)))
    return values


def draw_template_loo(cases, destination):
    fsl_cases = matched_cases(cases, "ukb", "hcp")
    raw_cases = matched_cases(cases, "gpu_raw_ukb", "gpu_raw_hcp")
    fsl_pearson = paired_values(
        fsl_cases, "ukb", "hcp",
        lambda case, arm: case["arms"][arm].get("loo_pearson"))
    fsl_dice = paired_values(
        fsl_cases, "ukb", "hcp",
        lambda case, arm: case["arms"][arm].get("loo_dice_0_5"))
    raw_pearson = metric_pairs(
        raw_cases, "gpu_raw_ukb_vs_gpu_raw_hcp",
        "paired_loo_pearson_ukb", "paired_loo_pearson_hcp")
    raw_dice = metric_pairs(
        raw_cases, "gpu_raw_ukb_vs_gpu_raw_hcp",
        "paired_loo_dice_0_5_ukb", "paired_loo_dice_0_5_hcp")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    shown = draw_paired(axes[0, 0], fsl_pearson, ("ukb", "hcp"),
                        "FSL template comparison", "LOO GM correlation")
    shown |= draw_paired(axes[0, 1], fsl_dice, ("ukb", "hcp"),
                         "FSL template comparison", "LOO GM Dice at 0.5")
    shown |= draw_paired(
        axes[1, 0], raw_pearson, ("gpu_raw_ukb", "gpu_raw_hcp"),
        "GPU template comparison", "Matched LOO GM correlation")
    shown |= draw_paired(
        axes[1, 1], raw_dice, ("gpu_raw_ukb", "gpu_raw_hcp"),
        "GPU template comparison", "Matched LOO GM Dice at 0.5")
    if shown:
        fig.suptitle("Paired leave-one-subject-out cohort consistency\n"
                     "Consistency is not a measure of anatomical accuracy")
        fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return shown


def draw_agreement(axis, values, labels, title, ylabel):
    if len(values) < MIN_PUBLIC_N:
        axis.text(0.5, 0.5, f"Not shown: matched N={len(values)} < {MIN_PUBLIC_N}",
                  ha="center", va="center", transform=axis.transAxes)
        axis.set_title(title)
        axis.axis("off")
        return False
    array = np.asarray(values)
    x = np.arange(2)
    colors = ("#4c78a8", "#e45756")
    for column, (label, color) in enumerate(zip(labels, colors)):
        q25, median, q75 = np.quantile(array[:, column], (0.25, 0.5, 0.75))
        axis.vlines(x[column], q25, q75, color=color, linewidth=8,
                    alpha=0.45, zorder=1)
        axis.scatter(x[column], median, s=55, color=color, zorder=2)
        axis.annotate(f"median {median:.3g}", (x[column], median),
                      xytext=(0, 9), textcoords="offset points",
                      ha="center", fontsize=8)
    axis.set_ylabel(ylabel)
    axis.set_xticks(x, labels)
    axis.set_xlim(-0.6, 1.6)
    axis.grid(axis="y", alpha=0.25)
    axis.set_title(f"{title} · matched N={len(values)}")
    return True


def draw_gpu_fsl_qc(cases, destination):
    pair_cases = matched_cases(cases, "ukb", "gpu_ukb")
    pearson = metric_pairs(pair_cases, "ukb_vs_gpu_ukb",
                           "warped_pearson", "modulated_pearson")
    dice = metric_pairs(pair_cases, "ukb_vs_gpu_ukb",
                        "warped_dice_0_5", "modulated_dice_0_5")
    invalid = paired_values(
        pair_cases, "ukb", "gpu_ukb",
        lambda case, arm: case["arms"][arm].get(
            "jacobian_invalid_or_nonpositive_fraction"))
    log_sd = paired_values(
        pair_cases, "ukb", "gpu_ukb",
        lambda case, arm: case["arms"][arm].get("log_jacobian_sd"))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    shown = draw_agreement(axes[0, 0], pearson, ("Warped", "Modulated"),
                           "GPU agreement with FSL", "Pearson correlation")
    shown |= draw_agreement(axes[0, 1], dice, ("Warped", "Modulated"),
                            "GPU agreement with FSL", "Dice at GM ≥ 0.5")
    shown |= draw_paired(
        axes[1, 0], invalid, ("ukb", "gpu_ukb"), "Jacobian validity",
        "Invalid or nonpositive fraction")
    shown |= draw_paired(
        axes[1, 1], log_sd, ("ukb", "gpu_ukb"), "Jacobian dispersion",
        "SD of log positive Jacobian")
    if shown:
        fig.suptitle("FSL–GPU agreement and deformation QC · matched FAST input and UKB template\n"
                     "Agreement measures reproducibility, not anatomical accuracy")
        fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return shown


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--subjects-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    cases = json.loads(args.evaluation.read_text())["cases"]
    if len(cases) < MIN_PUBLIC_N:
        raise ValueError(f"At least {MIN_PUBLIC_N} subjects are required for public figures")
    if len({case["case_id"] for case in cases}) != len(cases) or any(
            re.fullmatch(r"case\d+", case["case_id"]) is None for case in cases):
        raise ValueError("Evaluation JSON must contain unique anonymous caseXX IDs")
    cases.sort(key=lambda case: int(case["case_id"][4:]))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    generated = []
    for stem, left_arm, right_arm, title in MAP_PAIRS:
        pair_cases = matched_cases(cases, left_arm, right_arm)
        if len(pair_cases) < MIN_PUBLIC_N:
            print(f"Skipped {stem}: matched N={len(pair_cases)} < {MIN_PUBLIC_N}")
            continue
        for image_kind, filename in FILES.items():
            destination = args.out_dir / f"{stem}_{image_kind}_means.png"
            maps = cohort_maps(pair_cases, args.subjects_root,
                               (left_arm, right_arm), filename)
            draw_maps(maps, (left_arm, right_arm), title, image_kind,
                      len(pair_cases), destination)
            generated.append(destination.name)

    runtime_path = args.out_dir / "vbm_runtime_paired.png"
    if draw_times(cases, runtime_path):
        generated.append(runtime_path.name)
    loo_path = args.out_dir / "vbm_template_loo_paired.png"
    if draw_template_loo(cases, loo_path):
        generated.append(loo_path.name)
    qc_path = args.out_dir / "vbm_gpu_fsl_agreement_qc.png"
    if draw_gpu_fsl_qc(cases, qc_path):
        generated.append(qc_path.name)
    if not generated:
        raise ValueError("No comparison had at least ten matched cases")
    print(f"Saved {len(generated)} privacy-safe cohort figures.")


if __name__ == "__main__":
    main()
