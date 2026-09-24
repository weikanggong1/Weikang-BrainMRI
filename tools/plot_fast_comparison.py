#!/usr/bin/env python3
"""Plot FSL FAST and TorchFAST outputs from the same public brain image."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def load(path, reference=None):
    image = nib.as_closest_canonical(nib.load(path))
    data = np.asarray(image.dataobj, dtype=np.float32)
    if data.ndim != 3 or not np.isfinite(data).all():
        raise ValueError(f"expected a finite 3D image: {path}")
    if reference is not None and (data.shape != reference[0] or not np.allclose(
            image.affine, reference[1], atol=1e-5, rtol=0)):
        raise ValueError(f"image grid differs from the brain image: {path}")
    return data, (data.shape, image.affine)


def view(data, axis, index):
    return np.rot90(np.take(data, index, axis=axis))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brain", type=Path, required=True)
    parser.add_argument("--fsl-prefix", type=Path, required=True)
    parser.add_argument("--torch-prefix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    args = parser.parse_args()

    brain, geometry = load(args.brain)
    fsl_gm, _ = load(f"{args.fsl_prefix}_pve_1.nii.gz", geometry)
    torch_gm, _ = load(f"{args.torch_prefix}_pve_1.nii.gz", geometry)
    fsl_restore, _ = load(f"{args.fsl_prefix}_restore.nii.gz", geometry)
    torch_restore, _ = load(f"{args.torch_prefix}_restore.nii.gz", geometry)
    mask = brain > 0
    coordinates = np.argwhere(mask)
    if not len(coordinates):
        raise ValueError("brain image has no positive voxels")
    center = np.round(np.median(coordinates, axis=0)).astype(int)
    slices = ((2, int(center[2]), "Axial"), (1, int(center[1]), "Coronal"))
    low, high = np.percentile(brain[mask], (1, 99.5))
    corrected = np.concatenate((fsl_restore[mask], torch_restore[mask]))
    corrected_low, corrected_high = np.percentile(corrected, (1, 99.5))
    difference = np.abs(fsl_gm - torch_gm)

    figure, axes = plt.subplots(2, 6, figsize=(15.2, 6.4), constrained_layout=True)
    titles = ("Brain-only T1w", "FSL GM PVE", "TorchFAST GM PVE",
              "Absolute PVE difference", "FSL bias-corrected", "TorchFAST bias-corrected")
    for column, title in enumerate(titles):
        axes[0, column].set_title(title, fontsize=11)
    for row, (axis, index, label) in enumerate(slices):
        background = view(brain, axis, index)
        axes[row, 0].imshow(background, cmap="gray", vmin=low, vmax=high)
        for column, gm in ((1, fsl_gm), (2, torch_gm)):
            axes[row, column].imshow(background, cmap="gray", vmin=low, vmax=high)
            overlay = np.ma.masked_where(view(gm, axis, index) <= 0.05,
                                         view(gm, axis, index))
            axes[row, column].imshow(overlay, cmap="magma", vmin=0, vmax=1, alpha=0.72)
        axes[row, 3].imshow(view(difference, axis, index), cmap="viridis", vmin=0, vmax=0.2)
        axes[row, 4].imshow(view(fsl_restore, axis, index), cmap="gray",
                            vmin=corrected_low, vmax=corrected_high)
        axes[row, 5].imshow(view(torch_restore, axis, index), cmap="gray",
                            vmin=corrected_low, vmax=corrected_high)
        axes[row, 0].set_ylabel(label, fontsize=11)
        for panel in axes[row]:
            panel.set_xticks([])
            panel.set_yticks([])
    figure.suptitle("Same public T1w input: FSL FAST and TorchFAST", fontsize=14)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, facecolor="white")
    plt.close(figure)

    x, y = torch_gm[mask].astype(np.float64), fsl_gm[mask].astype(np.float64)
    xb, yb = x >= 0.5, y >= 0.5
    metrics = {
        "input": "examples/data/sub-02_T1w.nii.gz after the published reference SynthStrip output",
        "mask_voxels": int(mask.sum()),
        "gm_pearson": float(np.corrcoef(x, y)[0, 1]),
        "gm_mae": float(np.mean(np.abs(x - y))),
        "gm_dice_at_0_5": float(2 * (xb & yb).sum() / (xb.sum() + yb.sum())),
        "gm_volume_ratio_torch_over_fsl": float(x.sum() / y.sum()),
        "figure_slices_canonical_voxel": {label.lower(): index for _, index, label in slices},
    }
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
