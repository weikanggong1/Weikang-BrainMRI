"""Render a FreeSurfer-versus-package WMH comparison on one public FLAIR."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np


def plane(array, axis, index):
    return np.rot90(np.take(array, index, axis=axis))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reference_image = nib.load(args.reference)
    candidate_image = nib.load(args.candidate)
    reference = np.asarray(reference_image.dataobj)
    candidate = np.asarray(candidate_image.dataobj)
    if reference.shape != candidate.shape or not np.allclose(
            reference_image.affine, candidate_image.affine, atol=1e-5):
        raise ValueError("Reference and candidate segmentation grids differ")
    flair = resample_from_to(nib.load(args.image), reference_image, order=1).get_fdata()
    lesion = reference == 77
    slices = ((2, int(np.argmax(lesion.sum(axis=(0, 1)))), "Axial"),
              (1, int(np.argmax(lesion.sum(axis=(0, 2)))), "Coronal"))
    foreground = flair[flair > 0]
    upper = np.percentile(foreground, 99) if foreground.size else 1
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 7), constrained_layout=True)
    for row, (axis, index, direction) in enumerate(slices):
        background = plane(flair, axis, index)
        for column, segmentation in enumerate((None, reference, candidate)):
            ax = axes[row, column]
            ax.imshow(background, cmap="gray", vmin=0, vmax=upper, interpolation="nearest")
            if segmentation is not None:
                overlay = np.ma.masked_where(plane(segmentation, axis, index) != 77,
                                             plane(segmentation, axis, index))
                ax.imshow(overlay, cmap="autumn", vmin=77, vmax=78,
                          alpha=0.85, interpolation="nearest")
            ax.set_axis_off()
            if column == 0:
                ax.text(0.02, 0.98, direction, transform=ax.transAxes, color="white",
                        ha="left", va="top", fontsize=11,
                        bbox={"facecolor": "black", "alpha": 0.5, "edgecolor": "none"})
    for ax, title in zip(axes[0], ("Public FLAIR input", "FreeSurfer original", "Independent PyTorch")):
        ax.set_title(title, fontsize=12)
    disagree = int(np.count_nonzero(reference != candidate))
    fig.suptitle(f"WMH label 77 (red): {disagree:,} mismatching label voxels", fontsize=12)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, facecolor="white")
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
