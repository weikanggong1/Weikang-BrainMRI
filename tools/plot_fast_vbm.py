#!/usr/bin/env python3
"""Render one privacy-safe FastVBM example from saved NIfTI outputs."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def _load(path):
    data = nib.load(str(path)).get_fdata(dtype=np.float32)
    if data.ndim != 3:
        raise ValueError(f"expected a 3D image: {path}")
    return data


def _slice(data, index=None):
    if index is None:
        support = np.sum(np.isfinite(data) & (data != 0), axis=(0, 1))
        index = int(np.argmax(support))
    index = min(max(index, 0), data.shape[2] - 1)
    return np.rot90(data[:, :, index])


def _limits(data, *, signed=False):
    finite = data[np.isfinite(data)]
    positive = finite[finite > 0]
    if positive.size == 0:
        return (0.0, 1.0)
    low = float(np.percentile(positive, 1 if signed else 0))
    high = float(np.percentile(positive, 99.5))
    return low, max(high, low + 1e-6)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="raw T1w")
    parser.add_argument("--template", type=Path, required=True, help="GM template")
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    root = args.result_dir
    panels = [
        ("Raw T1w", args.input, "gray", None),
        ("SynthStrip brain", root / "T1_brain.nii.gz", "gray", None),
        ("Bias-corrected brain", root / "T1_brain_restore.nii.gz", "gray", None),
        ("TorchFAST GM PVE", root / "T1_brain_pve_1.nii.gz", "magma", (0, 1)),
        ("UKB GM template", args.template, "magma", None),
        ("Warped GM", root / "T1_GM_to_template_GM.nii.gz", "magma", (0, 1)),
        ("Nonlinear Jacobian", root / "T1_GM_JAC_nl.nii.gz", "coolwarm", (0.2, 2.0)),
        ("Modulated GM", root / "T1_GM_to_template_GM_mod.nii.gz", "magma", None),
    ]
    loaded = [(title, _load(path), cmap, limits) for title, path, cmap, limits in panels]

    native_index = int(np.argmax(np.sum(loaded[1][1] != 0, axis=(0, 1))))
    template_index = int(np.argmax(np.sum(loaded[4][1] > 0, axis=(0, 1))))
    figure, axes = plt.subplots(2, 4, figsize=(14, 7), constrained_layout=True)
    for position, (axis, (title, data, cmap, limits)) in enumerate(
        zip(axes.flat, loaded)
    ):
        index = native_index if position < 4 else template_index
        if limits is None:
            limits = _limits(data)
        image = axis.imshow(_slice(data, index), cmap=cmap, vmin=limits[0], vmax=limits[1])
        axis.set_title(title, fontsize=11)
        axis.axis("off")
        if title == "Nonlinear Jacobian":
            figure.colorbar(image, ax=axis, fraction=0.046, pad=0.02)
    figure.suptitle(
        "FastVBM public example: native-space tissue estimation and template-space modulation",
        fontsize=14,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(args.output)


if __name__ == "__main__":
    main()
