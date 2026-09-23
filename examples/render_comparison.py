"""Render actual FreeSurfer/PyTorch output pairs for the README figures."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import surfa as sf


ROOT = Path(__file__).resolve().parent


def load(path):
    image = nib.as_closest_canonical(nib.load(path))
    return image.get_fdata(dtype=np.float32), image.affine


def same_grid(first, second):
    return first[0].shape == second[0].shape and np.allclose(first[1], second[1], atol=1e-5, rtol=0)


def show(ax, array, plane, index, upper):
    sl = array[:, :, index] if plane == "axial" else array[:, index, :]
    ax.imshow(np.rot90(sl), cmap="gray", vmin=0, vmax=upper, interpolation="nearest")
    ax.axis("off")


def save_panel(columns, titles, slices, path, note):
    fig, axes = plt.subplots(2, len(columns), figsize=(3.6 * len(columns), 7.3),
                             facecolor="white", layout="constrained")
    for row, (plane, indices) in enumerate(slices):
        for col, ((volume, upper), title) in enumerate(zip(columns, titles)):
            show(axes[row, col], volume, plane, indices[col], upper)
            axes[row, col].set_title(title if row == 0 else "", fontsize=13)
            if col == 0:
                axes[row, col].set_ylabel(plane, fontsize=11)
    fig.suptitle(note, fontsize=12)
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, default=ROOT / "results/reference")
    parser.add_argument("--candidate-dir", type=Path, default=ROOT / "results/python")
    parser.add_argument("--output-dir", type=Path, default=ROOT.parent / "docs/figures")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = ROOT / "data"
    source = load(data / "sub-02_T1w.nii.gz")
    ref_brain = load(args.reference_dir / "sub-02_brain.nii.gz")
    torch_brain = load(args.candidate_dir / "sub-02_brain.nii.gz")
    ref_mask = load(args.reference_dir / "sub-02_mask.nii.gz")
    torch_mask = load(args.candidate_dir / "sub-02_mask.nii.gz")
    assert all(same_grid(source, item) for item in (ref_brain, torch_brain, ref_mask, torch_mask))
    ref_binary, torch_binary = ref_mask[0] > 0, torch_mask[0] > 0
    intersection = np.count_nonzero(ref_binary & torch_binary)
    dice = 2 * intersection / (np.count_nonzero(ref_binary) + np.count_nonzero(torch_binary))
    indices = np.nonzero(ref_binary)
    coronal, axial = int(np.median(indices[1])), int(np.median(indices[2]))
    vmax = float(np.percentile(source[0][source[0] > 0], 99.5))
    save_panel([(source[0], vmax), (ref_brain[0], vmax), (torch_brain[0], vmax)],
               ["Input T1w", "FreeSurfer brain", "PyTorch brain"],
               [("axial", [axial] * 3), ("coronal", [coronal] * 3)],
               args.output_dir / "synthstrip_comparison.png",
               "SynthStrip | OpenNeuro ds000114 sub-02 | matched voxel slices")

    fixed_brain = load(args.reference_dir / "sub-01_brain.nii.gz")
    fixed_mask = load(args.reference_dir / "sub-01_mask.nii.gz")
    ref_moved = load(args.reference_dir / "sub-02_in_sub-01.nii.gz")
    torch_moved = load(args.candidate_dir / "sub-02_in_sub-01.nii.gz")
    assert all(same_grid(fixed_brain, item) for item in (fixed_mask, ref_moved, torch_moved))
    display_mask = fixed_mask[0] > 0
    native_coronal, native_axial = coronal, axial
    fixed_indices = np.nonzero(display_mask)
    coronal, axial = int(np.median(fixed_indices[1])), int(np.median(fixed_indices[2]))
    upper = float(np.percentile(source[0][source[0] > 0], 99.5))
    fixed_upper = float(np.percentile(fixed_brain[0][fixed_brain[0] > 0], 99.5))
    save_panel([(ref_brain[0], upper), (fixed_brain[0], fixed_upper),
                (ref_moved[0] * display_mask, upper), (torch_moved[0] * display_mask, upper)],
               ["Moving: sub-02", "Fixed: sub-01", "FreeSurfer registered", "PyTorch registered"],
               [("axial", [native_axial, axial, axial, axial]),
                ("coronal", [native_coronal, coronal, coronal, coronal])],
               args.output_dir / "synthmorph_comparison.png",
               "SynthMorph joint | registered columns in fixed grid; displayed through fixed brain mask")

    moved_delta = torch_moved[0].astype(np.float64) - ref_moved[0].astype(np.float64)
    moved_nrmse = float(np.sqrt(np.mean(moved_delta ** 2)) /
                        np.sqrt(np.mean(ref_moved[0].astype(np.float64) ** 2)))
    first = sf.load_warp(args.reference_dir / "sub-02_to_sub-01.mgz").convert(format=sf.Warp.Format.disp_ras)
    second = sf.load_warp(args.candidate_dir / "sub-02_to_sub-01.mgz").convert(format=sf.Warp.Format.disp_ras)
    assert first.data.shape == second.data.shape
    displacement_max_mm = float(np.max(np.linalg.norm(first.data.astype(np.float64) -
                                                       second.data.astype(np.float64), axis=-1)))
    report = {"source": "OpenNeuro ds000114 CC0, processed examples/data/sub-02_T1w.nii.gz to sub-01",
              "reference": "FreeSurfer 8.2.0 native CPU commands",
              "candidate": "freesurfer-torch 0.2.0, two-GPU BatchRunner",
              "synthstrip": {"mask_dice": dice,
                             "mask_disagreeing_voxels": int(np.count_nonzero(ref_binary != torch_binary)),
                             "brain_max_abs_error": float(np.max(np.abs(ref_brain[0] - torch_brain[0]))),
                             "output_geometry_equal": True},
              "synthmorph": {"moved_nrmse": moved_nrmse,
                             "warp_vector_max_error_mm": displacement_max_mm,
                             "output_geometry_equal": True},
              "figure_note": "Registered images are displayed through the fixed brain mask only for visualization; metrics use the complete saved outputs.",
              "figures": ["synthstrip_comparison.png", "synthmorph_comparison.png"]}
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
