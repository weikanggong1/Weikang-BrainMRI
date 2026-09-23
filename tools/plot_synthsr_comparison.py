"""Show matching RAS slices in three NIfTI images (requires matplotlib)."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy.ndimage import map_coordinates


PLANES = (
    ("Axial", 0, 1, 2),
    ("Coronal", 0, 2, 1),
    ("Sagittal", 1, 2, 0),
)


def load_image(path):
    image = nib.load(str(path))
    if len(image.shape) != 3:
        raise ValueError(f"Expected a 3D NIfTI image: {path}")
    return image.get_fdata(dtype=np.float32), image.affine


def ras_bounds(shape, affine):
    corners = np.array(np.meshgrid(*[(0, size - 1) for size in shape], indexing="ij"))
    voxels = np.vstack((corners.reshape(3, -1), np.ones(8)))
    ras = (affine @ voxels)[:3]
    return ras.min(axis=1), ras.max(axis=1)


def foreground_center(data, affine):
    threshold = max(1.0, float(np.percentile(data, 99)) * 0.1)
    mask = np.isfinite(data) & (data > threshold)
    if not mask.any():
        voxel = (np.asarray(data.shape) - 1) / 2
    else:
        voxel = []
        for axis in range(3):
            counts = mask.sum(axis=tuple(i for i in range(3) if i != axis))
            voxel.append(np.searchsorted(np.cumsum(counts), counts.sum() / 2))
    return (affine @ np.r_[voxel, 1])[:3]


def sample_plane(data, affine, bounds, center, horizontal, vertical, fixed):
    u = np.arange(np.floor(bounds[0][horizontal]), np.ceil(bounds[1][horizontal]) + 1)
    v = np.arange(np.floor(bounds[0][vertical]), np.ceil(bounds[1][vertical]) + 1)
    x, y = np.meshgrid(u, v)
    ras = np.empty((3, x.size), dtype=np.float64)
    ras[horizontal] = x.ravel()
    ras[vertical] = y.ravel()
    ras[fixed] = center[fixed]
    voxels = np.linalg.inv(affine) @ np.vstack((ras, np.ones(x.size)))
    return map_coordinates(data, voxels[:3], order=1, mode="constant", cval=0).reshape(x.shape)


def contrast(planes):
    values = np.concatenate([plane.ravel() for plane in planes])
    values = values[np.isfinite(values) & (values != 0)]
    if values.size == 0:
        return 0.0, 1.0
    low, high = np.percentile(values, (1, 99))
    return float(low), float(max(high, low + 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True, help="original FLAIR or other 3D scan")
    parser.add_argument("--official", type=Path, required=True, help="FreeSurfer mri_synthsr output")
    parser.add_argument("--torch", type=Path, required=True, help="PyTorch SynthSR output")
    parser.add_argument("--output", type=Path, required=True, help="PNG path")
    parser.add_argument("--ras", type=float, nargs=3, metavar=("R", "A", "S"),
                        help="optional slice intersection in RAS millimetres")
    parser.add_argument("--input-label", default="Original input", help="first-column title")
    args = parser.parse_args()
    if args.output.suffix.lower() != ".png":
        parser.error("--output must end in .png")

    images = [load_image(path) for path in (args.original, args.official, args.torch)]
    bounds = ras_bounds(images[1][0].shape, images[1][1])
    center = np.asarray(args.ras) if args.ras else foreground_center(*images[1])
    if np.any(center < bounds[0]) or np.any(center > bounds[1]):
        parser.error("--ras is outside the official output's RAS bounding box")

    slices = [
        [sample_plane(data, affine, bounds, center, horizontal, vertical, fixed)
         for data, affine in images]
        for _, horizontal, vertical, fixed in PLANES
    ]
    input_scale = contrast([row[0] for row in slices])
    synth_scale = contrast([row[1] for row in slices])
    fig, axes = plt.subplots(3, 3, figsize=(12, 11), constrained_layout=True)
    for column, title in enumerate((args.input_label, "FreeSurfer mri_synthsr", "PyTorch SynthSR")):
        axes[0, column].set_title(title, fontsize=13)
    for row, (name, _, _, fixed) in enumerate(PLANES):
        for column in range(3):
            low, high = input_scale if column == 0 else synth_scale
            axes[row, column].imshow(slices[row][column], cmap="gray", origin="lower",
                                     vmin=low, vmax=high, interpolation="nearest")
            axes[row, column].set_aspect("equal")
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
        axes[row, 0].set_ylabel(f"{name}\n{'RAS'[fixed]}={center[fixed]:.1f} mm", fontsize=11)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, facecolor="white")
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
