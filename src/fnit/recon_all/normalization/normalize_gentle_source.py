"""Experimental control-point selection for FreeSurfer's first gentle pass.

Translates MRInormGentlyFindControlPoints at FreeSurfer d932c45; the 3D
Voronoi and bias interpolation are tracked separately.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage
import torch

def gentle_controls(image: torch.Tensor) -> tuple[torch.Tensor, dict]:
    if image.ndim != 3:
        raise ValueError("expected a 3D volume")
    # Native val0/val are int, so float input intensities truncate on read.
    src = np.trunc(np.asarray(image.detach().cpu(), dtype=np.float32))
    control = np.zeros(src.shape, dtype=bool)
    counts = []
    # Native low_thresh/hi_thresh are BUFTYPE, truncating fractional bounds.
    for width, low, high in ((7, int(110 - 1.5 * 7.5), int(110 + 1.5 * 25)),
                             (5, int(110 - 7.5), int(110 + 25))):
        minimum = ndimage.minimum_filter(src, size=width, mode="nearest")
        maximum = ndimage.maximum_filter(src, size=width, mode="nearest")
        control |= (minimum >= low) & (maximum <= high)
        counts.append(int(control.sum()))
    # mriRemoveOutliers scans z, then y, then x, mutating the mask in place.
    for z, y, x in zip(*np.nonzero(control.transpose(2, 1, 0))):
        # The transpose above yields coordinates (z, y, x) in source scan order.
        lo = (max(0, x - 1), max(0, y - 1), max(0, z - 1))
        hi = (min(src.shape[0], x + 2), min(src.shape[1], y + 2),
              min(src.shape[2], z + 2))
        if control[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].sum() - 1 < 2:
            control[x, y, z] = False
    return torch.as_tensor(control.astype(np.uint8), device=image.device), {
        "first_7x7x7": counts[0], "after_5x5x5": counts[1],
        "after_outlier_removal": int(control.sum())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    source = nib.load(str(args.input))
    values = torch.as_tensor(np.asarray(source.dataobj).astype(np.float32, copy=True), device=args.device)
    control, details = gentle_controls(values)
    # The source is a float MGH; controls need the native uint8 MGH type.
    nib.save(nib.MGHImage(control.cpu().numpy(), source.affine), str(args.output))
    print(json.dumps(details))


if __name__ == "__main__":
    main()
