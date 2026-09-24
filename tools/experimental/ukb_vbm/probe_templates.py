"""Compare the public UKB and local HCP-derived grey-matter templates."""

import hashlib
import json
import sys

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to


def describe(path):
    img = nib.load(path)
    data = np.asarray(img.dataobj, dtype=np.float32)
    return img, data, {
        "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest(),
        "shape": list(img.shape),
        "zooms": [float(x) for x in img.header.get_zooms()[:3]],
        "affine": np.round(img.affine, 5).tolist(),
        "range": [float(np.min(data)), float(np.max(data))],
        "nonzero_voxels": int(np.count_nonzero(data)),
        "gt_0_5_voxels": int(np.count_nonzero(data > 0.5)),
    }


ukb, a, info_a = describe(sys.argv[1])
hcp, b, info_b = describe(sys.argv[2])
if ukb.shape != hcp.shape or not np.allclose(ukb.affine, hcp.affine):
    b = np.asarray(resample_from_to(hcp, ukb, order=1).dataobj, dtype=np.float32)
mask = (a > 0.01) | (b > 0.01)
intersection = (a > 0.5) & (b > 0.5)
result = {
    "ukb": info_a,
    "hcp": info_b,
    "comparison_on_ukb_grid": {
        "pearson_union_0_01": float(np.corrcoef(a[mask], b[mask])[0, 1]),
        "mae_union_0_01": float(np.mean(np.abs(a[mask] - b[mask]))),
        "dice_gt_0_5": float(2 * intersection.sum() / ((a > 0.5).sum() + (b > 0.5).sum())),
    },
}
print(json.dumps(result, indent=2))
