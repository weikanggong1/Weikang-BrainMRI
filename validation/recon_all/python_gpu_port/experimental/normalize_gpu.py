"""Experimental, approximate Torch replay of two mri_normalize calls.

This does not reproduce FreeSurfer's spline, control-point, or Voronoi code.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

from fnit.recon_all.mgh_compat import save_same_dtype_mgh


def _mode(values: torch.Tensor, low: int, high: int) -> int:
    counts = torch.bincount(values.round().long().clamp(0, 255).flatten(), minlength=256).float()
    offset = torch.arange(-6, 7, device=values.device, dtype=torch.float32)
    kernel = torch.exp(-0.5 * (offset / 2).square())
    kernel /= kernel.sum()
    smooth = F.conv1d(counts[None, None], kernel[None, None], padding=6)[0, 0]
    return int(smooth[low:high + 1].argmax()) + low


def _blur(values: torch.Tensor, sigma: float) -> torch.Tensor:
    radius = round(3 * sigma)
    offset = torch.arange(-radius, radius + 1, device=values.device)
    kernel = torch.exp(-0.5 * (offset / sigma).square())
    kernel /= kernel.sum()
    result = values[None, None]
    for shape, padding in (((1, 1, -1), (radius, radius, 0, 0, 0, 0)),
                           ((1, -1, 1), (0, 0, radius, radius, 0, 0)),
                           ((-1, 1, 1), (0, 0, 0, 0, radius, radius))):
        result = F.conv3d(F.pad(result, padding, mode="replicate"),
                          kernel.reshape(1, 1, *shape))
    return result[0, 0]


def _ridge(mask: torch.Tensor) -> torch.Tensor:
    """Approximate WM medial-axis controls using a 26-neighbor depth map."""
    inside = mask
    depth = torch.zeros(mask.shape, device=mask.device, dtype=torch.float32)
    for _ in range(min(mask.shape)):
        if not bool(inside.any()):
            break
        depth += inside
        inside = F.max_pool3d((~inside).float()[None, None], 3, stride=1,
                              padding=1)[0, 0] == 0
    maxima = F.max_pool3d(depth[None, None], 3, stride=1, padding=1)[0, 0]
    return mask & (depth >= maxima)


@torch.inference_mode()
def normalize_tensor(source: torch.Tensor, *, aseg: torch.Tensor | None = None,
                     mask: torch.Tensor | None = None) -> tuple[torch.Tensor, dict]:
    if source.ndim != 3:
        raise ValueError("expected a 3D source volume")
    image = source.float()
    if aseg is None:
        peak = _mode(image, 90, 140)
        corrected = image * (110.0 / peak)
        details = {"phase": "first", "mode": peak, "control_voxels": 0}
    else:
        if aseg.shape != source.shape or mask is None or mask.shape != source.shape:
            raise ValueError("aseg and brain mask must match the source geometry")
        image = image * (mask > 0)
        wm = (aseg == 2) | (aseg == 41)
        peak = _mode(image[wm], 1, 254)
        control = _ridge(wm) & (image >= peak - 10)
        count = int(control.sum())
        if count == 0:
            raise ValueError("no white-matter control voxels remain")
        block = min(4, *source.shape)
        pooled = lambda data: F.avg_pool3d(data[None, None], block, block)[0, 0]
        numerator = _blur(pooled(image * control), 8.0 / block)
        denominator = _blur(pooled(control.float()), 8.0 / block)
        bias = torch.where(denominator > 1e-6, numerator / denominator.clamp_min(1e-6),
                           peak)
        bias = F.interpolate(bias[None, None], size=source.shape, mode="trilinear",
                             align_corners=False)[0, 0]
        corrected = image * 110.0 / bias.clamp_min(1)
        details = {"phase": "second", "mode": peak, "control_voxels": count}
    return corrected.add_(0.5).floor_().clamp_(0, 255).to(torch.uint8), details


def run(input_file: str | Path, output_file: str | Path, *, phase: str,
        aseg_file: str | Path | None = None, mask_file: str | Path | None = None,
        device: str = "cuda:0") -> dict:
    source = nib.load(str(input_file))
    if not isinstance(source, nib.MGHImage) or len(source.shape) != 3 or source.get_data_dtype() != np.uint8:
        raise ValueError("expected a 3D uint8 MGH/MGZ input")
    image = torch.as_tensor(np.asarray(source.dataobj).copy(), device=device)
    extra = {}
    if phase == "second":
        if not aseg_file or not mask_file:
            raise ValueError("the second pass requires aseg and brain mask")
        for name, filename in (("aseg", aseg_file), ("mask", mask_file)):
            loaded = nib.load(str(filename))
            if loaded.shape != source.shape or not np.array_equal(loaded.affine, source.affine):
                raise ValueError(f"{name} geometry differs from the source")
            dtype = np.int32 if name == "aseg" else np.uint8
            extra[name] = torch.as_tensor(np.asarray(loaded.dataobj).astype(dtype, copy=True), device=device)
    elif phase != "first":
        raise ValueError("phase must be first or second")
    result, details = normalize_tensor(image, **extra)
    save_same_dtype_mgh(input_file, output_file, result.cpu().numpy())
    return details


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--i", required=True, dest="input_file")
    parser.add_argument("--o", required=True, dest="output_file")
    parser.add_argument("--phase", choices=("first", "second"), required=True)
    parser.add_argument("--aseg")
    parser.add_argument("--mask")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    details = run(args.input_file, args.output_file, phase=args.phase,
                  aseg_file=args.aseg, mask_file=args.mask, device=args.device)
    print(json.dumps(details))


if __name__ == "__main__":
    main()
