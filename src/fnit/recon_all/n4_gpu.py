"""Experimental Torch bias-field correction for isolated N4 comparisons.

This estimates a smooth multiplicative field from tissue-intensity residuals.
It is not a translation of ANTs/ITK N4 and is not wired into recon-all.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F


def _blur(values: torch.Tensor, sigma: float) -> torch.Tensor:
    radius = max(1, round(3 * sigma))
    offsets = torch.arange(-radius, radius + 1, device=values.device, dtype=values.dtype)
    kernel = torch.exp(-0.5 * (offsets / sigma).square())
    kernel /= kernel.sum()
    result = values[None, None]
    for shape, padding in (((1, 1, -1), (radius, radius, 0, 0, 0, 0)),
                           ((1, -1, 1), (0, 0, radius, radius, 0, 0)),
                           ((-1, 1, 1), (0, 0, 0, 0, radius, radius))):
        weights = kernel.reshape(1, 1, *shape)
        result = F.conv3d(F.pad(result, padding, mode="replicate"), weights)
    return result[0, 0]


def _class_centres(values: torch.Tensor, count: int = 3) -> torch.Tensor:
    centres = torch.quantile(values, torch.linspace(0.15, 0.85, count, device=values.device))
    for _ in range(5):
        nearest = (values[:, None] - centres).abs().argmin(dim=1)
        for index in range(count):
            selected = values[nearest == index]
            if selected.numel():
                centres[index] = selected.mean()
    return centres


def estimate_log_bias(image: torch.Tensor, *, shrink: int = 4,
                      iterations: int = 4, sigma: float = 7.0) -> torch.Tensor:
    """Estimate a low-frequency log bias field on the input image grid."""
    if image.ndim != 3 or min(image.shape) < shrink or shrink < 1:
        raise ValueError("expected a 3D image larger than the shrink factor")
    if not torch.is_floating_point(image):
        image = image.float()
    if torch.any(image < 0):
        raise ValueError("intensities must be nonnegative")
    foreground = (image > 0).float()
    pooled_mask = F.avg_pool3d(foreground[None, None], shrink, shrink)[0, 0]
    pooled_image = F.avg_pool3d((image * foreground)[None, None], shrink, shrink)[0, 0]
    valid = pooled_mask > 0.25
    if not valid.any():
        raise ValueError("image has no foreground")
    log_image = (pooled_image / pooled_mask.clamp_min(1e-6)).clamp_min(1).log()
    field = torch.zeros_like(log_image)
    smooth_mask = _blur(valid.float(), sigma).clamp_min(1e-6)
    for _ in range(iterations):
        corrected = log_image - field
        centres = _class_centres(corrected[valid])
        labels = (corrected[..., None] - centres).abs().argmin(dim=-1)
        residual = torch.where(valid, corrected - centres[labels], 0)
        adjustment = _blur(residual, sigma) / smooth_mask
        adjustment -= adjustment[valid].mean()
        field += 0.65 * adjustment
    return F.interpolate(field[None, None], size=image.shape,
                         mode="trilinear", align_corners=False)[0, 0]


def correct(image: torch.Tensor) -> torch.Tensor:
    """Return a float corrected image on the same device as ``image``."""
    field = estimate_log_bias(image)
    return torch.where(image > 0, image.float() * torch.exp(-field), 0)


def run(input_file: str | Path, output_file: str | Path, *, device: str = "cuda:0") -> None:
    """Replay the isolated bias-correction step; output remains float32."""
    source = nib.load(str(input_file))
    if not isinstance(source, nib.MGHImage) or len(source.shape) != 3:
        raise ValueError("expected a 3D MGH/MGZ input")
    image = torch.from_numpy(np.asarray(source.dataobj).astype(np.float32)).to(device)
    corrected = correct(image).cpu().numpy()
    result = nib.MGHImage(corrected, source.affine, source.header)
    result.header.set_data_dtype(np.float32)
    nib.save(result, str(output_file))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--i", required=True, dest="input_file")
    parser.add_argument("--o", required=True, dest="output_file")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    run(args.input_file, args.output_file, device=args.device)


if __name__ == "__main__":
    main()
