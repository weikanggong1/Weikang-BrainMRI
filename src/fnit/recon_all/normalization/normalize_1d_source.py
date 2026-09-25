"""Translation of FreeSurfer's first mri_normalize 1D pass.

The histogram windows and peak filter follow utils/mrinorm.cpp at d932c45.
This module stops before MRI3dGentleNormalize.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import nibabel as nib
import numpy as np
from scipy.interpolate import CubicSpline
import torch

from ..mgh_compat import save_same_dtype_mgh


def _nint(value: float) -> int:
    return int(value + 0.5) if value >= 0 else int(value - 0.5)


def _talairach_origin(affine: np.ndarray, xfm_path: Path) -> tuple[int, int, int]:
    source = xfm_path.read_text()
    section = source.split("Linear_Transform =", 1)[1].split(";", 1)[0]
    values = [float(x) for x in re.findall(r"[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?", section)]
    if len(values) != 12:
        raise ValueError("expected a 3x4 Talairach affine")
    matrix = np.eye(4)
    matrix[:3] = np.asarray(values).reshape(3, 4)
    origin = np.linalg.solve(affine, np.linalg.solve(matrix, np.array([0., 0., 0., 1.])))
    return tuple(_nint(float(x)) for x in origin[:3])


def _smooth_histogram(counts: np.ndarray) -> np.ndarray:
    # HISTOsmooth(sigma=2) uses a Gaussian core and polynomial tails.
    sigma = 2.0
    offsets = np.arange(-8, 9, dtype=np.float32)
    distance = np.abs(offsets)
    kernel = np.where(distance <= 2 * sigma,
                      np.exp(-offsets * offsets / (2 * sigma * sigma)),
                      np.where(distance <= 4 * sigma,
                               (4 - distance / sigma) ** 4 / (16 * np.e ** 2), 0))
    kernel = kernel.astype(np.float32)
    kernel /= kernel.sum(dtype=np.float32)
    output = np.empty(len(counts), dtype=np.float32)
    for b in range(len(counts)):
        lo, hi = max(0, b - 8), min(len(counts), b + 9)
        weights = kernel[lo - b + 8:hi - b + 8]
        output[b] = np.sum(weights * counts[lo:hi], dtype=np.float32) / weights.sum(dtype=np.float32)
    return output


def _last_peak(bins: np.ndarray, counts: np.ndarray) -> int:
    # HISTOfindLastPeakInRegion(h, 7, .15, 3, nbins-3).
    b0 = next((max(0, b - 1) for b in range(len(bins)) if bins[b] >= 3), -1)
    b1 = next((min(len(bins) - 1, b + 1) for b in range(len(bins)) if bins[b] >= len(bins) - 3), len(bins) - 1)
    maximum = float(counts.max())
    if maximum <= 0 or b0 < 0:
        return -1
    for b in range(b1, b0 - 1, -1):
        lo, hi = max(0, b - 3), min(len(bins), b + 4)
        neighborhood = counts[lo:hi]
        center = float(counts[b])
        if float(neighborhood.max()) > center:
            continue
        mean = float(neighborhood.sum(dtype=np.float32)) / 7.0
        if mean >= 0.15 * maximum and center - mean >= 1.9:
            return b
    return -1


def _filter_peaks(positions: list[float], intensities: list[float]) -> tuple[np.ndarray, np.ndarray]:
    if len(positions) < 23 // 3:
        raise ValueError("too few valid white-matter peaks")
    anchor = 14 - 4
    upper, mid, lower = (intensities[anchor - 3], intensities[anchor], intensities[anchor + 3])
    if min(upper, lower) <= mid <= max(upper, lower):
        pass
    elif min(mid, lower) <= upper <= max(mid, lower):
        anchor -= 3
    else:
        anchor += 3
    keep = np.ones(len(positions), dtype=bool)
    previous = anchor
    for index in range(anchor + 1, len(positions)):
        gradient = abs((intensities[index] - intensities[previous]) /
                       (positions[index] - positions[index - 1]))
        keep[index] = gradient <= 1.0 and index - previous <= 2
        if keep[index]:
            previous = index
    previous = anchor
    for index in range(anchor - 1, -1, -1):
        gradient = abs((intensities[previous] - intensities[index]) /
                       (positions[index + 1] - positions[index]))
        keep[index] = gradient <= 1.0 and previous - index <= 2
        if keep[index]:
            previous = index
    return np.asarray(positions, dtype=np.float32)[keep], np.asarray(intensities, dtype=np.float32)[keep]


def _curve(shape: tuple[int, int, int], image: np.ndarray, origin: tuple[int, int, int],
           voxel_size_y: float) -> tuple[np.ndarray, dict]:
    x0, y0, z0 = origin
    window = _nint(10.0 / voxel_size_y)
    step = _nint(window * 0.5)
    positions, intensities = [], []
    for index in range(23):
        y = y0 - step * 14 + index * step
        x, z = x0 - 60, z0 - 84  # dx=120, dz=120+24, z offset=24
        region = image[max(0, x):min(shape[0], x + 120),
                       max(0, y):min(shape[1], y + window),
                       max(0, z):min(shape[2], z + 144)]
        if not region.size:
            continue
        low, high = int(region.min()), int(region.max())
        bins = np.arange(low, high + 1, dtype=np.float32)
        counts = np.bincount(region.ravel(), minlength=high + 1)[low:high + 1].astype(np.float32)
        counts[(bins <= 30) | (bins >= 225)] = 0
        peak = _last_peak(bins, _smooth_histogram(counts))
        if peak >= 0:
            positions.append(float(y + window / 2.0))
            intensities.append(float(bins[peak]))
    raw_positions, raw_intensities = positions.copy(), intensities.copy()
    knots, peaks = _filter_peaks(positions, intensities)
    if len(knots) < 2:
        raise ValueError("1D spline needs at least two peaks")
    factors = np.float32(110) / peaks
    spline = CubicSpline(knots.astype(np.float64), factors.astype(np.float64), bc_type=((1, 0.0), (1, 0.0)))
    axes = np.arange(shape[1], dtype=np.float64)
    scales = np.asarray(spline(np.clip(axes, knots[0], knots[-1])), dtype=np.float32)
    return scales, {"origin": list(origin), "window": window,
                    "window_step": step, "npeaks": len(knots),
                    "raw_peak_positions": raw_positions,
                    "raw_peak_intensities": raw_intensities,
                    "knots": knots.tolist(), "peak_intensities": peaks.tolist(),
                    "scale_min": float(scales.min()), "scale_max": float(scales.max())}


def normalize_1d(image: torch.Tensor, affine: np.ndarray, xfm_path: Path) -> tuple[torch.Tensor, dict]:
    if image.ndim != 3 or image.dtype != torch.uint8:
        raise ValueError("expected a 3D uint8 MGH volume")
    origin = _talairach_origin(affine, xfm_path)
    scales, details = _curve(tuple(image.shape), image.cpu().numpy(), origin,
                             float(np.linalg.norm(affine[:3, 1])))
    factors = torch.as_tensor(scales, device=image.device).reshape(1, -1, 1)
    result = torch.floor(image.float() * factors + 0.5).clamp(0, 255).to(torch.uint8)
    return result, details


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--xfm", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    volume = nib.load(str(args.input))
    data = torch.as_tensor(np.asarray(volume.dataobj).copy(), device=args.device)
    result, details = normalize_1d(data, volume.affine, args.xfm)
    save_same_dtype_mgh(args.input, args.output, result.cpu().numpy())
    print(json.dumps(details))


if __name__ == "__main__":
    main()
