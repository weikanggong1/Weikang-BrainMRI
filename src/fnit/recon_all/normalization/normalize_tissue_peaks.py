"""Histogram tissue peaks used by FreeSurfer's 3D control point search."""

from __future__ import annotations

import numpy as np


def _histogram(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.float32]:
    values = np.asarray(values, dtype=np.float32)
    minimum, maximum = np.float32(values.min()), np.float32(values.max())
    nbins = int(np.floor(np.float32(maximum - minimum + 1) + np.float32(0.5)))
    step = np.float32((maximum - minimum) / np.float32(nbins - 1))
    bins = minimum + step * np.arange(nbins, dtype=np.float32)
    indices = np.floor((values - minimum) / step + np.float32(0.5)).astype(np.int32)
    counts = np.bincount(np.clip(indices.ravel(), 0, nbins - 1), minlength=nbins).astype(np.float32)
    return bins, counts, step


def _smooth(counts: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    length = int(np.floor(8 * sigma + 0.5)) + 1
    if length % 2 == 0:
        length += 1
    half = length // 2
    kernel = np.zeros(length, dtype=np.float32)
    norm = np.float32(0)
    for index, offset in enumerate(range(-half, half + 1)):
        absolute = abs(offset)
        if absolute <= 2 * sigma:
            value = np.float32(np.exp(-offset * offset / (2 * sigma * sigma)))
        elif absolute <= 4 * sigma:
            value = np.float32((4 - absolute / sigma) ** 4 / (16 * np.e ** 2))
        else:
            value = np.float32(0)
        kernel[index] = value
        norm = np.float32(norm + value)
    kernel /= norm
    result = np.zeros_like(counts)
    for b in range(len(counts)):
        total = np.float32(0)
        included = np.float32(0)
        for index, weight in enumerate(kernel):
            neighbor = b + index - half
            if 0 <= neighbor < len(counts):
                included = np.float32(included + weight)
                total = np.float32(total + np.float32(weight * counts[neighbor]))
        result[b] = np.float32(total / included)
    return result


def _previous_peak(counts: np.ndarray, start: int, half_width: int = 10) -> int:
    if start > len(counts) - 2:
        return start
    for b in range(start - 1, half_width - 1, -1):
        if not np.any(counts[b - half_width:b + half_width + 1] > counts[b]):
            return b
    return -1


def tissue_peaks(source: np.ndarray, initial_controls: np.ndarray) -> tuple[float, float, dict]:
    """Return WM/GM peaks and source histogram diagnostics for `find_tissue_intensities`."""
    source = np.asarray(source, dtype=np.float32)
    selected_bins, selected_counts, _ = _histogram(source[initial_controls > 0])
    selected_wm = float(selected_bins[int(np.argmax(selected_counts))])

    bins, counts, step = _histogram(source)
    counts[(bins >= 0) & (bins <= 5)] = 0
    smooth = _smooth(counts)
    lo, hi = int(selected_wm - 20), int(selected_wm + 20)
    wm_index = max(range(max(lo, 0), min(hi, len(bins) - 1) + 1), key=lambda b: smooth[b])
    wm_value = float(bins[wm_index])
    gm_index = _previous_peak(smooth, int(wm_index - 10 * step))
    if gm_index < 0:
        gm_index = int(np.searchsorted(bins, wm_value * 0.75, side="left"))
    if gm_index > wm_index * 0.8:
        gm_index = int(wm_index * 0.8)
    if bins[gm_index] < 40:
        gm_index = int(0.7 * wm_index)
    return wm_value, float(bins[gm_index]), {
        "selected_wm_peak": selected_wm,
        "wm_index": wm_index,
        "gm_index": gm_index,
        "wm_peak": wm_value,
        "gm_peak": float(bins[gm_index]),
        "bin_size": float(step),
    }
