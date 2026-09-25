"""Initial inputs for the ``mri_normalize -aseg -mask`` branch."""

from __future__ import annotations

import numpy as np
from numba import njit

from .normalize_gaussian_source import smooth_bias
from .normalize_tissue_peaks import _smooth
from .normalize_voronoi_source import voronoi_fill


def prepare_aseg_source(
    norm: np.ndarray, brainmask: np.ndarray, aseg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the masked float32 source and the initial 2/41 WM controls.

    Arrays must already share a voxel grid. The frozen ``fs_sub01`` inputs do;
    native ``MRIresample`` takes an identity path for its type-only mismatch.
    """
    if norm.shape != brainmask.shape or norm.shape != aseg.shape:
        raise ValueError("norm, brainmask, and aseg must share a voxel grid")
    masked = np.where(brainmask == 0, 0, norm).astype(np.float32)
    controls = np.where((aseg == 2) | (aseg == 41), aseg, 0).astype(np.int32)
    return masked, controls


@njit(cache=True)
def _filter_ridge(source: np.ndarray, controls: np.ndarray, threshold: int) -> np.ndarray:
    removed = np.zeros(controls.shape, np.uint8)
    width, height, depth = controls.shape
    for x in range(width):
        for y in range(height):
            for z in range(depth):
                if controls[x, y, z] == 0:
                    continue
                maximum = 0.0
                for xx in range(max(0, x - 5), min(width, x + 6)):
                    for yy in range(max(0, y - 5), min(height, y + 6)):
                        for zz in range(max(0, z - 5), min(depth, z + 6)):
                            if controls[xx, yy, zz] and source[xx, yy, zz] > maximum:
                                maximum = source[xx, yy, zz]
                value = source[x, y, z]
                if value + 15 < maximum and value < threshold:
                    controls[x, y, z] = 0
                    removed[x, y, z] = 128
    return removed


def filter_aseg_ridge(source: np.ndarray, ridge: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Apply native scan-order WM outlier removal to a medial-ridge mask.

    The distance transform and ridge selection are a separate preceding stage.
    """
    source = np.ascontiguousarray(source, dtype=np.float32)
    controls = np.ascontiguousarray(ridge != 0, dtype=np.uint8)
    if source.shape != controls.shape:
        raise ValueError("source and ridge must share a voxel grid")
    values = source[controls != 0]
    minimum, maximum = np.float32(values.min()), np.float32(values.max())
    step = np.float32((maximum - minimum) / np.float32(255))
    bins = minimum + step * np.arange(256, dtype=np.float32)
    index = np.floor((values - minimum) / step + np.float32(0.5)).astype(np.int32)
    counts = np.bincount(np.clip(index, 0, 255), minlength=256).astype(np.float32)
    peak = int(bins[int(np.argmax(_smooth(counts)[1:]) + 1)])
    # The active -mprage option sets intensity_below to 15.
    removed = _filter_ridge(source, controls, peak - 15)
    return controls, removed, peak


def apply_initial_aseg_bias(source: np.ndarray, controls: np.ndarray) -> np.ndarray:
    """Reproduce the aseg branch's initial Voronoi, sigma-8 bias, and correction."""
    source = np.asarray(source, dtype=np.float32)
    voronoi, _ = voronoi_fill(source, controls)
    bias, _ = smooth_bias(voronoi, source, np.zeros_like(controls), strict=True)
    return np.float32(source.astype(np.float64) * (110.0 / bias.astype(np.float64)))
