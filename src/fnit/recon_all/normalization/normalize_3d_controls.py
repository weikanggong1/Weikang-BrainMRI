"""Experimental 3D WM controls from FreeSurfer MRInormFindControlPoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

from .normalize_tissue_peaks import tissue_peaks


def _homogeneous(image: np.ndarray, width: int, low: float, high: float) -> np.ndarray:
    return (ndimage.minimum_filter(image, size=width, mode="nearest") >= low) & (
        ndimage.maximum_filter(image, size=width, mode="nearest") <= high)


def _candidate_region(eligible: np.ndarray) -> tuple[slice, slice, slice]:
    bounds = []
    for axis in range(3):
        populated = np.any(eligible, axis=tuple(i for i in range(3) if i != axis))
        locations = np.flatnonzero(populated)
        bounds.append(slice(int(locations[0]), int(locations[-1]) + 1))
    return tuple(bounds)


def _neighbor_sum(image: np.ndarray, control: np.ndarray, kernel: np.ndarray,
                  region: tuple[slice, slice, slice]) -> tuple[np.ndarray, np.ndarray]:
    # Only eligible voxels can be added; one halo voxel preserves their 3³ neighborhood.
    halo = tuple(slice(max(0, s.start - 1), min(n, s.stop + 1))
                 for s, n in zip(region, image.shape))
    core = tuple(slice(s.start - h.start, s.stop - h.start) for s, h in zip(region, halo))
    count = ndimage.convolve(control[halo].astype(np.int16), kernel, mode="nearest")[core]
    total = ndimage.convolve((image[halo] * control[halo]).astype(np.float32),
                             kernel, mode="nearest")[core]
    return count, total


def controls_3d(source: np.ndarray, wm_peak: float | None = None,
                gm_peak: float | None = None) -> tuple[np.ndarray, dict]:
    if source.ndim != 3:
        raise ValueError("expected a 3D float image")
    raw = np.asarray(source, dtype=np.float32)
    image = raw.astype(np.int16)  # InWindow assigns MRIgetVoxVal to int.
    control = np.zeros(image.shape, dtype=bool)
    details = {}

    # First anchors estimate the tissue peaks. Reproduction of that estimate
    # is intentionally separate; the source clears these anchors afterwards.
    control |= _homogeneous(image, 7, 87, 147)
    details["initial_7"] = int(control.sum())
    control |= _homogeneous(image, 5, 95, 135)
    details["initial_5"] = int(control.sum())

    if wm_peak is None or gm_peak is None:
        wm_peak, gm_peak, peaks = tissue_peaks(raw, control)
        details["tissue_peaks"] = peaks

    control[:] = False
    control |= _homogeneous(image, 7, 87, 148)
    details["adaptive_7"] = int(control.sum())
    below = np.floor((wm_peak - gm_peak) / 3.0)
    low_5 = max(110 - 1.5 * 15, 110 - below)
    control |= _homogeneous(image, 5, np.floor(110 - np.ceil(110 - low_5)), 135)
    details["adaptive_5"] = int(control.sum())

    adaptive = (wm_peak - gm_peak) / 4.0
    lower = 110 - adaptive
    upper = 135
    six = np.zeros((3, 3, 3), dtype=np.int16)
    six[0, 1, 1] = six[2, 1, 1] = 1
    six[1, 0, 1] = six[1, 2, 1] = 1
    six[1, 1, 0] = six[1, 1, 2] = 1
    neighborhood_ok = (ndimage.minimum_filter(raw, size=3, mode="nearest") > lower) & (
        ndimage.maximum_filter(raw, size=3, mode="nearest") < upper)
    eligible = (raw >= lower) & (raw <= upper) & neighborhood_ok
    region = _candidate_region(eligible)
    candidates, selected, values = eligible[region], control[region], raw[region]
    three_added = 0
    while True:
        count, total = _neighbor_sum(raw, control, six, region)
        mean = np.divide(total, count, out=np.zeros(count.shape, dtype=np.float32), where=count>0)
        additions = candidates & ~selected & (count > 0) & ((values >= 110) | (mean - values < adaptive / 2))
        added = int(additions.sum())
        if added == 0:
            break
        selected |= additions
        three_added += added
    details["three_added"] = three_added

    lower = max(lower, 110 - 15)
    six_ok = (ndimage.minimum_filter(raw, footprint=six.astype(bool), mode="nearest") >= lower) & (
        ndimage.maximum_filter(raw, footprint=six.astype(bool), mode="nearest") <= upper)
    eligible = (raw >= lower) & (raw <= upper) & six_ok
    region = _candidate_region(eligible)
    candidates, selected, values = eligible[region], control[region], raw[region]
    cube = np.ones((3, 3, 3), dtype=np.int16)
    six_added = 0
    while True:
        count, total = _neighbor_sum(raw, control, cube, region)
        mean = np.divide(total, count, out=np.zeros(count.shape, dtype=np.float32), where=count>0)
        additions = candidates & ~selected & (count >= 4) & ((values >= 110) | (mean - values < adaptive / 2))
        added = int(additions.sum())
        if added == 0:
            break
        selected |= additions
        six_added += added
    details["six_added"] = six_added
    details["before_outlier_removal"] = int(control.sum())

    # mriRemoveOutliers mutates in z/y/x scan order.
    for z, y, x in zip(*np.nonzero(control.transpose(2, 1, 0))):
        lo = (max(0, x - 1), max(0, y - 1), max(0, z - 1))
        hi = (min(image.shape[0], x + 2), min(image.shape[1], y + 2),
              min(image.shape[2], z + 2))
        if control[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].sum() - 1 < 2:
            control[x, y, z] = False
    details["after_outlier_removal"] = int(control.sum())
    return control.astype(np.uint8), details


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wm-peak", type=float)
    parser.add_argument("--gm-peak", type=float)
    args = parser.parse_args()
    image = nib.load(str(args.input))
    result, details = controls_3d(np.asarray(image.dataobj), args.wm_peak, args.gm_peak)
    nib.save(nib.MGHImage(result, image.affine), str(args.output))
    print(json.dumps(details))


if __name__ == "__main__":
    main()
