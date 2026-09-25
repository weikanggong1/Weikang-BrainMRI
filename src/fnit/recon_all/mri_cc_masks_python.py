"""FreeSurfer 8.2 ``mri_cc`` callosum mask operations (d932c45).

These functions are independent verification units; the final ``mri_cc``
stage is not registered until its labelled volume matches the native output.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import ndimage


_FULL_3D = np.ones((3, 3, 3), dtype=bool)
_FULL_2D = np.ones((3, 3), dtype=bool)
_CONNECTED_3D = ndimage.generate_binary_structure(3, 1)
_CONNECTED_2D = ndimage.generate_binary_structure(2, 1)


def callosum_seed(aseg_xformed: np.ndarray, *, keep_steps: bool = False) -> tuple[np.ndarray, int, dict[str, np.ndarray]]:
    """Return the mask after ``s4`` in ``find_cc_with_aseg``."""
    a = aseg_xformed
    left = a == 2
    right = a == 41
    left_x = np.where(left)[0]
    right_x = np.where(right)[0]
    if len(left_x) == 0 or len(right_x) == 0:
        raise ValueError("no bilateral white matter after resampling")
    xmin = min(int(left_x.min()), int(right_x.max())) - 1
    xmax = max(int(left_x.min()), int(right_x.max())) + 1
    s = np.zeros(a.shape, dtype=np.int32)
    near_left = ndimage.maximum_filter(left, size=3, mode="nearest")
    near_right = ndimage.maximum_filter(right, size=3, mode="nearest")
    for x in range(xmin, xmax + 1):
        s[x][(left[x] & near_right[x]) | (right[x] & near_left[x])] = 128
    steps = {"s1": s.copy()} if keep_steps else {}
    best = max(range(xmin, xmax + 1), key=lambda x: np.count_nonzero(s[x]))
    s[:best] = 0
    s[best + 1:] = 0
    wm = left | right
    while True:
        count = ndimage.convolve((s[best] == 128).astype(np.uint8),
                                 np.ones((3, 3), dtype=np.uint8), mode="nearest")
        add = (s[best] == 0) & wm[best] & (count > 1)
        if not np.any(add):
            break
        s[best][add] = 128
    if keep_steps:
        steps["s2"] = s.copy()
    for offset in (1, 2):
        adjacent = ndimage.maximum_filter(s == 128, size=3, mode="nearest")
        for x in (best + offset, best - offset):
            s[x][wm[x] & adjacent[x]] = 128
    if keep_steps:
        steps["s3"] = s.copy()
    dilated = ndimage.maximum_filter(s, size=9, mode="nearest")
    components, _ = ndimage.label(dilated > 0, structure=_CONNECTED_3D)
    sizes = np.bincount(components.ravel())
    sizes[0] = 0
    s[components != int(np.argmax(sizes))] = 0
    if keep_steps:
        steps["s4"] = s.copy()
    return s, best, steps


def remove_fornix_slice(slice_z_y: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Return edited sagittal mask, pre-shaving diagnostic, fornix minimum."""
    raw = np.asarray(slice_z_y)
    coords = np.argwhere(raw > 0)
    if len(coords) == 0:
        raise ValueError("empty callosum slice")
    xmin, xmax = int(coords[:, 0].min()), int(coords[:, 0].max())
    ymax = int(coords[:, 1].max())
    edited = raw.copy()
    for x in range(xmin, xmax + 1):
        edges = 0
        last = 0
        for y in range(raw.shape[1]):
            val = int(raw[x, y])
            if val > 0 and last == 0:
                edges += 1
            last = val
            if val and edges > 1:
                edited[x, y] = 32
    tmp = np.zeros_like(raw)
    tmp[xmax - 1:xmax + 1][raw[xmax - 1:xmax + 1] > 0] = 128
    end = xmax - 2 * ((xmax - xmin) // 3)
    for x in range(xmax - 2, end - 1, -1):
        while True:
            changed = False
            for y in range(raw.shape[1]):
                if edited[x, y] <= 0:
                    continue
                if np.any(tmp[max(0, x-1):x+2, max(0, y-1):y+2] == 128):
                    changed |= tmp[x, y] != 128
                    tmp[x, y] = 128
            if not changed:
                break
    tmp[edited == 128] = 128
    edited[tmp == 128] = 128
    for x in range(xmin, xmax + 1):
        for y in range(raw.shape[1]):
            if edited[x, y] == 32 and not np.any(tmp[max(0, x-3):x+4, y] == 128):
                edited[x, y] = 96
    for x in range(xmin, xmax + 1):
        for y in range(raw.shape[1]):
            if edited[x, y] != 32:
                continue
            near = edited[max(0, x-1):x+2, max(0, y-1):y+2]
            edited[x, y] = 96 if np.any(near == 96) else 128
    diagnostic = edited.copy()
    for x in range(xmin, xmax + 1):
        for y in range(ymax - 10, ymax + 1):
            if edited[x, y] == 96 and np.all(edited[x-3:x, y] == 128):
                edited[x, y] = 128
    thicknesses = []
    min_fornix = raw.shape[0] - 1
    for x in range(xmin, xmax + 1):
        hit = np.flatnonzero((raw[x] == 128) & (edited[x] == 96))
        if len(hit):
            min_fornix = min(min_fornix, x)
            on = np.flatnonzero(edited[x, :hit[0] + 1] == 128)
            first = int(on[-1]) if len(on) else -1
            last = int(on[0]) if len(on) > 1 else -1
            thicknesses.append(first - last + 1)
    if thicknesses:
        mean = float(np.mean(thicknesses))
        std = math.sqrt(float(np.mean(np.square(thicknesses))) - mean * mean)
        maximum = math.ceil(mean + 3 * std)
        for x in range(min_fornix - 3, min_fornix + 1):
            on = np.flatnonzero(edited[x] == 128)
            if len(on) < 2:
                continue
            first, last = int(on[-1]), int(on[0])
            off = np.flatnonzero(edited[x, last:first] != 128)
            first_off = last + int(off[-1]) if len(off) else -1
            if first - last + 1 >= maximum:
                start = first_off if first_off > last else last + maximum - 1
                edited[x, max(0, start):][edited[x, max(0, start):] == 128] = 96
    edited[edited == 96] = 0
    components, _ = ndimage.label(edited > 0, structure=_CONNECTED_2D)
    sizes = np.bincount(components.ravel())
    sizes[0] = 0
    result = np.where(components == int(np.argmax(sizes)), raw, 0)
    return result, diagnostic, min_fornix
