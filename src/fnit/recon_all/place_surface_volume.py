"""MRI intensity preparation for the T1 white/pial placement calls."""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def prepare_placement_volume(
    brain: np.ndarray,
    wm: np.ndarray,
    *,
    surface: str,
    mid_gray: float,
    restore_255: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the CBV input MRI and bright-label mask from raw MRI volumes."""
    if surface not in ("white", "pial"):
        raise ValueError("surface must be white or pial")
    source = np.asarray(brain, dtype=np.uint8)
    wm_mask = np.asarray(wm, dtype=np.uint8)
    if source.shape != wm_mask.shape or source.ndim != 3:
        raise ValueError("brain and wm must be matching 3D volumes")

    volume = source.copy()
    volume[(wm_mask >= 5) & (volume > 110)] = 110
    neighborhood = np.ones((3, 3, 3), dtype=np.uint8)
    wm_neighbors = ndimage.convolve((wm_mask >= 5).astype(np.uint8), neighborhood, mode="nearest")
    seed = (wm_mask < 5) & (volume > 125) & (wm_neighbors < 14)
    closed = ndimage.minimum_filter(
        ndimage.maximum_filter(seed.astype(np.uint8), size=3, mode="nearest"),
        size=3, mode="nearest",
    ).astype(bool)
    expanded = closed | (ndimage.maximum_filter(closed, size=3, mode="nearest") & (volume >= 100))
    labels = np.zeros(volume.shape, dtype=np.uint8)
    labels[expanded & ~closed] = 100
    labels[closed] = 130
    for _ in range(3):
        labels[(volume == 0) & ndimage.maximum_filter(labels == 130, size=3, mode="nearest")] = 130

    if surface == "white":
        volume[(labels == 100) | (labels == 130)] = 0
        if restore_255:
            volume[source == 255] = 110
        return volume, labels

    volume[labels == 100] = np.uint8(np.floor(mid_gray + 0.5))
    volume[labels == 130] = 255
    return volume, labels
