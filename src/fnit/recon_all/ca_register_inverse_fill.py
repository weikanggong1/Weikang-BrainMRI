"""Voronoi fill for the fixed inverse-GCAM coordinate fields."""

from __future__ import annotations

import numpy as np
from numba import njit
from scipy.ndimage import binary_dilation


@njit(cache=True)
def _fill_front(values: np.ndarray, marked: np.ndarray, front: np.ndarray) -> None:
    width, height, depth = values.shape
    for z in range(depth):
        for y in range(height):
            for x in range(width):
                if not front[x, y, z]:
                    continue
                mean = np.float32(0)
                neighbors = 0
                for dz in range(-1, 2):
                    zi = min(max(z + dz, 0), depth - 1)
                    for dy in range(-1, 2):
                        yi = min(max(y + dy, 0), height - 1)
                        for dx in range(-1, 2):
                            xi = min(max(x + dx, 0), width - 1)
                            if marked[xi, yi, zi]:
                                neighbors += 1
                                mean += values[xi, yi, zi]
                if neighbors:
                    values[x, y, z] = mean / np.float32(neighbors)


def voronoi_fill(control_values: np.ndarray, control_mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Expand marked coordinates in Chebyshev shells, matching native traversal."""
    marked = np.asarray(control_mask, dtype=bool).copy()
    values = np.where(marked, np.asarray(control_values, dtype=np.float32), 0).astype(np.float32)
    frontier = marked.copy()
    neighborhood = np.ones((3, 3, 3), dtype=bool)
    iterations = 0
    while not np.all(marked):
        next_front = binary_dilation(frontier, structure=neighborhood) & ~marked
        if not np.any(next_front):
            break
        _fill_front(values, marked, next_front)
        marked |= next_front
        frontier = next_front
        iterations += 1
    return values, iterations


@njit(cache=True)
def _seed_soap(values: np.ndarray, control: np.ndarray, bounds: tuple[int, int, int, int, int, int]) -> float:
    width, height, depth = values.shape
    x1, x2, y1, y2, z1, z2 = bounds
    max_change = 0.0
    for z in range(max(0, z1 - 2), min(depth - 1, z2 + 2) + 1):
        for y in range(max(0, y1 - 2), min(height - 1, y2 + 2) + 1):
            for x in range(max(0, x1 - 2), min(width - 1, x2 + 2) + 1):
                if control[x, y, z]:
                    continue
                mean = np.float32(0)
                neighbors = 0
                for dz in range(-2, 3):
                    zi = min(max(z + dz, 0), depth - 1)
                    for dy in range(-2, 3):
                        yi = min(max(y + dy, 0), height - 1)
                        for dx in range(-2, 3):
                            xi = min(max(x + dx, 0), width - 1)
                            if control[xi, yi, zi]:
                                mean += values[xi, yi, zi]
                                neighbors += 1
                if neighbors:
                    old = values[x, y, z]
                    new = np.float32(mean / np.float32(neighbors))
                    max_change = max(max_change, abs(float(new) - float(old)))
                    values[x, y, z] = new
    return max_change


@njit(cache=True)
def _soap_iteration(
    values: np.ndarray,
    target: np.ndarray,
    control: np.ndarray,
    bounds: tuple[int, int, int, int, int, int],
    max_change: float,
) -> float:
    width, height, depth = values.shape
    x1, x2, y1, y2, z1, z2 = bounds
    for z in range(z1, z2 + 1):
        for y in range(y1, y2 + 1):
            for x in range(x1, x2 + 1):
                if control[x, y, z]:
                    continue
                mean = np.float32(0)
                for dz in range(-1, 2):
                    zi = min(max(z + dz, 0), depth - 1)
                    for dy in range(-1, 2):
                        yi = min(max(y + dy, 0), height - 1)
                        for dx in range(-1, 2):
                            xi = min(max(x + dx, 0), width - 1)
                            mean += values[xi, yi, zi]
                old = target[x, y, z]
                max_change = max(max_change, abs(float(mean) / 27.0 - float(old)))
                target[x, y, z] = np.float32(mean / np.float32(27.0))
    return max_change


def soap_bubble_float(
    voronoi_values: np.ndarray, control_mask: np.ndarray, niter: int = 50, min_change: float = 1.0
) -> tuple[np.ndarray, int]:
    """Run FreeSurfer's fixed float-field control seeding and 3³ smoothing."""
    values = np.asarray(voronoi_values, dtype=np.float32).copy()
    control = np.asarray(control_mask, dtype=bool)
    target = values.copy()
    coords = np.nonzero(control)
    bounds = (
        int(coords[0].min()), int(coords[0].max()),
        int(coords[1].min()), int(coords[1].max()),
        int(coords[2].min()), int(coords[2].max()),
    )
    max_change = _seed_soap(values, control, bounds)
    width, height, depth = values.shape
    for iteration in range(niter):
        max_change = _soap_iteration(values, target, control, bounds, max_change)
        np.copyto(values, target)
        x1, x2, y1, y2, z1, z2 = bounds
        bounds = (
            max(x1 - 1, 0), min(x2 + 1, width - 1),
            max(y1 - 1, 0), min(y2 + 1, height - 1),
            max(z1 - 1, 0), min(z2 + 1, depth - 1),
        )
        if max_change < min_change:
            return values, iteration + 1
        max_change = 0.0
    return values, niter
