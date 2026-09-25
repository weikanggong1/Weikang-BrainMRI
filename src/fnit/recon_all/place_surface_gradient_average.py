"""Native-order signed averaging of surface-placement gradients."""

from __future__ import annotations

import numpy as np
from numba import njit

from .place_surface_smoothing import _ordered_neighbors


@njit(cache=True)
def _average(gradient: np.ndarray, ripped: np.ndarray, neighbors: np.ndarray, valid: np.ndarray, iterations: int) -> np.ndarray:
    current = gradient.copy()
    for _ in range(iterations):
        following = current.copy()
        for vertex in range(len(current)):
            if ripped[vertex]:
                continue
            vx, vy, vz = current[vertex]
            sx, sy, sz = np.float64(vx), np.float64(vy), np.float64(vz)
            count = 1
            for rank in range(neighbors.shape[1]):
                if not valid[vertex, rank]:
                    break
                other = neighbors[vertex, rank]
                if ripped[other]:
                    continue
                nx, ny, nz = current[other]
                dot = np.float32(np.float32(vx * nx + vy * ny) + vz * nz)
                if dot < 0:
                    continue
                sx += np.float64(nx)
                sy += np.float64(ny)
                sz += np.float64(nz)
                count += 1
            following[vertex, 0] = np.float32(sx / count)
            following[vertex, 1] = np.float32(sy / count)
            following[vertex, 2] = np.float32(sz / count)
        current = following
    return current


def average_signed_gradients(
    gradient: np.ndarray, faces: np.ndarray, ripped: np.ndarray, iterations: int,
    *, ordered_neighbors: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Match fixed pial signed averaging using source-order 1-ring neighbors."""
    if iterations < 0:
        raise ValueError("iterations must be nonnegative")
    values = np.asarray(gradient, dtype=np.float32)
    if iterations == 0:
        return values.copy()
    if ordered_neighbors is None:
        neighbors, valid, _ = _ordered_neighbors(np.asarray(faces, dtype=np.int32), len(values))
    else:
        neighbors, valid = ordered_neighbors
    return _average(values, np.asarray(ripped, dtype=np.bool_), neighbors, valid, iterations)
