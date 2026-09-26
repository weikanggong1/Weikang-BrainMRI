"""One-ring gradient smoothing in FreeSurfer's standard sphere integration."""

from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(parallel=True, cache=True)
def average_standard_gradient(gradient: np.ndarray, offsets: np.ndarray,
                              neighbors: np.ndarray, rounds: int) -> np.ndarray:
    """Apply source-order float32 self-and-neighbor averages for each round."""
    current = gradient.copy()
    next_value = np.empty_like(current)
    for _ in range(rounds):
        for vertex in prange(len(current)):
            dx, dy, dz = current[vertex]
            begin, end = offsets[vertex], offsets[vertex + 1]
            for index in range(begin, end):
                neighbor = neighbors[index]
                dx = np.float32(dx + current[neighbor, 0])
                dy = np.float32(dy + current[neighbor, 1])
                dz = np.float32(dz + current[neighbor, 2])
            inverse = np.float32(1.0) / np.float32(end - begin + 1)
            next_value[vertex, 0] = np.float32(dx * inverse)
            next_value[vertex, 1] = np.float32(dy * inverse)
            next_value[vertex, 2] = np.float32(dz * inverse)
        current, next_value = next_value, current
    return current
