"""Unconstrained first surface-placement step used to isolate collisions."""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def _step(xyz: np.ndarray, gradient: np.ndarray, ripped: np.ndarray, dt: float, max_mm: float) -> tuple[np.ndarray, np.ndarray]:
    result = xyz.copy()
    offsets = np.zeros_like(xyz)
    for vertex in range(len(xyz)):
        if ripped[vertex]:
            continue
        dx = np.float32(dt * gradient[vertex, 0])
        dy = np.float32(dt * gradient[vertex, 1])
        dz = np.float32(dt * gradient[vertex, 2])
        magnitude_squared = np.float32(np.float32(dx * dx + dy * dy) + dz * dz)
        magnitude = np.sqrt(np.float64(magnitude_squared))
        if magnitude > max_mm:
            scale = max_mm / magnitude
            dx = np.float32(dx * scale)
            dy = np.float32(dy * scale)
            dz = np.float32(dz * scale)
        offsets[vertex, 0] = dx
        offsets[vertex, 1] = dy
        offsets[vertex, 2] = dz
        result[vertex, 0] = np.float32(xyz[vertex, 0] + dx)
        result[vertex, 1] = np.float32(xyz[vertex, 1] + dy)
        result[vertex, 2] = np.float32(xyz[vertex, 2] + dz)
    return result, offsets


def unconstrained_step(
    vertices: np.ndarray, gradient: np.ndarray, ripped: np.ndarray,
    *, dt: float = 0.5, max_mm: float = 0.3,
) -> np.ndarray:
    """Apply native momentum zero and max-distance clipping without collisions."""
    return unconstrained_step_with_offsets(
        vertices, gradient, ripped, dt=dt, max_mm=max_mm,
    )[0]


def unconstrained_step_with_offsets(
    vertices: np.ndarray, gradient: np.ndarray, ripped: np.ndarray,
    *, dt: float = 0.5, max_mm: float = 0.3,
) -> tuple[np.ndarray, np.ndarray]:
    """Return native-order clipped endpoints and unrounded displacement vectors."""
    return _step(
        np.asarray(vertices, dtype=np.float32), np.asarray(gradient, dtype=np.float32),
        np.asarray(ripped, dtype=np.bool_), float(np.float32(dt)), float(np.float32(max_mm)),
    )
