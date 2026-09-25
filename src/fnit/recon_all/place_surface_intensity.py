"""First pial placement intensity gradient with native float32 field arithmetic."""

from __future__ import annotations

import numpy as np
from numba import njit

from .place_surface_border import _sample, _voxel


@njit(cache=True)
def _gradient(
    volume: np.ndarray, xyz: np.ndarray, normals: np.ndarray,
    ripped: np.ndarray, values: np.ndarray, sigmas: np.ndarray,
    affine: np.ndarray, voxel_step: float, weight: float, sigma_global: float,
) -> np.ndarray:
    result = np.zeros_like(xyz)
    for vertex in range(len(xyz)):
        if ripped[vertex] or values[vertex] < 0:
            continue
        x, y, z = float(xyz[vertex, 0]), float(xyz[vertex, 1]), float(xyz[vertex, 2])
        nx, ny, nz = float(normals[vertex, 0]), float(normals[vertex, 1]), float(normals[vertex, 2])
        xw, yw, zw = _voxel(affine, x, y, z)
        current = _sample(volume, xw, yw, zw)
        sigma = float(sigmas[vertex])
        if abs(sigma) < 1e-10:
            sigma = sigma_global
        if abs(sigma) < 1e-10:
            sigma = 0.25
        step = min(sigma / 2.0, voxel_step)
        outside = inside = total_outside = total_inside = 0.0
        distance = step
        while distance <= 2.0 * sigma:
            kernel = np.exp(-distance * distance / (2.0 * sigma * sigma))
            xo, yo, zo = _voxel(affine, x + distance * nx, y + distance * ny, z + distance * nz)
            xi, yi, zi = _voxel(affine, x - distance * nx, y - distance * ny, z - distance * nz)
            outside += kernel * _sample(volume, xo, yo, zo)
            inside += kernel * _sample(volume, xi, yi, zi)
            total_outside += kernel
            total_inside += kernel
            distance += step
        if total_outside:
            outside /= total_outside
        if total_inside:
            inside /= total_inside
        error = float(values[vertex]) - current
        error = min(max(error, -5.0), 5.0)
        slope = (outside - inside) / 2.0
        sign = slope / abs(slope) if abs(slope) >= 1e-10 else -1.0
        displacement = weight * error * sign
        result[vertex, 0] = np.float32(nx * displacement)
        result[vertex, 1] = np.float32(ny * displacement)
        result[vertex, 2] = np.float32(nz * displacement)
    return result


def intensity_gradient(
    placement_volume: np.ndarray, vertices: np.ndarray, normals: np.ndarray,
    ripped: np.ndarray, target_values: np.ndarray, vertex_sigma: np.ndarray,
    surface_ras_to_voxel: np.ndarray, voxel_sizes: np.ndarray,
    *, weight: float = 0.2, sigma_global: float = 2.0,
) -> np.ndarray:
    """Return the fixed pial first-step intensity gradient."""
    return _gradient(
        np.asarray(placement_volume, dtype=np.uint8),
        np.asarray(vertices, dtype=np.float32),
        np.asarray(normals, dtype=np.float32),
        np.asarray(ripped, dtype=np.bool_),
        np.asarray(target_values, dtype=np.float32),
        np.asarray(vertex_sigma, dtype=np.float32),
        np.asarray(surface_ras_to_voxel, dtype=np.float32),
        float(np.min(np.asarray(voxel_sizes, dtype=np.float32))) * 0.5,
        float(np.float32(weight)),
        float(sigma_global),
    )
