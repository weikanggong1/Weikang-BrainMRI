"""First-step normal and tangential spring gradients for surface placement."""

from __future__ import annotations

import numpy as np
from numba import njit

from .place_surface_smoothing import _ordered_neighbors


@njit(cache=True)
def _spring(
    xyz: np.ndarray, normals: np.ndarray, ripped: np.ndarray,
    border: np.ndarray, negative: np.ndarray, neighbors: np.ndarray,
    valid: np.ndarray, weight: float, tangent: bool,
) -> np.ndarray:
    result = np.zeros_like(xyz)
    for vertex in range(len(xyz)):
        if ripped[vertex] or (tangent and border[vertex] and not negative[vertex]):
            continue
        x, y, z = xyz[vertex, 0], xyz[vertex, 1], xyz[vertex, 2]
        sx = sy = sz = np.float32(0)
        count = 0
        for rank in range(neighbors.shape[1]):
            if not valid[vertex, rank]:
                break
            other = neighbors[vertex, rank]
            if ripped[other]:
                continue
            sx = np.float32(sx + np.float32(xyz[other, 0] - x))
            sy = np.float32(sy + np.float32(xyz[other, 1] - y))
            sz = np.float32(sz + np.float32(xyz[other, 2] - z))
            count += 1
        if count:
            divisor = np.float32(count)
            sx, sy, sz = np.float32(sx / divisor), np.float32(sy / divisor), np.float32(sz / divisor)
        nx, ny, nz = normals[vertex, 0], normals[vertex, 1], normals[vertex, 2]
        component = np.float32(np.float32(sx * nx + sy * ny) + sz * nz)
        if tangent:
            result[vertex, 0] = np.float32(weight * float(np.float32(sx - np.float32(component * nx))))
            result[vertex, 1] = np.float32(weight * float(np.float32(sy - np.float32(component * ny))))
            result[vertex, 2] = np.float32(weight * float(np.float32(sz - np.float32(component * nz))))
        else:
            result[vertex, 0] = np.float32(weight * float(component) * float(nx))
            result[vertex, 1] = np.float32(weight * float(component) * float(ny))
            result[vertex, 2] = np.float32(weight * float(component) * float(nz))
    return result


def spring_gradient(
    vertices: np.ndarray, normals: np.ndarray, faces: np.ndarray,
    ripped: np.ndarray, *, weight: float, direction: str,
    border: np.ndarray | None = None, negative: np.ndarray | None = None,
    ordered_neighbors: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Return one source-order normal or tangential spring contribution."""
    if direction not in ("normal", "tangent"):
        raise ValueError("direction must be normal or tangent")
    xyz = np.asarray(vertices, dtype=np.float32)
    if ordered_neighbors is None:
        neighbors, valid, _ = _ordered_neighbors(np.asarray(faces, dtype=np.int32), len(xyz))
    else:
        neighbors, valid = ordered_neighbors
    flags = np.zeros(len(xyz), dtype=np.bool_)
    return _spring(
        xyz, np.asarray(normals, dtype=np.float32), np.asarray(ripped, dtype=np.bool_),
        flags if border is None else np.asarray(border, dtype=np.bool_),
        flags if negative is None else np.asarray(negative, dtype=np.bool_),
        neighbors, valid, float(np.float32(weight)), direction == "tangent",
    )
