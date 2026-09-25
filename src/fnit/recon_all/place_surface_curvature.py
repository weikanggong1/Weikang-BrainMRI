"""First pial optimizer tangent basis and quadratic curvature inputs."""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def _cross(a0: np.float32, a1: np.float32, a2: np.float32, b0: np.float32, b1: np.float32, b2: np.float32) -> tuple:
    return (
        np.float32(a1 * b2 - a2 * b1),
        np.float32(a2 * b0 - a0 * b2),
        np.float32(a0 * b1 - a1 * b0),
    )


@njit(cache=True)
def _normalize(x: np.float32, y: np.float32, z: np.float32) -> tuple:
    length = np.float32(np.sqrt(np.float32(np.float32(x * x + y * y) + z * z)))
    if length < 1e-5:
        length = np.float32(1.0)
    else:
        length = np.float32(np.float32(1.0) / length)
    return np.float32(x * length), np.float32(y * length), np.float32(z * length)


@njit(cache=True)
def _tangent_basis(normals: np.ndarray) -> np.ndarray:
    basis = np.zeros((len(normals), 6), dtype=np.float32)
    for vertex in range(len(normals)):
        nx, ny, nz = normals[vertex]
        e1x, e1y, e1z = _cross(nx, ny, nz, ny, nz, nx)
        length = np.sqrt(np.float32(np.float32(e1x * e1x + e1y * e1y) + e1z * e1z))
        if length < 0.001:
            e1x, e1y, e1z = _cross(nx, ny, nz, ny, np.float32(-nz), nx)
        e2x, e2y, e2z = _cross(nx, ny, nz, e1x, e1y, e1z)
        basis[vertex, :3] = _normalize(e1x, e1y, e1z)
        basis[vertex, 3:] = _normalize(e2x, e2y, e2z)
    return basis


def tangent_basis(normals: np.ndarray) -> np.ndarray:
    """Return source-order float32 orthonormal tangent basis."""
    return _tangent_basis(np.asarray(normals, dtype=np.float32))


def two_ring_neighbors(
    faces: np.ndarray, vertex_count: int, *,
    ordered_neighbors: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ordered two-ring neighbors from original face topology."""
    from .place_surface_smoothing import _ordered_neighbors

    if ordered_neighbors is None:
        immediate, valid, _ = _ordered_neighbors(np.asarray(faces, dtype=np.int32), vertex_count)
    else:
        immediate, valid = ordered_neighbors
    offsets = np.zeros(vertex_count + 1, dtype=np.int32)
    flat: list[int] = []
    for vertex in range(vertex_count):
        first = immediate[vertex, valid[vertex]].tolist()
        seen = {vertex, *first}
        flat.extend(first)
        for neighbor in first:
            for candidate in immediate[neighbor, valid[neighbor]]:
                other = int(candidate)
                if other not in seen:
                    seen.add(other)
                    flat.append(other)
        offsets[vertex + 1] = len(flat)
    return offsets, np.asarray(flat, dtype=np.int32)


@njit(cache=True)
def _inverse5_qr(gram: np.ndarray) -> np.ndarray:
    """VNL float32 QR inverse, including f2c SNRM2's double sqrt."""
    work = gram.copy()
    aux = np.zeros(5, dtype=np.float32)
    for col in range(4):
        scale = np.float32(0)
        ssq = np.float32(1)
        for row in range(col, 5):
            value = work[row, col]
            if value == 0:
                continue
            absolute = np.float32(abs(value))
            if scale < absolute:
                ratio = np.float32(scale / absolute)
                ssq = np.float32(np.float32(ssq * np.float32(ratio * ratio)) + np.float32(1))
                scale = absolute
            else:
                ratio = np.float32(absolute / scale)
                ssq = np.float32(ssq + np.float32(ratio * ratio))
        norm = np.float32(np.float64(scale) * np.sqrt(np.float64(ssq)))
        if norm == 0:
            continue
        if work[col, col] < 0:
            norm = np.float32(-norm)
        reciprocal = np.float32(np.float32(1.0) / norm)
        for row in range(col, 5):
            work[row, col] = np.float32(work[row, col] * reciprocal)
        work[col, col] = np.float32(work[col, col] + np.float32(1))
        for other in range(col + 1, 5):
            dot = np.float32(0)
            for row in range(col, 5):
                dot = np.float32(dot + np.float32(work[row, col] * work[row, other]))
            factor = np.float32(-dot / work[col, col])
            for row in range(col, 5):
                work[row, other] = np.float32(work[row, other] + np.float32(factor * work[row, col]))
        aux[col] = work[col, col]
        work[col, col] = np.float32(-norm)

    inverse = np.zeros((5, 5), dtype=np.float32)
    for rhs in range(5):
        result = np.zeros(5, dtype=np.float32)
        result[rhs] = np.float32(1)
        for col in range(4):
            diagonal = work[col, col]
            work[col, col] = aux[col]
            dot = np.float32(0)
            for row in range(col, 5):
                dot = np.float32(dot + np.float32(work[row, col] * result[row]))
            factor = np.float32(-dot / work[col, col])
            for row in range(col, 5):
                result[row] = np.float32(result[row] + np.float32(factor * work[row, col]))
            work[col, col] = diagonal
        for col in range(4, -1, -1):
            result[col] = np.float32(result[col] / work[col, col])
            factor = np.float32(-result[col])
            for row in range(col):
                result[row] = np.float32(result[row] + np.float32(factor * work[row, col]))
        inverse[:, rhs] = result
    return inverse


@njit(cache=True)
def _quadratic_curvature(
    xyz: np.ndarray, normals: np.ndarray, basis: np.ndarray,
    ripped: np.ndarray, offsets: np.ndarray, candidates: np.ndarray,
) -> np.ndarray:
    scalar = np.zeros(len(xyz), dtype=np.float32)
    for vertex in range(len(xyz)):
        if ripped[vertex]:
            continue
        count = offsets[vertex + 1] - offsets[vertex]
        if count < 5:
            continue
        design = np.zeros((count, 5), dtype=np.float32)
        height = np.zeros(count, dtype=np.float32)
        for row in range(count):
            other = candidates[offsets[vertex] + row]
            ex = np.float32(xyz[other, 0] - xyz[vertex, 0])
            ey = np.float32(xyz[other, 1] - xyz[vertex, 1])
            ez = np.float32(xyz[other, 2] - xyz[vertex, 2])
            height[row] = np.float32(np.float32(ex * normals[vertex, 0] + ey * normals[vertex, 1]) + ez * normals[vertex, 2])
            u = np.float32(np.float32(ex * basis[vertex, 0] + ey * basis[vertex, 1]) + ez * basis[vertex, 2])
            v = np.float32(np.float32(ex * basis[vertex, 3] + ey * basis[vertex, 4]) + ez * basis[vertex, 5])
            design[row, 0] = np.float32(u * u)
            design[row, 1] = np.float32(v * v)
            design[row, 2] = u
            design[row, 3] = v
            design[row, 4] = np.float32(1)
        gram = np.zeros((5, 5), dtype=np.float32)
        for row in range(5):
            for col in range(5):
                for item in range(count):
                    gram[row, col] = np.float32(gram[row, col] + np.float32(design[item, row] * design[item, col]))
        inverse = _inverse5_qr(gram)
        for col in range(count):
            pseudo = np.float32(0)
            for item in range(5):
                pseudo = np.float32(pseudo + np.float32(inverse[4, item] * design[col, item]))
            scalar[vertex] = np.float32(scalar[vertex] + np.float32(pseudo * height[col]))
    return scalar


def quadratic_curvature(
    vertices: np.ndarray, normals: np.ndarray, basis: np.ndarray,
    ripped: np.ndarray, offsets: np.ndarray, candidates: np.ndarray,
) -> np.ndarray:
    """Fit the pial quadratic-curvature scalar with float32 normal equations.

    The 5x5 inverse follows VNL's float32 LINPACK QR operation order.
    """
    return _quadratic_curvature(
        np.asarray(vertices, dtype=np.float32), np.asarray(normals, dtype=np.float32),
        np.asarray(basis, dtype=np.float32), np.asarray(ripped, dtype=np.bool_),
        np.asarray(offsets, dtype=np.int32), np.asarray(candidates, dtype=np.int32),
    )
