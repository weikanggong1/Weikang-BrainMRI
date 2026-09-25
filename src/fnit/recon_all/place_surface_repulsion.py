"""First placement surface repulsion on the original vertex hash grid."""

from __future__ import annotations

import numpy as np
from numba import njit

from .place_surface_normals import _unit


@njit(cache=True)
def _original_normals(xyz: np.ndarray, faces: np.ndarray, ids: np.ndarray, corners: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    result = np.zeros_like(xyz)
    for vertex in range(len(xyz)):
        normal = np.zeros(3, dtype=np.float32)
        for entry in range(offsets[vertex], offsets[vertex + 1]):
            face = faces[ids[entry]]
            corner = corners[entry]
            previous = face[(corner + 2) % 3]
            following = face[(corner + 1) % 3]
            v0 = np.empty(3, dtype=np.float32)
            v1 = np.empty(3, dtype=np.float32)
            for axis in range(3):
                v0[axis] = np.float32(xyz[vertex, axis] - xyz[previous, axis])
                v1[axis] = np.float32(xyz[following, axis] - xyz[vertex, axis])
            _unit(v0)
            _unit(v1)
            face_normal = np.array([
                np.float32(-v1[1] * v0[2] + v0[1] * v1[2]),
                np.float32(v1[0] * v0[2] - v0[0] * v1[2]),
                np.float32(-v1[0] * v0[1] + v0[0] * v1[1]),
            ], dtype=np.float32)
            normal[0] = np.float32(normal[0] + face_normal[0])
            normal[1] = np.float32(normal[1] + face_normal[1])
            normal[2] = np.float32(normal[2] + face_normal[2])
        _unit(normal)
        result[vertex] = normal
    return result


def original_vertex_normals(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Recompute mrisComputeOrigNormal from ordered faces without face ripping."""
    xyz = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(triangles, dtype=np.int32)
    counts = np.bincount(faces.ravel(), minlength=len(xyz))
    offsets = np.empty(len(xyz) + 1, dtype=np.int32)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    ids = np.empty(len(faces) * 3, dtype=np.int32)
    corners = np.empty_like(ids)
    cursor = offsets[:-1].copy()
    for face_id, triangle in enumerate(faces):
        for corner, vertex in enumerate(triangle):
            position = cursor[vertex]
            ids[position] = face_id
            corners[position] = corner
            cursor[vertex] += 1
    return _original_normals(xyz, faces, ids, corners, offsets)


def vertex_buckets(current: np.ndarray, original: np.ndarray, ripped: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return original-vertex bucket candidates in source insertion order."""
    original_key = (np.asarray(original, dtype=np.float32) + np.float32(1000)).astype(np.int32)
    current_key = (np.asarray(current, dtype=np.float32) + np.float32(1000)).astype(np.int32)
    buckets: dict[tuple[int, int, int], list[int]] = {}
    for vertex in range(len(original_key)):
        if not ripped[vertex]:
            buckets.setdefault(tuple(original_key[vertex]), []).append(vertex)
    offsets = np.zeros(len(current_key) + 1, dtype=np.int32)
    flat: list[int] = []
    for vertex in range(len(current_key)):
        if not ripped[vertex]:
            flat.extend(buckets.get(tuple(current_key[vertex]), ()))
        offsets[vertex + 1] = len(flat)
    return offsets, np.asarray(flat, dtype=np.int32)


@njit(cache=True)
def _gradient(
    xyz: np.ndarray, normals: np.ndarray, original: np.ndarray,
    original_normals: np.ndarray, ripped: np.ndarray, cropped: np.ndarray,
    offsets: np.ndarray, candidates: np.ndarray, weight: float,
) -> np.ndarray:
    result = np.zeros_like(xyz)
    for vertex in range(len(xyz)):
        if ripped[vertex] or cropped[vertex]:
            continue
        sx = sy = sz = np.float32(0.0)
        x, y, z = xyz[vertex]
        nx, ny, nz = normals[vertex]
        for index in range(offsets[vertex], offsets[vertex + 1]):
            other = candidates[index]
            dx = np.float32(x - original[other, 0])
            dy = np.float32(y - original[other, 1])
            dz = np.float32(z - original[other, 2])
            dot = np.float32(np.float32(dx * original_normals[other, 0] + dy * original_normals[other, 1]) + dz * original_normals[other, 2])
            if dot > 1:
                continue
            dot = min(max(float(dot), -40.0), 40.0)
            scale = weight * (1.0 - dot) ** 4.0
            sx = np.float32(sx + scale * float(nx))
            sy = np.float32(sy + scale * float(ny))
            sz = np.float32(sz + scale * float(nz))
        result[vertex, 0] = sx
        result[vertex, 1] = sy
        result[vertex, 2] = sz
    return result


def surface_repulsion_gradient(
    vertices: np.ndarray, normals: np.ndarray, original_vertices: np.ndarray,
    original_normals: np.ndarray, ripped: np.ndarray,
    offsets: np.ndarray, candidates: np.ndarray, *, weight: float = 5.0,
    cropped: np.ndarray | None = None,
) -> np.ndarray:
    """Return original-surface repulsion, excluding collision-cropped vertices."""
    return _gradient(
        np.asarray(vertices, dtype=np.float32), np.asarray(normals, dtype=np.float32),
        np.asarray(original_vertices, dtype=np.float32), np.asarray(original_normals, dtype=np.float32),
        np.asarray(ripped, dtype=np.bool_),
        np.zeros(len(vertices), dtype=np.bool_) if cropped is None else np.asarray(cropped, dtype=np.bool_),
        np.asarray(offsets, dtype=np.int32),
        np.asarray(candidates, dtype=np.int32), float(np.float32(weight)),
    )
