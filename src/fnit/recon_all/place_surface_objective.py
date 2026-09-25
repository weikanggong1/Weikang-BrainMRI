"""Active objective terms for the fixed FreeSurfer pial placement call."""

from __future__ import annotations

import math

import numpy as np
from numba import njit

from .place_surface_border import _sample, _voxel
from .place_surface_smoothing import _ordered_neighbors


@njit(cache=True)
def _intensity_error(
    volume: np.ndarray, vertices: np.ndarray, values: np.ndarray,
    ripped: np.ndarray, affine: np.ndarray,
) -> tuple[float, float, int]:
    total = 0.0
    count = 0
    for vertex in range(len(vertices)):
        if ripped[vertex] or values[vertex] < 0:
            continue
        count += 1
        x, y, z = _voxel(
            affine, float(vertices[vertex, 0]),
            float(vertices[vertex, 1]), float(vertices[vertex, 2]),
        )
        delta = _sample(volume, x, y, z) - float(values[vertex])
        total += delta * delta
    return total, math.sqrt(total / count), count


def intensity_error(
    volume: np.ndarray, vertices: np.ndarray, target_values: np.ndarray,
    ripped: np.ndarray, surface_ras_to_voxel: np.ndarray,
) -> tuple[float, float, int]:
    """Return unweighted intensity SSE, RMS and sampled vertex count."""
    return _intensity_error(
        np.asarray(volume, dtype=np.uint8), np.asarray(vertices, dtype=np.float32),
        np.asarray(target_values, dtype=np.float32), np.asarray(ripped, dtype=np.bool_),
        np.asarray(surface_ras_to_voxel, dtype=np.float32),
    )


@njit(cache=True)
def _tangential_energy(
    xyz: np.ndarray, normals: np.ndarray, ripped: np.ndarray,
    neighbors: np.ndarray, valid: np.ndarray,
) -> float:
    total = 0.0
    for vertex in range(len(xyz)):
        if ripped[vertex]:
            continue
        x, y, z = xyz[vertex]
        nx, ny, nz = normals[vertex]
        vertex_sum = 0.0
        for rank in range(neighbors.shape[1]):
            if not valid[vertex, rank]:
                break
            other = neighbors[vertex, rank]
            dx = np.float32(xyz[other, 0] - x)
            dy = np.float32(xyz[other, 1] - y)
            dz = np.float32(xyz[other, 2] - z)
            component = np.float32(np.float32(dx * nx + dy * ny) + dz * nz)
            dx = np.float32(dx - np.float32(component * nx))
            dy = np.float32(dy - np.float32(component * ny))
            dz = np.float32(dz - np.float32(component * nz))
            squared = np.float32(np.float32(dx * dx + dy * dy) + dz * dz)
            vertex_sum += float(squared)
        total += vertex_sum
    return total


def tangential_spring_energy(
    vertices: np.ndarray, normals: np.ndarray, faces: np.ndarray, ripped: np.ndarray,
    *, ordered_neighbors: tuple[np.ndarray, np.ndarray] | None = None,
) -> float:
    """Return native-order tangential spring SSE before its cost weight."""
    xyz = np.asarray(vertices, dtype=np.float32)
    if ordered_neighbors is None:
        neighbors, valid, _ = _ordered_neighbors(np.asarray(faces, dtype=np.int32), len(xyz))
    else:
        neighbors, valid = ordered_neighbors
    return _tangential_energy(
        xyz, np.asarray(normals, dtype=np.float32),
        np.asarray(ripped, dtype=np.bool_), neighbors, valid,
    )


@njit(cache=True)
def _surface_total_area(xyz: np.ndarray, faces: np.ndarray, face_ripped: np.ndarray) -> np.float32:
    total = 0.0
    for face_index in range(len(faces)):
        if face_ripped[face_index]:
            continue
        a, b, c = faces[face_index]
        ax = np.float32(xyz[b, 0] - xyz[a, 0])
        ay = np.float32(xyz[b, 1] - xyz[a, 1])
        az = np.float32(xyz[b, 2] - xyz[a, 2])
        bx = np.float32(xyz[c, 0] - xyz[a, 0])
        by = np.float32(xyz[c, 1] - xyz[a, 1])
        bz = np.float32(xyz[c, 2] - xyz[a, 2])
        cx = np.float32(np.float32(ay * bz) - np.float32(az * by))
        cy = np.float32(np.float32(az * bx) - np.float32(ax * bz))
        cz = np.float32(np.float32(ax * by) - np.float32(ay * bx))
        squared = np.float32(np.float32(cx * cx + cy * cy) + cz * cz)
        total += float(np.float32(math.sqrt(float(squared)) * 0.5))
    return np.float32(total)


def surface_total_area(vertices: np.ndarray, faces: np.ndarray, face_ripped: np.ndarray | None = None) -> np.float32:
    """Return FreeSurfer's float32 total face area after double accumulation."""
    xyz = np.asarray(vertices, dtype=np.float32)
    triangles = np.asarray(faces, dtype=np.int32)
    skip = np.zeros(len(triangles), dtype=np.bool_) if face_ripped is None else np.asarray(face_ripped, dtype=np.bool_)
    return _surface_total_area(xyz, triangles, skip)


def pial_placement_sse(
    intensity_sse: float, spring_energy: float, orig_area: np.float32, total_area: np.float32,
) -> float:
    """Weighted active terms of the fixed FreeSurfer pial placement objective."""
    area_scale = float(np.float32(np.float32(orig_area) / np.float32(total_area)))
    return float(np.float32(0.3)) * spring_energy * area_scale + float(np.float32(0.2)) * intensity_sse
