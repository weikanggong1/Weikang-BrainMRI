"""First topology-pass principal curvatures from ordered orig and sphere meshes.

This is a bounded FreeSurfer 8.2 translation for the first global curvature
histogram. The 3x3 float SVD inverse matches isolated native source cases;
the complete installed FreeSurfer histogram has not yet matched.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from .smooth_surface_python import ordered_neighbors
from .topology_preflight_python import (center_sphere, defect_component_labels,
                                        project_and_smooth_sphere)
from .topology_vnl_svd import add, mul, svd_inverse_3


def _face_index(faces: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    slots = [[] for _ in range(count)]
    for face, row in enumerate(faces):
        for corner, vertex in enumerate(row):
            slots[int(vertex)].append((face, corner))
    offsets = np.zeros(count + 1, np.int32)
    for vertex, row in enumerate(slots):
        offsets[vertex + 1] = offsets[vertex] + len(row)
    face_ids = np.fromiter((face for row in slots for face, _ in row), np.int32)
    corners = np.fromiter((corner for row in slots for _, corner in row), np.int32)
    return offsets, face_ids, corners


@njit
def _unit(vector: np.ndarray) -> np.ndarray:
    squared = np.float32(vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2])
    length = np.float32(np.sqrt(float(squared)))
    if length > 0:
        vector[0] = np.float32(vector[0] / length)
        vector[1] = np.float32(vector[1] / length)
        vector[2] = np.float32(vector[2] / length)
    return vector


@njit
def _vertex_normals(xyz: np.ndarray, faces: np.ndarray, offsets: np.ndarray,
                    face_ids: np.ndarray, corners: np.ndarray,
                    ripped: np.ndarray) -> np.ndarray:
    result = np.zeros_like(xyz)
    for vertex in range(len(xyz)):
        if ripped[vertex]:
            continue
        normal = np.zeros(3, np.float32)
        for slot in range(offsets[vertex], offsets[vertex + 1]):
            face = faces[face_ids[slot]]
            corner = corners[slot]
            before = xyz[face[(corner + 2) % 3]]
            after = xyz[face[(corner + 1) % 3]]
            a = _unit((xyz[vertex] - before).copy())
            b = _unit((after - xyz[vertex]).copy())
            contribution = np.empty(3, np.float32)
            contribution[0] = np.float32(-b[1] * a[2] + a[1] * b[2])
            contribution[1] = np.float32(b[0] * a[2] - a[0] * b[2])
            contribution[2] = np.float32(-b[0] * a[1] + a[0] * b[1])
            contribution = _unit(contribution)
            normal[0] = np.float32(normal[0] + contribution[0])
            normal[1] = np.float32(normal[1] + contribution[1])
            normal[2] = np.float32(normal[2] + contribution[2])
        result[vertex] = _unit(normal)
    return result


def _two_rings(neighbors: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    for vertex, one in enumerate(neighbors):
        seen = {vertex, *one}
        row = list(one)
        for neighbor in one:
            for second in neighbors[neighbor]:
                if second not in seen:
                    seen.add(second)
                    row.append(second)
        rows.append(row)
    degree = np.fromiter((len(row) for row in rows), np.int32, count=len(rows))
    indices = np.zeros((len(rows), int(degree.max())), np.int32)
    for vertex, row in enumerate(rows):
        indices[vertex, :len(row)] = row
    return indices, degree


@njit
def _fit_normal_equations(xyz: np.ndarray, normals: np.ndarray,
                          neighbors: np.ndarray, degree: np.ndarray):
    count = len(xyz)
    gram = np.zeros((count, 3, 3), np.float32)
    rhs = np.zeros((count, 3), np.float32)
    extreme = np.empty((count, 2), np.float32)
    valid = np.zeros(count, np.int32)
    for vertex in range(count):
        nx, ny, nz = normals[vertex]
        e1x = np.float32(ny * nx - nz * nz)
        e1y = np.float32(nz * ny - nx * nx)
        e1z = np.float32(nx * nz - ny * ny)
        length = np.sqrt(float(np.float32(e1x * e1x + e1y * e1y + e1z * e1z)))
        if length < 0.001:
            e1x = np.float32(ny * nx + nz * nz)
            e1y = np.float32(nz * ny - nx * nx)
            e1z = np.float32(-nx * nz - ny * ny)
            length = np.sqrt(float(np.float32(e1x * e1x + e1y * e1y + e1z * e1z)))
        if length == 0:
            continue
        # The pinned source forms e2 from the unnormalized e1, then normalizes both.
        e2x = np.float32(ny * e1z - nz * e1y)
        e2y = np.float32(nz * e1x - nx * e1z)
        e2z = np.float32(nx * e1y - ny * e1x)
        scale = np.float32(np.float32(1) / np.float32(length))
        e1x = np.float32(e1x * scale)
        e1y = np.float32(e1y * scale)
        e1z = np.float32(e1z * scale)
        length = np.sqrt(float(np.float32(e2x * e2x + e2y * e2y + e2z * e2z)))
        if length == 0:
            continue
        scale = np.float32(np.float32(1) / np.float32(length))
        e2x = np.float32(e2x * scale)
        e2y = np.float32(e2y * scale)
        e2z = np.float32(e2z * scale)
        kmax = np.float32(-10000)
        kmin = np.float32(10000)
        for slot in range(degree[vertex]):
            other = neighbors[vertex, slot]
            dx = np.float32(xyz[other, 0] - xyz[vertex, 0])
            dy = np.float32(xyz[other, 1] - xyz[vertex, 1])
            dz = np.float32(xyz[other, 2] - xyz[vertex, 2])
            u = np.float32(np.float32(dx * e1x + dy * e1y) + dz * e1z)
            v = np.float32(np.float32(dx * e2x + dy * e2y) + dz * e2z)
            z = np.float32(np.float32(dx * nx + dy * ny) + dz * nz)
            radius_squared = np.float32(np.float64(u) * np.float64(u) + np.float64(v) * np.float64(v))
            if abs(radius_squared) < 1e-6:
                continue
            k = np.float32(z / radius_squared)
            kmax = max(kmax, k)
            kmin = min(kmin, k)
            row = np.empty(3, np.float32)
            row[0] = np.float32(np.float64(u) * np.float64(u))
            row[1] = np.float32(2.0 * np.float64(u) * np.float64(v))
            row[2] = np.float32(np.float64(v) * np.float64(v))
            for i in range(3):
                rhs[vertex, i] = np.float32(rhs[vertex, i] + np.float32(row[i] * z))
                for j in range(3):
                    gram[vertex, i, j] = np.float32(gram[vertex, i, j] + np.float32(row[i] * row[j]))
            valid[vertex] += 1
        extreme[vertex, 0] = kmin
        extreme[vertex, 1] = kmax
    return gram, rhs, extreme, valid


@njit
def _svd_coefficients(gram: np.ndarray, rhs: np.ndarray,
                      good: np.ndarray) -> np.ndarray:
    coeff = np.zeros((len(gram), 3), np.float32)
    for vertex in range(len(gram)):
        if not good[vertex]:
            continue
        inverse = svd_inverse_3(gram[vertex])
        for row in range(3):
            value = np.float32(0)
            for col in range(3):
                value = add(value, mul(inverse[row, col], rhs[vertex, col]))
            coeff[vertex, row] = value
    return coeff


def first_principal_curvatures(original: np.ndarray, qsphere: np.ndarray,
                               faces: np.ndarray) -> dict[str, np.ndarray]:
    """Fit first global k1/k2 values without native curvature or plot inputs."""
    xyz = np.ascontiguousarray(original, np.float32)
    sphere = np.ascontiguousarray(qsphere, np.float32)
    triangles = np.ascontiguousarray(faces, np.int32)
    if xyz.shape != sphere.shape or xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("original and qsphere must have matching (vertex, 3) shapes")
    offsets, face_ids, corners = _face_index(triangles, len(xyz))
    canonical = center_sphere(project_and_smooth_sphere(sphere, triangles))[0]
    labels = defect_component_labels(canonical, triangles)
    ripped = labels != 0
    normals = _vertex_normals(xyz, triangles, offsets, face_ids, corners, ripped)
    sphere_normals = _vertex_normals(canonical, triangles, offsets, face_ids,
                                    corners, np.zeros(len(xyz), np.bool_))
    normals[ripped] = sphere_normals[ripped]
    neighbors = ordered_neighbors(triangles, len(xyz))
    two_rings, degree = _two_rings(neighbors)
    gram, rhs, extreme, valid = _fit_normal_equations(xyz, normals, two_rings, degree)
    eigen = np.linalg.eigvalsh(gram.astype(np.float64))
    condition = np.divide(eigen[:, -1], eigen[:, 0],
                          out=np.full(len(xyz), np.inf), where=eigen[:, 0] > 0)
    good = (valid >= 4) & (np.abs(np.linalg.det(gram.astype(np.float64))) > 1e-15)
    coeff = _svd_coefficients(gram, rhs, good)
    a, b, c = coeff.T
    # VNL copies the float32 Hessian into double before its EISPACK rs_ call.
    h00 = np.float32(np.float32(2) * a).astype(np.float64)
    h01 = np.float32(np.float32(2) * b).astype(np.float64)
    h11 = np.float32(np.float32(2) * c).astype(np.float64)
    center = (h00 + h11) / 2.0
    spread = np.sqrt(((h00 - h11) / 2.0) ** 2 + h01 ** 2)
    low = np.float32(center - spread)
    high = np.float32(center + spread)
    first = np.abs(low) >= np.abs(high)
    k1 = np.where(first, low, high).astype(np.float32)
    k2 = np.where(first, high, low).astype(np.float32)
    ill = condition >= 500000.0
    k1[ill] = extreme[ill, 1]
    k2[ill] = extreme[ill, 0]
    return {"k1": k1, "k2": k2, "defect_labels": labels,
            "invalid": ~good, "ill_conditioned": ill}
