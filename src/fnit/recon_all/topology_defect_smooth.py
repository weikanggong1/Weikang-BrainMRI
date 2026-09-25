"""Source-order curvature-adaptive smoothing for an already retessellated patch.

The first-candidate validator derives histogram means from native diagnostic
plots and MGH metadata. Native-free histogram construction and subsequent MRI
matching remain separate topology steps.
"""

from __future__ import annotations

import numpy as np
from numba import njit

@njit
def _smooth_core(xyz, faces, inside, neighbors, degree, offsets, face_ids,
                 k1_mean, k2_mean):
    current = xyz.copy()
    face_normal = np.zeros((len(faces), 3), np.float32)
    rmin = np.float32(-1.0 / k1_mean)
    rmax = np.float32(-1.0 / k2_mean)
    e = np.float32((np.float32(1.0 / rmin) + np.float32(1.0 / rmax)) / np.float32(2))
    fscale = np.float32(6.0 / np.float32(np.float32(1.0 / rmin) - np.float32(1.0 / rmax)))
    for _ in range(25):
        for fi in range(len(faces)):
            a, b, c = faces[fi]
            ax = np.float32(current[b, 0] - current[a, 0])
            ay = np.float32(current[b, 1] - current[a, 1])
            az = np.float32(current[b, 2] - current[a, 2])
            bx = np.float32(current[c, 0] - current[a, 0])
            by = np.float32(current[c, 1] - current[a, 1])
            bz = np.float32(current[c, 2] - current[a, 2])
            nx = np.float32(ay * bz - az * by)
            ny = np.float32(az * bx - ax * bz)
            nz = np.float32(ax * by - ay * bx)
            length = np.sqrt(np.float32(nx * nx + ny * ny + nz * nz))
            if length < 1e-5:
                length = np.float32(1.0)
            face_normal[fi, 0] = np.float32(nx / length)
            face_normal[fi, 1] = np.float32(ny / length)
            face_normal[fi, 2] = np.float32(nz / length)
        next_xyz = current.copy()
        for v in inside:
            nx = ny = nz = np.float32(0)
            for slot in range(offsets[v], offsets[v + 1]):
                fi = face_ids[slot]
                nx = np.float32(nx + face_normal[fi, 0])
                ny = np.float32(ny + face_normal[fi, 1])
                nz = np.float32(nz + face_normal[fi, 2])
            length = np.sqrt(np.float32(nx * nx + ny * ny + nz * nz))
            if length < 1e-5:
                length = np.float32(1.0)
            nx = np.float32(nx / length)
            ny = np.float32(ny / length)
            nz = np.float32(nz / length)
            x, y, z = current[v]
            sx = sy = sz = sd = np.float32(0)
            n = 0
            while n < degree[v]:
                other = neighbors[v, n]
                dx = np.float32(current[other, 0] - x)
                dy = np.float32(current[other, 1] - y)
                dz = np.float32(current[other, 2] - z)
                sx = np.float32(sx + dx)
                sy = np.float32(sy + dy)
                sz = np.float32(sz + dz)
                sd = np.float32(sd + np.sqrt(np.float32(dx * dx + dy * dy + dz * dz)))
                n += 2  # source increments n in the loop body and in the for header
            denom = np.float32(n)
            sx = np.float32(sx / denom)
            sy = np.float32(sy / denom)
            sz = np.float32(sz / denom)
            sd = np.float32(sd / denom)
            nc = np.float32(sx * nx + sy * ny + sz * nz)
            sxn = np.float32(nc * nx)
            syn = np.float32(nc * ny)
            szn = np.float32(nc * nz)
            sxt = np.float32(sx - sxn)
            syt = np.float32(sy - syn)
            szt = np.float32(sz - szn)
            radius = np.float32(sd * sd / np.float32(2 * abs(nc))) if nc != 0 else np.float32(np.inf)
            gain = np.float32((1.0 + np.tanh(float(np.float32(fscale * np.float32(np.float32(1.0 / radius) - e))))) / 2.0)
            next_xyz[v, 0] = np.float32(float(x) + 0.1 * float(np.float32(sxt + gain * sxn)))
            next_xyz[v, 1] = np.float32(float(y) + 0.1 * float(np.float32(syt + gain * syn)))
            next_xyz[v, 2] = np.float32(float(z) + 0.1 * float(np.float32(szt + gain * szn)))
        current = next_xyz
    return current


def defect_smooth_type2(xyz: np.ndarray, faces: np.ndarray, inside_vertices: np.ndarray,
                        runtime_neighbors: list[list[int]],
                        k1_mean: float, k2_mean: float) -> np.ndarray:
    """Run the source's 25 iterations on retained interior patch vertices."""
    points = np.ascontiguousarray(xyz, np.float32)
    triangles = np.ascontiguousarray(faces, np.int32)
    inside = np.ascontiguousarray(inside_vertices, np.int32)
    if len(inside) == 0:
        return points.copy()
    degree = np.zeros(len(points), np.int32)
    maximum = max(map(len, runtime_neighbors))
    neighbors = np.zeros((len(points), maximum), np.int32)
    for vertex, row in zip(inside, runtime_neighbors):
        degree[vertex] = len(row)
        neighbors[vertex, :len(row)] = row
    slots = [[] for _ in range(len(points))]
    for fi, face in enumerate(triangles):
        for vertex in face:
            slots[int(vertex)].append(fi)
    offsets = np.zeros(len(points) + 1, np.int32)
    for vertex, row in enumerate(slots):
        offsets[vertex + 1] = offsets[vertex] + len(row)
    face_ids = np.asarray([fi for row in slots for fi in row], np.int32)
    return _smooth_core(points, triangles, inside, neighbors, degree,
                        offsets, face_ids, np.float32(k1_mean), np.float32(k2_mean))
