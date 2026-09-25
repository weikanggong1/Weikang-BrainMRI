"""White midline/BG and pial label ripping for the fixed T1 placement calls."""

from __future__ import annotations

import numpy as np
from numba import njit

from .place_surface_border import _sample, _voxel
from .place_surface_smoothing import _ordered_neighbors


def rip_outside_label(vertex_count: int, label_vertices: np.ndarray) -> np.ndarray:
    """Return the initial ``MRISripNotLabel`` flags in vertex order."""
    ripped = np.ones(vertex_count, dtype=np.int32)
    ripped[np.asarray(label_vertices, dtype=np.int64)] = 0
    return ripped


# The white.preaparc branch has no annotation, label rip, or separate rip
# surface. The two midline passes use the same mesh with progressively
# accumulated rip flags.
@njit(cache=True)
def _nearest(seg: np.ndarray, affine: np.ndarray, x: float, y: float, z: float) -> int:
    vx, vy, vz = _voxel(affine, x, y, z)
    i = int(np.floor(vx + 0.5))
    j = int(np.floor(vy + 0.5))
    k = int(np.floor(vz + 0.5))
    if i < 0 or j < 0 or k < 0 or i >= seg.shape[0] or j >= seg.shape[1] or k >= seg.shape[2]:
        return 0
    return int(seg[i, j, k])


@njit(cache=True)
def _label_normal_counts(seg: np.ndarray, affine: np.ndarray, x: float, y: float, z: float, label: int) -> tuple[int, int, int]:
    vx, vy, vz = _voxel(affine, x, y, z)
    cx, cy, cz = int(vx), int(vy), int(vz)
    counts = np.zeros(3, dtype=np.int32)
    for dx in range(-3, 4):
        ix = min(max(cx + dx, 0), seg.shape[0] - 1)
        for dy in range(-3, 4):
            iy = min(max(cy + dy, 0), seg.shape[1] - 1)
            for dz in range(-3, 4):
                iz = min(max(cz + dz, 0), seg.shape[2] - 1)
                if seg[ix, iy, iz] != label:
                    continue
                for axis in range(3):
                    for direction in (-1, 1):
                        nx, ny, nz = ix, iy, iz
                        if axis == 0:
                            nx = min(max(ix + direction, 0), seg.shape[0] - 1)
                        elif axis == 1:
                            ny = min(max(iy + direction, 0), seg.shape[1] - 1)
                        else:
                            nz = min(max(iz + direction, 0), seg.shape[2] - 1)
                        if seg[nx, ny, nz] != label:
                            counts[axis] += 1
    return counts[0], counts[1], counts[2]


@njit(cache=True)
def _is_wmsa(label: int) -> bool:
    return label in (77, 78, 79, 87, 88, 89)


@njit(cache=True)
def _mark_midline(
    xyz: np.ndarray,
    normals: np.ndarray,
    seg: np.ndarray,
    brain: np.ndarray,
    affine: np.ndarray,
    ripped: np.ndarray,
    values: np.ndarray,
    left: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    marked = ripped.astype(np.int32)
    marked2 = np.zeros(len(xyz), dtype=np.int32)
    own_wm, own_gm, contra_wm = (2, 3, 41) if left else (41, 42, 2)
    forbidden = (contra_wm, 4, 30, 62, 85, 31, 63, 14, 43, 26, 58,
                 25, 57, 77, 78, 79, 81, 82, 11, 50, 13, 52,
                 10, 49, 16, 28, 60)
    for vertex in range(len(xyz)):
        if ripped[vertex]:
            continue
        x, y, z = float(xyz[vertex, 0]), float(xyz[vertex, 1]), float(xyz[vertex, 2])
        nx, ny, nz = float(normals[vertex, 0]), float(normals[vertex, 1]), float(normals[vertex, 2])
        vx, vy, vz = _voxel(affine, x, y, z)
        center_value = np.float32(_sample(brain, vx, vy, vz))
        for step in range(5):
            d = 0.5 * step
            label = _nearest(seg, affine, x + d*nx, y + d*ny, z + d*nz)
            if d > 0 and label == own_gm:
                break
            if label in forbidden or 251 <= label <= 255:
                if label in (25, 57) or _is_wmsa(label):
                    marked2[vertex] = 1
                values[vertex] = center_value
                marked[vertex] = 1
        for step in range(5):
            d = 0.5 * step
            label = _nearest(seg, affine, x - d*nx, y - d*ny, z - d*nz)
            if d < 1 and (label in (25, 57) or _is_wmsa(label)):
                marked[vertex] = 1
                marked2[vertex] = 1
            if label in (12, 51, 26, 58, 138, 139):
                ax, ay, az = _label_normal_counts(seg, affine, x - d*nx, y - d*ny, z - d*nz, label)
                if ax > ay and ax > az:
                    if label not in (26, 58):
                        values[vertex] = center_value
                    marked[vertex] = 1
                    break
        for step in range(5):
            d = 0.5 * step
            label = _nearest(seg, affine, x - d*nx, y - d*ny, z - d*nz)
            if d < 1.1 and label in (own_wm, own_gm):
                break
            if (label in forbidden or 251 <= label <= 255 or
                    (d < 1.1 and label in (12, 51))):
                if label in (4, 43) and d > 1:
                    break
                values[vertex] = center_value
                marked[vertex] = 1
        nput = 0
        adjacent = False
        for step in range(21):
            d = 0.5 * step
            label = _nearest(seg, affine, x, y, z + d)
            if label in (12, 51):
                nput += 1
                if d < 1.5:
                    adjacent = True
        if adjacent and nput / 21.0 > 0.5:
            label = _nearest(seg, affine, x, y, z)
            ax, ay, az = _label_normal_counts(seg, affine, x, y, z, label)
            if ay > 0 and ay > ax and ay > az:
                values[vertex] = center_value
                marked[vertex] = 1
    return marked, marked2, values


@njit(cache=True)
def _close_marks(marked: np.ndarray, ripped: np.ndarray, neighbors: np.ndarray, valid: np.ndarray) -> np.ndarray:
    current = marked.copy()
    for dilate in (True, False):
        for _ in range(3):
            following = current.copy()
            for vertex in range(len(current)):
                if ripped[vertex]:
                    continue
                state = current[vertex]
                for rank in range(neighbors.shape[1]):
                    if not valid[vertex, rank]:
                        break
                    other = current[neighbors[vertex, rank]]
                    if dilate:
                        state = max(state, other)
                    else:
                        state = min(state, other)
                following[vertex] = state
            current = following
    return current


def _small_components(
    marked: np.ndarray, ripped: np.ndarray, neighbors: np.ndarray, valid: np.ndarray,
    xyz: np.ndarray, faces: np.ndarray,
) -> np.ndarray:
    vertex_area = np.zeros(len(xyz), dtype=np.float64)
    edge_a = xyz[faces[:, 1]].astype(np.float64) - xyz[faces[:, 0]].astype(np.float64)
    edge_b = xyz[faces[:, 2]].astype(np.float64) - xyz[faces[:, 0]].astype(np.float64)
    area = np.linalg.norm(np.cross(edge_a, edge_b), axis=1) / 6.0
    for corner in range(3):
        np.add.at(vertex_area, faces[:, corner], area)
    result = marked.copy()
    visited = np.zeros(len(marked), dtype=np.bool_)
    for vertex in np.flatnonzero((marked != 0) & (ripped == 0)):
        if visited[vertex]:
            continue
        cluster = [int(vertex)]
        visited[vertex] = True
        for current in cluster:
            for other in neighbors[current, valid[current]]:
                if marked[other] and not ripped[other] and not visited[other]:
                    visited[other] = True
                    cluster.append(int(other))
        if len(cluster) < 5 and vertex_area[cluster].sum() >= 1.0:
            result[cluster] = 0
    return result


def rip_white_preaparc_pass(
    vertices: np.ndarray, normals: np.ndarray, faces: np.ndarray,
    segmentation: np.ndarray, placement_volume: np.ndarray, sras2vox: np.ndarray,
    *, hemisphere: str, ripped: np.ndarray | None = None, values: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run one fixed-branch ``MRISripMidline`` plus BG pass before targeting."""
    xyz = np.asarray(vertices, dtype=np.float32)
    normals = np.asarray(normals, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)
    seg = np.asarray(segmentation, dtype=np.int32)
    volume = np.asarray(placement_volume, dtype=np.uint8)
    transform = np.asarray(sras2vox, dtype=np.float32)
    rip = np.zeros(len(xyz), dtype=np.int32) if ripped is None else np.asarray(ripped, dtype=np.int32).copy()
    val = np.full(len(xyz), -1.0, dtype=np.float32) if values is None else np.asarray(values, dtype=np.float32).copy()
    neighbors, valid, _ = _ordered_neighbors(faces, len(xyz))
    marked, _, val = _mark_midline(xyz, normals, seg, volume, transform, rip, val, hemisphere == "lh")
    marked = _close_marks(marked, rip, neighbors, valid)
    marked = _small_components(marked, rip, neighbors, valid, xyz, faces)
    rip[marked != 0] = 1
    for vertex in np.flatnonzero(rip == 0):
        x, y, z = xyz[vertex]
        nx, ny, nz = normals[vertex]
        for step in range(9):
            d = -2.0 + 0.5*step
            label = _nearest(seg, transform, float(x)+d*float(nx), float(y)+d*float(ny), float(z)+d*float(nz))
            if label in (11, 12, 26, 50, 51, 58, 138, 139):
                rip[vertex] = 1
                break
    return rip, val
