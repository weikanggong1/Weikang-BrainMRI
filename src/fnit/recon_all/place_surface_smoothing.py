"""The initial vertex averaging in FreeSurfer 8.2 ``mris_place_surface``.

The fixed white.preaparc call runs this before ripping or border placement.
Faces are visited in file order, and each face contributes the previous then
next vertex to each incident vertex's ordered neighbor list.
"""

from __future__ import annotations

import numpy as np
import torch


def _ordered_neighbors(triangles: np.ndarray, nvertices: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    neighbors: list[list[int]] = [[] for _ in range(nvertices)]
    for a, b, c in triangles:
        for vertex, previous, following in ((a, c, b), (b, a, c), (c, b, a)):
            row = neighbors[vertex]
            if previous not in row:
                row.append(int(previous))
            if following not in row:
                row.append(int(following))
    width = max(map(len, neighbors), default=0)
    indices = np.zeros((nvertices, width), dtype=np.int64)
    valid = np.zeros((nvertices, width), dtype=np.bool_)
    counts = np.empty((nvertices, 1), dtype=np.float32)
    for vertex, row in enumerate(neighbors):
        indices[vertex, : len(row)] = row
        valid[vertex, : len(row)] = True
        counts[vertex, 0] = len(row) + 1
    return indices, valid, counts


def average_vertex_positions(
    vertices: np.ndarray,
    faces: np.ndarray,
    iterations: int,
    *,
    device: str = "cpu",
) -> np.ndarray:
    """Return the float32 coordinates after native-order vertex averaging."""
    if iterations < 0:
        raise ValueError("iterations must be nonnegative")
    xyz = np.asarray(vertices, dtype=np.float32)
    triangles = np.asarray(faces, dtype=np.int32)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("expected vertices (N, 3) and faces (M, 3)")
    if iterations == 0:
        return xyz.copy()

    indices, valid, counts = _ordered_neighbors(triangles, len(xyz))
    width = indices.shape[1]

    position = torch.as_tensor(xyz, device=device)
    neighbor_index = torch.as_tensor(indices, device=device)
    neighbor_valid = torch.as_tensor(valid, device=device)
    divisor = torch.as_tensor(counts, device=device)
    for _ in range(iterations):
        averaged = position.clone()
        for rank in range(width):
            values = position[neighbor_index[:, rank]]
            averaged += torch.where(neighbor_valid[:, rank, None], values, 0)
        position = averaged / divisor
    return position.cpu().numpy()


def average_marked_values(
    values: np.ndarray,
    marked: np.ndarray,
    ripped: np.ndarray,
    faces: np.ndarray,
    iterations: int,
    *,
    device: str = "cpu",
) -> np.ndarray:
    """Match ``MRISaverageMarkedVals`` after border-value search."""
    if iterations < 0:
        raise ValueError("iterations must be nonnegative")
    data = np.asarray(values, dtype=np.float32)
    active = np.asarray(marked, dtype=np.bool_) & ~np.asarray(ripped, dtype=np.bool_)
    if data.ndim != 1 or active.shape != data.shape:
        raise ValueError("values, marked and ripped must have one element per vertex")
    if iterations == 0:
        return data.copy()
    indices, valid, _ = _ordered_neighbors(np.asarray(faces, dtype=np.int32), len(data))
    current = torch.as_tensor(data, device=device)
    neighbor_index = torch.as_tensor(indices, device=device)
    neighbor_active = torch.as_tensor(valid & active[indices], device=device)
    center_active = torch.as_tensor(active, device=device)
    for _ in range(iterations):
        result = current.clone()
        count = torch.ones_like(current)
        for rank in range(indices.shape[1]):
            neighbor = neighbor_active[:, rank]
            result += torch.where(neighbor, current[neighbor_index[:, rank]], 0)
            count += neighbor.to(torch.float32)
        current = torch.where(center_active, result / count, current)
    return current.cpu().numpy()
