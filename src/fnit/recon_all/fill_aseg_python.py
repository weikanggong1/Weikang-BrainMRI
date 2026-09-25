"""Aseg-guided hemisphere fill from FreeSurfer's mri_fill (8.2.0)."""

from __future__ import annotations

import numpy as np
from numba import njit
from scipy import ndimage as ndi


_LEFT = (2, 3, 4, 5, 10, 11, 12, 13, 17, 18, 19, 20, 25, 30, 26, 28, 31, 27,
         32, 33, 34, 35, 36, 37, 38, 39)
_RIGHT = (41, 42, 43, 44, 49, 50, 51, 52, 53, 54, 55, 56, 57, 62, 58, 60, 63,
          59, 64, 65, 66, 67, 68, 69, 70, 71)
_ERASE = (7, 8, 16, 46, 47)
_WMSA = (77, 78, 79, 87, 88)


def _cc_outside_distance(seg: np.ndarray, cc: np.ndarray, label: int) -> np.ndarray:
    """Fast-marching outside distance, stopping once all CC voxels are settled."""
    points = np.argwhere(cc)
    label_points = np.argwhere(seg == label)
    lo = np.maximum(np.minimum(points.min(axis=0), label_points.min(axis=0)) - 12, 0)
    hi = np.minimum(np.maximum(points.max(axis=0), label_points.max(axis=0)) + 13, seg.shape)
    area = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    target = seg[area] == label
    query = cc[area]
    # FreeSurfer initializes outside boundary voxels at +0.5 voxel.
    distance = np.full(target.shape, np.float32(100.0), dtype=np.float32)
    state = np.zeros(target.shape, dtype=np.uint8)  # 0 far, 1 trial, 2 alive
    state[target] = 3  # forbidden
    distance[target] = 0.0
    heap: list[tuple[int, int, int]] = []
    sx, sy, sz = target.shape

    alive: list[tuple[int, int, int]] = []

    def add_alive(x: int, y: int, z: int) -> None:
        if state[x, y, z] == 3:
            return
        distance[x, y, z] = np.float32(0.5)
        if state[x, y, z] == 0:
            state[x, y, z] = 2
            alive.append((x, y, z))

    for z in range(sz):
        for y in range(sy):
            for x in range(sx):
                val = target[x, y, z]
                changed = False
                if x + 1 < sx and val != target[x + 1, y, z]:
                    changed = True
                    add_alive(x + 1, y, z)
                if y + 1 < sy and val != target[x, y + 1, z]:
                    changed = True
                    add_alive(x, y + 1, z)
                if z + 1 < sz and val != target[x, y, z + 1]:
                    changed = True
                    add_alive(x, y, z + 1)
                if changed:
                    add_alive(x, y, z)
    surface = state == 2

    def priority(point: tuple[int, int, int]) -> float:
        return float(distance[point])

    def push(point: tuple[int, int, int]) -> None:
        heap.append(point)
        index = len(heap) - 1
        while index:
            parent = (index - 1) // 2
            if priority(heap[parent]) <= priority(point):
                break
            heap[index] = heap[parent]
            index = parent
        heap[index] = point

    def pop() -> tuple[int, int, int]:
        first = heap[0]
        last = heap.pop()
        if heap:
            index = 0
            length = len(heap)
            while 2 * index + 1 < length:
                child = 2 * index + 1
                if child + 1 < length and priority(heap[child + 1]) <= priority(heap[child]):
                    child += 1
                heap[index] = heap[child]
                index = child
            while index:
                parent = (index - 1) // 2
                if priority(heap[parent]) <= priority(last):
                    break
                heap[index] = heap[parent]
                index = parent
            heap[index] = last
        return first

    def update(x: int, y: int, z: int) -> None:
        if state[x, y, z] >= 2:
            return
        a = min(distance[x - 1, y, z] if x else 100,
                distance[x + 1, y, z] if x + 1 < sx else 100)
        b = min(distance[x, y - 1, z] if y else 100,
                distance[x, y + 1, z] if y + 1 < sy else 100)
        c = min(distance[x, y, z - 1] if z else 100,
                distance[x, y, z + 1] if z + 1 < sz else 100)
        a, b, c = sorted((np.float32(a), np.float32(b), np.float32(c)))
        one = np.float32(1)
        value = np.float32(a + one)
        sum3 = np.float32(np.float32(a + b) + c)
        squares3 = np.float32(np.float32(a * a + b * b) + c * c)
        delta = np.float32(sum3 * sum3 - np.float32(3) * np.float32(squares3 - one))
        solved = False
        if delta >= 0:
            solution = np.float32((float(sum3) + float(np.sqrt(float(delta)))) / 3.0)
            if np.float32(solution + np.float32(1e-6)) >= c:
                value = solution
                solved = True
        if not solved:
            sum2 = np.float32(a + b)
            squares2 = np.float32(a * a + b * b)
            delta = np.float32(sum2 * sum2 - np.float32(2) * np.float32(squares2 - one))
            if delta >= 0:
                solution = np.float32((float(sum2) + float(np.sqrt(float(delta)))) / 2.0)
                if np.float32(solution + np.float32(1e-6)) >= b:
                    value = solution
        distance[x, y, z] = np.float32(value)
        if state[x, y, z] == 0:
            state[x, y, z] = 1
            push((x, y, z))

    def update_neighbors(x: int, y: int, z: int, initial: bool = False) -> None:
        for xx, yy, zz in ((x - 1, y, z), (x + 1, y, z),
                           (x, y - 1, z), (x, y + 1, z),
                           (x, y, z - 1), (x, y, z + 1)):
            if 0 <= xx < sx and 0 <= yy < sy and 0 <= zz < sz:
                if not initial or state[xx, yy, zz] == 0:
                    update(xx, yy, zz)

    for x, y, z in alive:
        update_neighbors(x, y, z, initial=True)
    remaining = int(np.count_nonzero(query & ~surface))
    while remaining and heap:
        x, y, z = pop()
        state[x, y, z] = 2
        if query[x, y, z]:
            remaining -= 1
        update_neighbors(x, y, z)
    result = np.full(seg.shape, np.float32(100.0), dtype=np.float32)
    result[area] = distance
    return result


@njit(cache=True)
def _voronoi_round(values: np.ndarray, distances: np.ndarray,
                   coordinates: np.ndarray, radius: int) -> None:
    sx, sy, sz = values.shape
    for i in range(len(coordinates)):
        x, y, z = coordinates[i]
        total = 0
        count = 0
        for dx in (-1, 0, 1):
            xi = min(max(x + dx, 0), sx - 1)
            for dy in (-1, 0, 1):
                yi = min(max(y + dy, 0), sy - 1)
                for dz in (-1, 0, 1):
                    zi = min(max(z + dz, 0), sz - 1)
                    if distances[xi, yi, zi] < radius:
                        total += int(values[xi, yi, zi])
                        count += 1
        if count:
            values[x, y, z] = total // count


@njit(cache=True)
def _edited_on_votes(fill: np.ndarray, wm: np.ndarray, aseg: np.ndarray,
                     voxel_xsize: float) -> None:
    sx, sy, sz = wm.shape
    coords = np.argwhere(wm == 255)
    for i in range(len(coords)):
        x, y, z = coords[i]
        radius = int(np.ceil(5.0 / voxel_xsize))
        while True:
            left = 0
            right = 0
            for dx in range(-radius, radius + 1):
                xi = min(max(x + dx, 0), sx - 1)
                for dy in range(-radius, radius + 1):
                    yi = min(max(y + dy, 0), sy - 1)
                    for dz in range(-radius, radius + 1):
                        zi = min(max(z + dz, 0), sz - 1)
                        label = aseg[xi, yi, zi]
                        if label == 2 or label == 3:
                            left += 1
                        elif label == 41 or label == 42:
                            right += 1
            if left or right or radius > 20:
                break
            radius += 1
        fill[x, y, z] = 255 if left > right else 127


def _largest_then_fill_holes(mask: np.ndarray) -> np.ndarray:
    labels, count = ndi.label(mask, structure=ndi.generate_binary_structure(3, 2))
    if count:
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        mask = labels == np.argmax(sizes)
    return ndi.binary_fill_holes(mask, structure=ndi.generate_binary_structure(3, 1))


def _replace_cc_with_wm(aseg: np.ndarray) -> np.ndarray:
    seg = np.asarray(aseg, dtype=np.int32).copy()
    if np.any((seg >= 251) & (seg <= 255)):
        cc = (seg >= 251) & (seg <= 255)
        left_distance = _cc_outside_distance(seg, cc, 2)
        right_distance = _cc_outside_distance(seg, cc, 41)
        seg[cc] = np.where(left_distance[cc] < right_distance[cc], 2, 41)
    return seg


def _fill_preclassified(wm: np.ndarray, seg: np.ndarray, voxel_xsize: float,
                        cc_cut_mask: np.ndarray | None) -> np.ndarray:

    image = wm.copy()
    if cc_cut_mask is not None:
        if cc_cut_mask.shape != wm.shape:
            raise ValueError("cc_cut_mask shape must match wm")
        image[np.asarray(cc_cut_mask, dtype=bool) & np.isin(image, (200, 210, 220, 230, 240))] = 0
    image[np.isin(seg, _ERASE)] = 0
    left = np.isin(seg, _LEFT)
    right = np.isin(seg, _RIGHT)
    control = left | right
    fill = np.zeros(wm.shape, dtype=np.uint8)
    fill[left] = 255
    fill[right] = 127

    active = (image >= 5) | np.isin(seg, (25, 57, *_WMSA))
    if not np.any(control) or not np.any(active):
        return np.zeros(wm.shape, dtype=np.uint8)
    distances = ndi.distance_transform_cdt(~control, metric="chessboard")
    for radius in range(1, int(distances[active].max()) + 1):
        coordinates = np.argwhere(distances == radius)
        _voronoi_round(fill, distances, coordinates, radius)

    _edited_on_votes(fill, image, seg, voxel_xsize)
    active &= image != 1
    left_mask = active & (fill != 127)
    right_mask = active & (fill == 127)
    left_mask = _largest_then_fill_holes(left_mask)
    right_mask = _largest_then_fill_holes(right_mask)
    result = np.zeros(wm.shape, dtype=np.uint8)
    result[right_mask] = 127
    result[left_mask] = 255
    return result


def fill_with_aseg(wm: np.ndarray, aseg: np.ndarray, voxel_xsize: float = 1.0,
                   cc_cut_mask: np.ndarray | None = None) -> np.ndarray:
    """Return the 0/127/255 aseg-guided fill from FreeSurfer CRS voxel arrays."""
    if wm.shape != aseg.shape or wm.ndim != 3 or wm.dtype != np.uint8:
        raise ValueError("wm and aseg must be equal-shape 3D volumes, wm uint8")
    seg = _replace_cc_with_wm(aseg)
    return _fill_preclassified(wm, seg, voxel_xsize, cc_cut_mask)
