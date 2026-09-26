"""Pial medial-wall pinning and native-order intersection repair."""

from __future__ import annotations

import numpy as np

from .mris_remove_intersection_python import mark_intersections
from .place_surface_smoothing import _ordered_neighbors


def pin_medial_wall(pial: np.ndarray, white: np.ndarray, cortex_vertices: np.ndarray) -> np.ndarray:
    """Match MRISpinMedialWallToWhite on the ordered vertex arrays."""
    result = np.asarray(pial, dtype=np.float32).copy()
    outside = np.ones(len(result), dtype=np.bool_)
    outside[np.asarray(cortex_vertices, dtype=np.int64)] = False
    result[outside] = np.asarray(white, dtype=np.float32)[outside]
    return result


def repair_intersections(
    vertices: np.ndarray, faces: np.ndarray, ripped: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Replay MRISremoveIntersections with 100 soap-bubble steps per cycle."""
    result = np.asarray(vertices, dtype=np.float32).copy()
    faces = np.asarray(faces, dtype=np.int32)
    ripped = np.asarray(ripped, dtype=np.bool_)
    marked, count = mark_intersections(result, faces)
    if count == 0:
        return result, {"intersecting_faces_before": 0, "intersecting_faces_after": 0,
                        "marked_vertices": 0, "smoothing_cycles": 0}
    neighbors, valid, _ = _ordered_neighbors(faces, len(result))
    first_count, first_marked = count, int(marked.sum())
    best, minimum = result.copy(), len(result)
    old_count, no_progress, cycles = len(result), 0, 0
    smoothed = 0
    trace = [count]
    while count:
        if count > old_count or count == old_count and no_progress >= 0:
            no_progress += 1
            if no_progress > 15:
                break
        else:
            no_progress = 0 if count < old_count else no_progress + 1
            if count < minimum:
                minimum, best = count, result.copy()
        old_count = count
        moving = np.flatnonzero(marked & ~ripped)
        smoothed += len(moving)
        for _ in range(100):
            next_xyz = result.copy()
            for vertex in moving:
                x, y, z = (np.float32(value) for value in result[vertex])
                n = np.float32(1)
                for slot in range(neighbors.shape[1]):
                    if not valid[vertex, slot]:
                        continue
                    other = neighbors[vertex, slot]
                    if ripped[other]:
                        continue
                    x = np.float32(x + result[other, 0])
                    y = np.float32(y + result[other, 1])
                    z = np.float32(z + result[other, 2])
                    n = np.float32(n + np.float32(1))
                next_xyz[vertex, 0] = np.float32(x / n)
                next_xyz[vertex, 1] = np.float32(y / n)
                next_xyz[vertex, 2] = np.float32(z / n)
            result = next_xyz
        cycles += 1
        if cycles > 101:
            break
        marked, count = mark_intersections(result, faces)
        trace.append(count)
    if count > minimum:
        result = best
        _, count = mark_intersections(result, faces)
    return result, {"intersecting_faces_before": first_count,
                    "intersecting_faces_after": count,
                    "intersecting_faces_trace": trace,
                    "marked_vertices": first_marked,
                    "smoothed_vertices": smoothed,
                    "smoothing_cycles": cycles,
                    "smoothing_iterations": 100 * cycles}
