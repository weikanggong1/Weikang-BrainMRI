"""Source-order first-defect old-edge segmentation for topology correction.

The caller supplies scored and sorted old edges in corrected-surface index order.
This bounded primitive stops before FreeSurfer's genetic patch search.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from .topology_preflight_python import _edge_intersects


@njit
def _overlaps(sphere: np.ndarray, edges: np.ndarray) -> np.ndarray:
    result = np.zeros((len(edges), len(edges)), np.bool_)
    for i in range(len(edges)):
        for j in range(i + 1, len(edges)):
            result[i, j] = result[j, i] = _edge_intersects(
                sphere, edges[i, 0], edges[i, 1], edges[j, 0], edges[j, 1])
    return result


def _clusters(edges: np.ndarray, overlaps: np.ndarray) -> tuple[list[list[int]], np.ndarray]:
    incident: list[list[int]] = [[] for _ in range(int(edges.max()) + 1)]
    for index, (a, b) in enumerate(edges):
        incident[a].append(index)
        incident[b].append(index)
    groups: list[list[int]] = []
    exclusions: list[set[int]] = []
    assigned = np.full(len(edges), -1, np.int32)

    def compatible(group: int, edge: int) -> bool:
        return (group >= 0 and edge not in exclusions[group] and
                not any(overlaps[edge, member] for member in groups[group]))

    def add(group: int, edge: int) -> None:
        groups[group].append(edge)
        assigned[edge] = group
        exclusions[group].update(np.flatnonzero(overlaps[edge]).tolist())

    for edge, (a, b) in enumerate(edges):
        nearby: list[int] = []
        for vertex in (a, b):
            for neighbor in incident[vertex]:
                group = int(assigned[neighbor])
                if group not in nearby and compatible(group, edge):
                    nearby.append(group)
        if not nearby:
            try:
                target = next(i for i, row in enumerate(groups) if not row)
            except StopIteration:
                target = len(groups)
                groups.append([])
                exclusions.append(set())
            add(target, edge)
        elif len(nearby) == 1:
            add(nearby[0], edge)
        else:
            for _ in range(1, len(nearby)):
                if len(groups[nearby[1]]) < len(groups[nearby[0]]):
                    nearby[0], nearby[1] = nearby[1], nearby[0]
            target = nearby[0]
            add(target, edge)
            for other in nearby[1:]:
                if any(member in exclusions[other] for member in groups[target]) or \
                        any(member in exclusions[target] for member in groups[other]):
                    continue
                for member in groups[other]:
                    assigned[member] = target
                groups[target].extend(groups[other])
                exclusions[target].update(exclusions[other])
                groups[other].clear()
                exclusions[other].clear()

    active = [index for index, row in enumerate(groups) if row]
    active.sort(key=lambda index: -len(groups[index]))
    remap = {old: new for new, old in enumerate(active)}
    reordered = [groups[old] for old in active]
    assigned = np.asarray([remap[int(x)] for x in assigned], np.int32)
    for group in range(3, len(reordered)):
        if group >= 10 or len(reordered[group]) < 5:
            for edge in reordered[group]:
                assigned[edge] = -1
            reordered[group] = []
    changed = True
    while changed:
        changed = False
        for edge, (a, b) in enumerate(edges):
            if assigned[edge] >= 0:
                continue
            target = -1
            size = -1
            for vertex in (a, b):
                for neighbor in incident[vertex]:
                    group = int(assigned[neighbor])
                    if group < 0:
                        continue
                    if size == -1:
                        target, size = group, len(reordered[group])
                    elif len(reordered[group]) < size:
                        target, size = group, len(reordered[target])
            if target >= 0:
                assigned[edge] = target
                reordered[target].append(edge)
                changed = True
    return reordered, assigned



def segment_intersecting_edges(
    sphere: np.ndarray, old_edges: np.ndarray, inside_vertices: np.ndarray,
) -> tuple[np.ndarray, list[list[int]], np.ndarray]:
    """Match the source's inside-first edge order and overlapping-edge groups."""
    inside = np.zeros(len(sphere), np.bool_)
    inside[inside_vertices] = True
    both_inside = inside[old_edges[:, 0]] & inside[old_edges[:, 1]]
    edges = np.ascontiguousarray(
        np.concatenate((old_edges[both_inside], old_edges[~both_inside])), np.int32,
    )
    groups, assigned = _clusters(edges, _overlaps(np.asarray(sphere, np.float32), edges))
    return edges, groups, assigned
