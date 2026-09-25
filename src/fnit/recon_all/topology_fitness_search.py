"""Source-order score composition and ranking for topology GA patch candidates.

Component likelihoods and candidate patches must be computed separately.
This does not perform mutation, crossover, or final topology repair.
"""

from __future__ import annotations

import numpy as np


def compose_patch_fitness(face_ll: float, vertex_ll: float, normal_dot_ll: float,
                          quadratic_curvature_ll: float, volume_unmri_ll: float,
                          valid_faces: bool) -> float:
    """Compose the fixed ``mris_fix_topology -ga`` fitness weights."""
    score = face_ll + vertex_ll
    score += 10.0 * volume_unmri_ll
    score += quadratic_curvature_ll
    score += normal_dot_ll
    if not valid_faces:
        score -= 10_000_000.0
    return score


def rank_patch_fitness(fitness: np.ndarray) -> np.ndarray:
    """Return pinned ``defectPatchRank`` ranks, including 1e-5 ties."""
    values = np.asarray(fitness, np.float64)
    ranks = np.zeros(len(values), np.int32)
    for index, value in enumerate(values):
        for other, peer in enumerate(values):
            if other == index:
                continue
            if abs(peer - value) < 1e-5:
                if index > other:
                    ranks[index] += 1
            elif peer > value:
                ranks[index] += 1
    return ranks


def invalid_modified_edges(topology_edges: np.ndarray, faces: np.ndarray,
                           modified_vertices: np.ndarray) -> int:
    """Count first-patch edges that would trigger ``mrisCheckDefectFaces``."""
    triangles = np.asarray(faces, np.int32)
    edge_rows = np.concatenate((triangles[:, [0, 1]], triangles[:, [1, 2]],
                                triangles[:, [2, 0]]), axis=0)
    edge_rows.sort(axis=1)
    unique, counts = np.unique(edge_rows, axis=0, return_counts=True)
    incidence = {tuple(edge): int(count) for edge, count in zip(unique, counts)}
    marked = set(np.asarray(modified_vertices, np.int32).tolist())
    checked = {tuple(sorted((int(a), int(b)))) for a, b in topology_edges
               if int(a) in marked or int(b) in marked}
    return sum(incidence.get(edge, 0) != 2 for edge in checked)
