"""Relabel small connected GCSA annotation islands."""

from __future__ import annotations

import numpy as np

from .gcsa_gibbs import GibbsModel


def vertex_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    triangles = np.asarray(vertices, np.float64)[faces]
    face_third = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                          triangles[:, 2] - triangles[:, 0]), axis=1) / 6
    area = np.zeros(len(vertices), np.float64)
    np.add.at(area, np.asarray(faces).ravel(), np.repeat(face_third, 3))
    return area


def annotated_segments(labels: np.ndarray, neighbors: list[list[int]]) -> list[list[int]]:
    marked = np.zeros(len(labels), bool)
    segments = []
    for start in range(len(labels)):
        if marked[start] or labels[start] == 0:
            continue
        annotation = labels[start]
        area = [start]
        marked[start] = True
        for vertex in area:
            for neighbor in neighbors[vertex]:
                if not marked[neighbor] and labels[neighbor] == annotation:
                    marked[neighbor] = True
                    area.append(neighbor)
        segments.append(area)
    return segments


def relabel_islands(model: GibbsModel, area: np.ndarray,
                    *, max_iterations: int = 5, min_area_fraction: float = 0.1) -> list[dict]:
    """Follow ``GCSArelabelIslands`` and return per-iteration counts."""
    labels = model.labels
    history = []
    for iteration in range(max_iterations + 2):
        segments = annotated_segments(labels, model.neighbors)
        segment_area = [float(area[vertices].sum()) for vertices in segments]
        available = np.ones(len(segments), bool)
        marked = np.zeros(len(labels), bool)
        changed = 0
        deleted = 0
        for i, vertices in enumerate(segments):
            if not available[i]:
                continue
            annotation = int(labels[vertices[0]])
            group = [j for j in range(i, len(segments))
                     if available[j] and int(labels[segments[j][0]]) == annotation]
            max_area = max(segment_area[j] for j in group)
            for j in group:
                if segment_area[j] < min_area_fraction * max_area:
                    deleted += 1
                    for vertex in segments[j]:
                        if marked[vertex]:
                            continue
                        best = int(labels[vertex])
                        max_ll = -100000000
                        for neighbor in model.neighbors[vertex]:
                            candidate = int(labels[neighbor])
                            if candidate == annotation:
                                continue
                            ll = int(model.neighborhood_log_likelihood(vertex, candidate))
                            if ll > max_ll or best == labels[vertex]:
                                max_ll = ll
                                best = candidate
                        if best != labels[vertex]:
                            labels[vertex] = best
                            marked[vertex] = True
                            changed += 1
                available[j] = False
        history.append({"iteration": iteration, "segments": len(segments),
                        "deleted": deleted, "changed": changed})
        if changed == 0 or iteration >= max_iterations:
            break
    return history
