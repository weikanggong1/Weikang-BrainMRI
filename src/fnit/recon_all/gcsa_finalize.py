"""FreeSurfer GCSA annotation mode filter and cortex mask correction."""

from __future__ import annotations

import numpy as np

from .gcsa_aseg import _annotation
from .gcsa_initial import InitialAtlas, initial_label
from .surface_curvature_gpu import _neighbours


def ordered_neighbors(faces: np.ndarray, count: int) -> list[list[int]]:
    """Build the FreeSurfer face-order one-ring adjacency."""
    neighbours = [[] for _ in range(count)]
    for a, b, c in np.asarray(faces, np.int32):
        for vertex, previous, following in ((a, c, b), (b, a, c), (c, b, a)):
            row = neighbours[vertex]
            if previous not in row:
                row.append(int(previous))
            if following not in row:
                row.append(int(following))
    return neighbours


def mode_filter_annotations(labels: np.ndarray, faces: np.ndarray,
                            color_table: dict[int, tuple[str, int, int, int, int]],
                            iterations: int = 10) -> np.ndarray:
    """Apply the two-ring synchronous ``MRISmodeFilterAnnotations`` vote."""
    annotations = np.zeros(max(color_table) + 1, dtype=np.int32)
    for index, entry in color_table.items():
        annotations[index] = _annotation(entry)
    sorted_order = np.argsort(annotations)
    sorted_annotations = annotations[sorted_order]
    out = np.asarray(labels, dtype=np.int32).copy()
    _, _, neighbours, valid = _neighbours(np.asarray(faces, np.int32), len(out))
    rows = np.repeat(np.arange(len(out), dtype=np.int64), valid.sum(axis=1))
    columns = neighbours[valid]
    nlabels = len(annotations)
    for _ in range(iterations):
        positions = np.searchsorted(sorted_annotations, out)
        matched = (positions < len(sorted_annotations)) & (
            sorted_annotations[np.minimum(positions, len(sorted_annotations) - 1)] == out)
        indices = np.where(matched, sorted_order[np.minimum(positions, len(sorted_order) - 1)], -1)
        selected = indices[columns] >= 0
        counts = np.bincount(rows[selected] * nlabels + indices[columns[selected]],
                             minlength=len(out) * nlabels).reshape(len(out), nlabels)
        present = indices >= 0
        counts[np.flatnonzero(present), indices[present]] += 1
        maxima = counts.max(axis=1)
        best = np.argmax(counts[:, 1:], axis=1) + 1
        best = np.where((maxima == 0) | (present & (counts[np.arange(len(out)),
                                                       indices.clip(min=0)] == maxima)),
                        indices.clip(min=0), best)
        out = annotations[best]
    return out


def apply_cortex_label(labels: np.ndarray, cortex_vertices: np.ndarray,
                       atlas: InitialAtlas, classifier_indices: np.ndarray,
                       prior_indices: np.ndarray, feature: np.ndarray,
                       faces: np.ndarray) -> np.ndarray:
    """Clear noncortex IDs and reclassify excluded cortical annotations."""
    out = np.asarray(labels, dtype=np.int32).copy()
    cortex = np.zeros(len(out), dtype=bool)
    cortex[np.asarray(cortex_vertices, dtype=np.int64)] = True
    out[~cortex] = 0
    excluded = {_annotation(entry) for entry in atlas.color_table.values()
                if entry[0].lower() in ("medial_wall", "unknown", "corpuscallosum")}
    pending = cortex & ((out <= 0) | np.isin(out, list(excluded)))
    for vertex in np.flatnonzero(pending):
        classifier = atlas.classifier_nodes[int(classifier_indices[vertex])]
        prior = atlas.prior_nodes[int(prior_indices[vertex])]
        choices = [entry for entry in initial_label(classifier, prior,
                                                   float(feature[vertex]))[1]
                   if entry[0] not in excluded]
        if choices:
            out[vertex] = max(choices, key=lambda entry: entry[2])[0]
            pending[vertex] = False

    if np.any(pending):
        neighbours = ordered_neighbors(faces, len(out))
        while np.any(pending):
            changed = 0
            for vertex in np.flatnonzero(pending):
                for neighbor in neighbours[vertex]:
                    if (not pending[neighbor] and out[neighbor] > 0
                            and out[neighbor] not in excluded):
                        out[vertex] = out[neighbor]
                        pending[vertex] = False
                        changed += 1
                        break
            if not changed:
                break
    return out
