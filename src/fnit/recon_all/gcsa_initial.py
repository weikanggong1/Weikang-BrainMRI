"""Read the fixed FreeSurfer GCS atlas fields used by surface labeling."""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class InitialAtlas:
    classifier_nodes: list[tuple[tuple[int, int, float, float], ...]]
    prior_nodes: list[tuple[tuple[int, float], ...]]
    color_table: dict[int, tuple[str, int, int, int, int]]
    source_name: str
    average_variance: float
    minimum_determinant: float
    singular_count: int
    regularized_count: int
    gibbs_neighbours: list[tuple[tuple[dict[int, float], dict[int, float]], ...]] | None = None


def _vertex_count(ico_order: int) -> int:
    if not 0 <= ico_order <= 7:
        raise ValueError(f"Unsupported GCS icosahedron order: {ico_order}")
    return 10 * 4**ico_order + 2


def read_initial_atlas(path: str | Path, *, include_gibbs: bool = False) -> InitialAtlas:
    """Parse one-input GCSA classifier/prior fields and embedded color table.

    Matches ``GCSAread`` and its one-dimensional covariance regularization for
    the six FreeSurfer 8.2 DK, DKT and Destrieux GCS atlases. Each classifier
    entry is ``(annotation, training_count, mean_curvature, variance)``.
    """
    with open(path, "rb") as stream:
        def exact(size: int) -> bytes:
            value = stream.read(size)
            if len(value) != size:
                raise ValueError("Truncated GCS atlas")
            return value

        def i32() -> int:
            return struct.unpack(">i", exact(4))[0]

        def f32() -> float:
            return struct.unpack(">f", exact(4))[0]

        def scalar_matrix() -> float:
            if stream.readline().split() != [b"1", b"1", b"1"]:
                raise ValueError("Expected a 1x1 real GCS matrix")
            return struct.unpack("f", struct.pack("f", float(stream.readline().split()[0])))[0]

        if i32() & 0xFFFFFFFF != 0xABABCDCD:
            raise ValueError("Invalid GCS magic")
        ninputs, classifier_order, prior_order = i32(), i32(), i32()
        if ninputs != 1:
            raise ValueError(f"Only the fixed one-input GCS atlases are supported: {ninputs}")
        input_type, name_length = i32(), i32()
        name = exact(name_length).rstrip(b"\x00").decode("utf-8")
        navgs, flags = i32(), i32()
        if (input_type, navgs, flags) != (0, 5, 0):
            raise ValueError(f"Unexpected GCS curvature descriptor: {(input_type, navgs, flags)}")

        classifiers = []
        variance_sum = 0.0
        variance_count = 0
        for _ in range(_vertex_count(classifier_order)):
            nlabels, _total_training = i32(), i32()
            records = []
            for _ in range(nlabels):
                label, training = i32(), i32()
                mean, variance = scalar_matrix(), scalar_matrix()
                records.append((label, training, mean, variance))
                if (training == 0 and variance > 1) or training > 5:
                    variance_sum += variance
                    variance_count += 1
            classifiers.append(tuple(records))

        if not variance_count:
            raise ValueError("No trained classifier covariance in GCS atlas")
        average_variance = variance_sum / variance_count
        minimum_determinant = average_variance / 100.0
        fixed, regularized = 0, 0
        repaired = []
        for node in classifiers:
            labels = []
            for label, training, mean, variance in node:
                if variance <= 0:
                    fixed += 1
                    variance += average_variance
                elif (training < 4 and variance < 0.1) or (
                    variance < minimum_determinant and training < 8
                ):
                    regularized += 1
                    variance += average_variance
                labels.append((label, training, mean, variance))
            repaired.append(tuple(labels))

        priors = []
        gibbs = [] if include_gibbs else None
        for _ in range(_vertex_count(prior_order)):
            nlabels, _total_training = i32(), i32()
            records = []
            gibbs_records = []
            for _ in range(nlabels):
                records.append((i32(), f32()))
                directions = []
                for direction in range(4):
                    _total_neighbors, nneighbor_labels = i32(), i32()
                    if nneighbor_labels < 0:
                        raise ValueError("Negative GCS neighbor label count")
                    if include_gibbs and direction in (0, 2):
                        directions.append({i32(): f32() for _ in range(nneighbor_labels)})
                    else:
                        exact(8 * nneighbor_labels)
                if include_gibbs:
                    gibbs_records.append((directions[0], directions[1]))
            priors.append(tuple(records))
            if include_gibbs:
                gibbs.append(tuple(gibbs_records))

        if i32() != 1 or i32() != -2:
            raise ValueError("Expected a FreeSurfer v2 GCS color table")
        nentries, filename_length = i32(), i32()
        source_name = exact(filename_length).rstrip(b"\x00").decode("utf-8")
        color_table = {}
        for _ in range(i32()):
            index, length = i32(), i32()
            if not 0 <= index < nentries:
                raise ValueError("GCS color table index out of range")
            name = exact(length).rstrip(b"\x00").decode("utf-8")
            color_table[index] = (name, i32(), i32(), i32(), i32())
        if stream.read(1):
            raise ValueError("Unexpected trailing data in GCS atlas")

    return InitialAtlas(repaired, priors, color_table, source_name,
                        average_variance, minimum_determinant, fixed, regularized, gibbs)


def initial_label(classifier_node: tuple[tuple[int, int, float, float], ...],
                  prior_node: tuple[tuple[int, float], ...],
                  mean_curvature: float) -> tuple[int, list[tuple[int, float, float]]]:
    """Compute ``GCSANclassify`` for one curvature input at mapped ico nodes."""
    classifiers = {label: (mean, variance) for label, _, mean, variance in classifier_node}
    candidates = []
    for label, prior in prior_node:
        if label not in classifiers:
            continue
        mean, variance = classifiers[label]
        likelihood = math.exp(-0.5 * (mean - mean_curvature) ** 2 / variance) / math.sqrt(variance)
        candidates.append((label, likelihood, prior * likelihood))
    return (max(candidates, key=lambda entry: entry[2])[0] if candidates else -1,
            candidates)


def read_ico_vertices(path: str | Path):
    """Load the ordered vertex coordinates from a FreeSurfer ``ic*.tri``."""
    import numpy as np

    with open(path, encoding="ascii") as stream:
        count = int(stream.readline())
    vertices = np.loadtxt(path, skiprows=1, max_rows=count,
                          usecols=(1, 2, 3), dtype=np.float64)
    return 100.0 * vertices / np.linalg.norm(vertices, axis=1, keepdims=True)


def map_initial_nodes(sphere_vertices, classifier_ico, prior_ico):
    """Map sphere vertices to ico7 priors, then each prior to ico4 classifiers.

    The ico arrays are returned by :func:`read_ico_vertices`. The compiled
    implementation uses a spatial hash; this uses exact nearest neighbors.
    """
    import numpy as np
    from scipy.spatial import cKDTree

    source = np.asarray(sphere_vertices, dtype=np.float32)
    center = ((source.min(axis=0).astype(np.float64)
               + source.max(axis=0).astype(np.float64)) * 0.5).astype(np.float32)
    centered = (source - center).astype(np.float64)
    distance = np.sqrt(np.sum(centered * centered, axis=1, keepdims=True))
    sphere = (centered - (1.0 - 100.0 / distance) * centered).astype(np.float32)
    prior_indices = cKDTree(prior_ico).query(sphere)[1]
    classifier_tree = cKDTree(classifier_ico)
    distances, choices = classifier_tree.query(prior_ico[prior_indices], k=2)
    classifier_indices = choices[:, 0].copy()
    for vertex in np.flatnonzero(distances[:, 1] - distances[:, 0] <= 1e-9):
        tied = classifier_tree.query_ball_point(prior_ico[prior_indices[vertex]],
                                                distances[vertex, 0] + 1e-9)
        classifier_indices[vertex] = min(tied, key=lambda v: (*classifier_ico[v], v))
    return classifier_indices, prior_indices
