"""One-feature GCSA Gibbs neighborhood likelihood on a surface mesh."""

from __future__ import annotations

import math

import numpy as np

from .gcsa_initial import InitialAtlas


class GibbsModel:
    def __init__(self, atlas: InitialAtlas, classifier_indices: np.ndarray,
                 prior_indices: np.ndarray, feature: np.ndarray,
                 original_vertices: np.ndarray, neighbors: list[list[int]],
                 principal_directions: np.ndarray, labels: np.ndarray):
        if atlas.gibbs_neighbours is None:
            raise ValueError("GCS atlas was read without Gibbs neighbor probabilities")
        self.labels = labels
        self.feature = feature
        self.classifier_indices = classifier_indices
        self.prior_indices = prior_indices
        self.classifier = [{label: (mean, variance)
                            for label, _, mean, variance in node}
                           for node in atlas.classifier_nodes]
        self.prior = [dict(zip((label for label, _ in node),
                               ((prior, directions) for (_, prior), directions
                                in zip(node, gibbs_node))))
                      for node, gibbs_node in zip(atlas.prior_nodes, atlas.gibbs_neighbours)]
        self.neighbors = neighbors
        xyz = np.asarray(original_vertices, np.float32)
        basis = np.asarray(principal_directions, np.float32)
        self.edge_slots = []
        for vertex, row in enumerate(neighbors):
            delta = xyz[row] - xyz[vertex]
            dot1 = delta @ basis[vertex, 0]
            dot2 = delta @ basis[vertex, 1]
            self.edge_slots.append(np.where(np.abs(dot1) > np.abs(dot2), 0, 1))

    def vertex_log_likelihood(self, vertex: int, input_value: float) -> float:
        label = int(self.labels[vertex])
        prior = self.prior[int(self.prior_indices[vertex])].get(label)
        classifier = self.classifier[int(self.classifier_indices[vertex])].get(label)
        if prior is None or classifier is None:
            return -10000000.0
        probability, directions = prior
        mean, variance = classifier
        diff = float(np.float32(mean - input_value))
        ll = -0.5 * diff * diff / variance - 0.5 * math.log(variance)
        for neighbor, slot in zip(self.neighbors[vertex], self.edge_slots[vertex]):
            chance = directions[int(slot)].get(int(self.labels[neighbor]))
            ll += math.log(chance) if chance is not None and chance != 0 else -10000000.0
        return ll + math.log(probability)

    def neighborhood_log_likelihood(self, vertex: int, candidate: int) -> float:
        old = self.labels[vertex]
        self.labels[vertex] = candidate
        input_value = float(self.feature[vertex])
        value = self.vertex_log_likelihood(vertex, input_value)
        for neighbor in self.neighbors[vertex]:
            value += self.vertex_log_likelihood(neighbor, input_value)
        self.labels[vertex] = old
        return value
