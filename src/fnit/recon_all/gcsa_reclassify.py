"""Seeded asynchronous GCSA Gibbs label reclassification."""

from __future__ import annotations

import numpy as np

from .gcsa_gibbs import GibbsModel
from .gcsa_initial import InitialAtlas
from .gcsa_permutation import VnlRandom, vertex_permutation


def reclassify_gibbs(model: GibbsModel, atlas: InitialAtlas, *, seed: int = 1234,
                     max_iterations: int | None = None,
                     snapshot=None) -> list[dict]:
    """Run ``GCSAreclassifyUsingGibbsPriors`` on mapped surface vertices."""
    labels = model.labels
    mark = np.ones(len(labels), np.uint8)
    random = VnlRandom(seed)
    random.open_ran1()  # setRandomSeed primes OpenRan1 once.
    history = []
    iteration = 0
    while True:
        changed = 0
        examined = 0
        for vertex in vertex_permutation(random, len(labels)):
            if mark[vertex] == 0:
                continue
            mark[vertex] = 0
            examined += 1
            choices = atlas.prior_nodes[int(model.prior_indices[vertex])]
            if len(choices) <= 1:
                continue
            old = int(labels[vertex])
            best = old
            maximum = model.neighborhood_log_likelihood(int(vertex), old)
            for candidate, _ in choices:
                likelihood = model.neighborhood_log_likelihood(int(vertex), candidate)
                if likelihood > maximum:
                    maximum = likelihood
                    best = candidate
            if best != old:
                labels[vertex] = best
                mark[vertex] = 1
                changed += 1
        iteration += 1
        history.append({"iteration": iteration - 1, "changed": changed,
                        "examined": examined})
        if snapshot is not None:
            snapshot(iteration, labels)
        if changed == 0 or (max_iterations is not None and iteration >= max_iterations):
            break
        for vertex in np.flatnonzero(mark == 1):
            for neighbor in model.neighbors[vertex]:
                if mark[neighbor] != 1:
                    mark[neighbor] = 2
    return history
