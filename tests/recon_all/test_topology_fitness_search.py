import numpy as np

from fnit.recon_all.topology_fitness_search import (
    compose_patch_fitness, rank_patch_fitness,
)


def test_source_rank_keeps_earlier_candidate_first_inside_fzero_tie():
    scores = np.asarray([-1.0, -1.0 + 5e-6, -2.0], np.float64)
    assert rank_patch_fitness(scores).tolist() == [0, 1, 2]


def test_invalid_patch_receives_source_face_penalty():
    valid = compose_patch_fitness(-1.0, -2.0, -3.0, -4.0, -5.0, True)
    invalid = compose_patch_fitness(-1.0, -2.0, -3.0, -4.0, -5.0, False)
    assert valid == -60.0
    assert invalid - valid == -10_000_000.0
