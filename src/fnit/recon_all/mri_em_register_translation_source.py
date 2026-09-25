"""Isolated source-order float32 translation grid for fixed-T1 registration."""

import numpy as np

from .mri_em_register_search_jit import log_sample_probability_jit


def _grid_values(lower, upper, steps):
    # The C++ operands and division are float; loop coordinates are double.
    delta = float(np.float32(
        np.float32(np.float32(upper) - np.float32(lower)) / np.float32(steps)))
    value = float(np.float32(lower))
    while value <= float(np.float32(upper)):
        yield value
        value += delta


def find_optimal_translation_source(samples, source, base_transform):
    matrix = np.asarray(base_transform, np.float32).copy()
    lower, upper = np.float32(-200.), np.float32(200.)
    history = []
    maximum = log_sample_probability_jit(samples, source, matrix)
    for _ in range(8):
        best = (0., 0., 0.)
        for x in _grid_values(lower, upper, 19):
            for y in _grid_values(lower, upper, 19):
                for z in _grid_values(lower, upper, 19):
                    trial = matrix.copy()
                    trial[:3, 3] += np.asarray((x, y, z), np.float32)
                    score = log_sample_probability_jit(samples, source, trial)
                    if score > maximum:
                        maximum, best = score, (x, y, z)
        matrix[:3, 3] += np.asarray(best, np.float32)
        maximum = log_sample_probability_jit(samples, source, matrix)
        history.append((maximum, best, matrix.copy()))
        # mean and quarter width are stored as double after float arithmetic.
        middle = float(np.float32(
            np.float32(lower + upper) / np.float32(2.)))
        quarter_width = float(np.float32(
            np.float32(upper - lower) / np.float32(4.)))
        lower = np.float32(middle - quarter_width)
        upper = np.float32(middle + quarter_width)
    return matrix, history
