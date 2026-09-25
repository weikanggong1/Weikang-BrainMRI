"""Isolated Numba candidate for the fixed-T1 GCA search score.

The exhaustive search still uses the reference Python candidate order. This
module only replaces its per-candidate sample likelihood calculation.
"""

import math

import numpy as np
from numba import njit

from .mri_em_register import StableSamples, _vnl_affine_inverse


@njit(cache=True)
def _sample_log_values(coordinates, means, variances, priors, source, matrix):
    values = np.empty(len(means), np.float64)
    width, height, depth = source.shape
    for sample in range(len(means)):
        voxel = np.empty(3, np.int32)
        for axis in range(3):
            value = np.float32(0.0)
            for source_axis in range(3):
                value = np.float32(value + np.float32(
                    matrix[axis, source_axis] * np.float32(coordinates[sample, source_axis])
                ))
            value = np.float32(value + matrix[axis, 3])
            if value < 0:
                voxel[axis] = math.ceil(float(value) - 0.5)
            else:
                voxel[axis] = math.floor(float(value) + 0.5)
        x, y, z = voxel[0], voxel[1], voxel[2]
        if x < 0 or y < 0 or z < 0 or x >= width or y >= height or z >= depth:
            values[sample] = -1000000.0
            continue
        difference = np.float32(np.float32(source[x, y, z]) - means[sample])
        squared = np.float32(difference * difference)
        mahalanobis = np.float32(squared / variances[sample])
        likelihood = (-math.log(math.sqrt(float(variances[sample])))
                      - 0.5 * float(mahalanobis) + math.log(float(priors[sample])))
        values[sample] = max(likelihood, -6.0)
    return values


def log_sample_probability_jit(samples: StableSamples, source: np.ndarray,
                               source_to_atlas: np.ndarray) -> float:
    """Evaluate the same source-order score with a compiled sample loop."""

    inverse = _vnl_affine_inverse(source_to_atlas)
    prior_to_source = inverse @ np.diag(np.array([2, 2, 2, 1], np.float32))
    values = _sample_log_values(samples.coordinates, samples.means,
                                samples.variances, samples.priors, source,
                                prior_to_source)
    return float(np.float32(np.float32(np.sum(values, dtype=np.float64)) /
                            np.float32(len(values))))
