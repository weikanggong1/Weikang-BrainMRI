"""Regression checks for source-order Fast Marching float precision."""

import numpy as np

from fnit.recon_all.normalization.normalize_aseg_ridge import (
    _update_value, signed_distance)


def test_fast_marching_starts_half_a_voxel_from_wm_boundary():
    mask = np.zeros((5, 5, 5), dtype=np.uint8)
    mask[2, 2, 2] = 1
    distance, _ = signed_distance(mask)
    assert distance[2, 2, 2] == np.float32(0.5)
    assert distance[3, 2, 2] == np.float32(-0.5)


def test_fast_marching_quadratic_uses_double_sqrt_before_float_rounding():
    distance = np.zeros((3, 3, 3), dtype=np.float32)
    a, b, c = 3.0939881801605225, 3.389066219329834, 3.450075626373291
    distance[0, 1, 1] = distance[2, 1, 1] = a
    distance[1, 0, 1] = distance[1, 2, 1] = b
    distance[1, 1, 0] = distance[1, 1, 2] = c
    state = np.zeros(distance.size, dtype=np.uint8)
    result = _update_value(13, distance.ravel(), state, 1, np.float32(512), 3, 3, 3)
    assert np.asarray(result, dtype=np.float32).view(np.uint32) == 1081572850
