"""Source-order float32 average in standard sphere integration."""

import numpy as np

from fnit.recon_all.sphere_standard_average import average_standard_gradient


def test_standard_gradient_average_keeps_self_and_neighbor_order():
    gradient = np.array([[1, 2, 3], [3, 5, 7], [11, 13, 17]], np.float32)
    offsets = np.array([0, 2, 4, 6], np.int64)
    neighbors = np.array([1, 2, 0, 2, 0, 1], np.int32)
    result = average_standard_gradient(gradient, offsets, neighbors, 2)
    first = np.empty_like(gradient)
    second = np.empty_like(gradient)
    for src, dst in ((gradient, first), (first, second)):
        for vertex in range(3):
            value = src[vertex].copy()
            for index in range(offsets[vertex], offsets[vertex + 1]):
                value = np.float32(value + src[neighbors[index]])
            dst[vertex] = np.float32(value * np.float32(1 / 3))
    assert np.array_equal(result, second)
    assert np.array_equal(average_standard_gradient(gradient, offsets, neighbors, 0),
                          gradient)
