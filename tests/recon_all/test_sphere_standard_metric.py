"""The native reciprocal pass updates the first reverse duplicate in order."""

import numpy as np

from fnit.recon_all.sphere_standard_metric import average_standard_metric


def test_reciprocal_average_with_duplicate_neighbors() -> None:
    offsets = np.asarray([0, 3, 5], np.int64)
    neighbors = np.asarray([1, 1, 1, 0, 0], np.int32)
    distances = np.asarray([1, 2, 3, 4, 5], np.float32)
    result, matched, _ = average_standard_metric(offsets, neighbors, distances)
    np.testing.assert_array_equal(
        result, np.asarray([3.78125, 2.25, 2.625, 2.5625, 3.78125], np.float32))
    assert matched == 5
    np.testing.assert_array_equal(distances, [1, 2, 3, 4, 5])
