"""Ordered one-ring and center-inclusive surface smoothing."""

import numpy as np

from fnit.recon_all.smooth_surface_python import (
    average_positions, ordered_neighbors,
)


def test_face_order_neighbors_and_one_step_average():
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    assert ordered_neighbors(faces, 4) == [
        [2, 1, 3], [0, 2], [1, 0, 3], [2, 0],
    ]
    vertices = np.array([[0, 0, 0], [2, 0, 0],
                         [2, 2, 0], [0, 2, 0]], dtype=np.float32)
    result = average_positions(vertices, faces, iterations=1)
    np.testing.assert_array_equal(result[0], [1, 1, 0])
    np.testing.assert_array_equal(result[1], np.array([4 / 3, 2 / 3, 0], dtype=np.float32))
