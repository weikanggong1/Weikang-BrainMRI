"""Operation tests for the fixed cortical-label translation."""

import numpy as np

from fnit.recon_all.label_cortex_python import (
    _adjacency,
    _dilate,
    _erode,
    _largest_component,
    _nearest,
)


def test_nearest_voxel_rounding_and_outside_value():
    volume = np.arange(27, dtype=np.int16).reshape(3, 3, 3)
    points = np.array([[0.49, 0, 0], [0.5, 0, 0], [2.2, 1, 1], [5, 1, 1]])
    np.testing.assert_array_equal(_nearest(volume, points), [0, 9, 22, 0])


def test_surface_close_and_largest_component():
    faces = np.array([[0, 1, 2], [2, 1, 3], [4, 5, 6]], np.int32)
    graph = _adjacency(faces, 7)
    selected = np.array([True, True, False, True, True, False, False])
    closed = _erode(_dilate(selected, graph, 1), graph, 1)
    np.testing.assert_array_equal(closed, [True, True, True, True, True, True, True])
    np.testing.assert_array_equal(_largest_component(closed, graph),
                                  [True, True, True, True, False, False, False])
