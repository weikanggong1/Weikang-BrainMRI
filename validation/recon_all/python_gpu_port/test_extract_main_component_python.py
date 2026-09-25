"""Quad splitting and connected-component choices from FreeSurfer 8.2."""

import numpy as np

from fnit.recon_all.extract_main_component_python import (
    largest_component, quad_triangles,
)


def test_free_surfer_quad_split_uses_two_vertex_numbers():
    quads = np.array([[0, 1, 5, 4], [0, 51, 52, 1]], dtype=np.int32)
    triangles = quad_triangles(quads)
    np.testing.assert_array_equal(triangles, np.array([
        [0, 1, 4], [5, 4, 1], [0, 51, 52], [0, 52, 1],
    ], dtype=np.int32))


def test_equal_sized_components_keep_first_vertex_component():
    vertices = np.arange(18, dtype=np.float32).reshape(6, 3)
    faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
    kept_vertices, kept_faces, count = largest_component(vertices, faces)
    assert count == 2
    np.testing.assert_array_equal(kept_vertices, vertices[:3])
    np.testing.assert_array_equal(kept_faces, faces[:1])
