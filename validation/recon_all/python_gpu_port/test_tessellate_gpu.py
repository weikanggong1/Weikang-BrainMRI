"""Small exact-order check for FreeSurfer's quad tessellation."""

import numpy as np

from fnit.recon_all.tessellate_gpu import tessellate_quads


def test_single_interior_voxel_quad_order():
    volume = np.zeros((3, 3, 3), dtype=np.uint8)
    volume[1, 1, 1] = 255
    vertices, quads = tessellate_quads(volume, 255, np.eye(4), device="cpu")
    expected_vertices = np.array([
        [0.5, 0.5, 0.5], [1.5, 0.5, 0.5],
        [0.5, 1.5, 0.5], [1.5, 1.5, 0.5],
        [0.5, 0.5, 1.5], [1.5, 0.5, 1.5],
        [0.5, 1.5, 1.5], [1.5, 1.5, 1.5],
    ], dtype=np.float32)
    expected_quads = np.array([
        [0, 1, 3, 2], [0, 4, 5, 1], [0, 2, 6, 4],
        [1, 5, 7, 3], [2, 3, 7, 6], [4, 6, 7, 5],
    ], dtype=np.int32)
    np.testing.assert_array_equal(vertices, expected_vertices)
    np.testing.assert_array_equal(quads, expected_quads)
