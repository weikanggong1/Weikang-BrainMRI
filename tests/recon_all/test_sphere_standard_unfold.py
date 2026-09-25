"""Regression for FreeSurfer's near-degenerate face-normal rule."""

import numpy as np

from fnit.recon_all.sphere_standard_unfold import _face_geometry


def test_face_normal_retains_cross_product_below_float_epsilon():
    vertices = np.array([[1, 0, 0], [1, 2e-4, 0], [1, 0, 2e-4]], np.float32)
    faces = np.array([[0, 1, 2]], np.int32)
    area, normal = _face_geometry(vertices, faces)
    assert np.isclose(area[0], 2e-8, rtol=1e-6)
    assert np.isclose(normal[0, 0], 4e-8, rtol=1e-6)
    assert normal[0, 1] == normal[0, 2] == 0
