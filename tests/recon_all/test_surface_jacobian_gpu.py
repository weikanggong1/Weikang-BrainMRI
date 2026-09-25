import numpy as np
import pytest

from fnit.recon_all.surface_jacobian_gpu import jacobian_values


def test_uniform_scale_has_unit_jacobian():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0],
                         [0, 0, 1]], dtype=np.float32)
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    values = jacobian_values(vertices, 2.5 * vertices, faces, device="cpu")
    np.testing.assert_allclose(values, 1.0, rtol=1e-6)


def test_mismatched_vertex_count_is_rejected():
    xyz = np.zeros((4, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="same"):
        jacobian_values(xyz, xyz[:3], np.array([[0, 1, 2]]), device="cpu")
