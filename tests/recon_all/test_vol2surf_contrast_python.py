import numpy as np

from fnit.recon_all.vol2surf_contrast_python import (
    _inverse_affine32, _transform_points32, _trilinear,
)


def test_trilinear_uses_xyz_order_and_native_boundary_rule():
    x, y, z = np.indices((3, 4, 5), dtype=np.float32)
    volume = x + 10 * y + 100 * z
    points = np.array([[0.25, 1.5, 2.75], [-0.4, 1.0, 1.0],
                       [-0.6, 1.0, 1.0]], dtype=np.float64)
    np.testing.assert_array_equal(_trilinear(volume, points, "cpu"),
                                  np.array([290.25, 110.0, 0.0], dtype=np.float32))


def test_float32_matrix_inverse_and_native_point_accumulation_order():
    matrix = np.array([[2, 0, 0, 4], [0, 4, 0, -8],
                       [0, 0, 8, 16], [0, 0, 0, 1]], dtype=np.float32)
    np.testing.assert_array_equal(_inverse_affine32(matrix),
                                  np.array([[.5, 0, 0, -2], [0, .25, 0, 2],
                                            [0, 0, .125, -2], [0, 0, 0, 1]],
                                           dtype=np.float32))
    matrix = np.array([[1, 1, 0, -16777216], [0, 1, 0, 0],
                       [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
    points = np.array([[16777216, 1, 0]], dtype=np.float32)
    np.testing.assert_array_equal(_transform_points32(points, matrix),
                                  np.array([[0, 1, 0]], dtype=np.float32))
