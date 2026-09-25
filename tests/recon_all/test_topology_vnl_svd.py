"""Source-order ITK 5.3 VNL / FreeSurfer 8.2 float SVD regression."""

import numpy as np

from fnit.recon_all.topology_vnl_svd import svd_inverse_3, svdc_3


def test_lh_first_curvature_difference_matches_native_vnl_bits():
    gram = np.array([[73, 42, 21], [42, 84, 42], [21, 42, 73]], np.float32)
    u, w, v = svdc_3(gram)
    expected_u = np.array([
        [0xbf0542a9, 0x3f3504f4, 0x3ef50721],
        [0xbf2d42c9, 0x33167000, 0xbf3c756b],
        [0xbf0542aa, 0xbf3504f2, 0x3ef50722],
    ], np.uint32)
    expected_w = np.array([0x43149b68, 0x42500001, 0x41eb24c6], np.uint32)
    expected_v = np.array([
        [0xbf0542aa, 0x3f3504f4, 0x3ef50725],
        [0xbf2d42c7, 0xb1820000, 0xbf3c756c],
        [0xbf0542a9, 0xbf3504f4, 0x3ef50725],
    ], np.uint32)
    expected_inverse = np.array([
        [0x3c9d89d8, 0xbc1d89d8, 0x30000000],
        [0xbc1d89d8, 0x3cb04b05, 0xbc1d89d8],
        [0xb0c00000, 0xbc1d89da, 0x3c9d89d8],
    ], np.uint32)
    np.testing.assert_array_equal(u.view(np.uint32), expected_u)
    np.testing.assert_array_equal(w.view(np.uint32), expected_w)
    np.testing.assert_array_equal(v.view(np.uint32), expected_v)
    inverse = svd_inverse_3(gram)
    np.testing.assert_array_equal(inverse.view(np.uint32), expected_inverse)
    rhs = np.array([-1, -2, -1], np.float32)
    coefficients = np.zeros(3, np.float32)
    for row in range(3):
        for col in range(3):
            coefficients[row] = np.float32(
                coefficients[row] + np.float32(inverse[row, col] * rhs[col]))
    hessian = np.array([2 * coefficients[0], 2 * coefficients[1],
                        2 * coefficients[1], 2 * coefficients[2]], np.float32)
    np.testing.assert_array_equal(hessian.view(np.uint32), [
        0xb0800000, 0xbd430c32, 0xbd430c32, 0x32400000,
    ])
