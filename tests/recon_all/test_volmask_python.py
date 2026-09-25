import numpy as np

from fnit.recon_all.volmask_python import ribbon_arrays


FACES = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
                  [0, 1, 5], [0, 5, 4], [3, 7, 6], [3, 6, 2],
                  [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5]])


def cube(low, high, x_shift=0):
    a, b = low, high
    xyz = np.array([[a, a, a], [b, a, a], [b, b, a], [a, b, a],
                    [a, a, b], [b, a, b], [b, b, b], [a, b, b]], dtype=np.float32)
    xyz[:, 0] += x_shift
    return xyz, FACES


def test_nested_surfaces_make_bilateral_white_and_gray_masks():
    surfaces = {hemi: {"white": cube(1.5, 3.5, shift),
                       "pial": cube(0.5, 4.5, shift)}
                for hemi, shift in (("lh", 0), ("rh", 6))}
    ribbon, left, right = ribbon_arrays((12, 6, 6), np.eye(4), surfaces)
    assert ribbon[2, 2, 3] == 2
    assert ribbon[1, 2, 3] == 3
    assert ribbon[8, 2, 3] == 41
    assert ribbon[7, 2, 3] == 42
    assert ribbon[0, 0, 0] == 0
    assert left[1, 2, 3] == 1 and right[7, 2, 3] == 1
