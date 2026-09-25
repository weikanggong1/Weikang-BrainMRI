import numpy as np

from fnit.recon_all.relabel_hypointensities_python import relabel_hypointensities


def test_two_neighbor_passes_recover_cortex():
    aseg = np.zeros((5, 5, 5), dtype=np.int32)
    aseg[1, 2, 2] = 3
    aseg[2, 2, 2] = 78
    aseg[3, 2, 2] = 79
    xyz = np.array([[1, 2, 2], [2, 2, 2], [1, 3, 2], [1, 2, 3]], dtype=np.float32)
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    surfaces = {hemi: (xyz, faces) for hemi in ("lh", "rh")}
    result = relabel_hypointensities(aseg, np.eye(4), surfaces)
    np.testing.assert_array_equal(result[1:4, 2, 2], [3, 3, 3])
    np.testing.assert_array_equal(aseg[1:4, 2, 2], [3, 78, 79])
