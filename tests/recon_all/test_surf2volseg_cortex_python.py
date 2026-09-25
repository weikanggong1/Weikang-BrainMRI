import numpy as np

from fnit.recon_all.surf2volseg_cortex_python import (
    _nearest_with_dot, label_cortex_voxels,
)


def test_nearest_vertex_respects_normal_direction():
    xyz = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]])
    ids = np.arange(3)
    _, valid = _nearest_with_dot(np.array([[0, 0, 0.5]]), xyz, faces, ids, +1)
    _, invalid = _nearest_with_dot(np.array([[0, 0, -0.5]]), xyz, faces, ids, +1)
    assert valid[0] == 0
    assert invalid[0] == -1


def test_cortical_annotation_offsets():
    aseg = np.zeros((3, 3, 3), dtype=np.int32)
    aseg[0, 0, 0] = 3
    aseg[2, 0, 0] = 42
    faces = np.array([[0, 1, 2]])
    hemispheres = {}
    for hemi, x in (("lh", 0), ("rh", 2)):
        white = np.array([[x, 0, -0.1], [x + 1, 0, -0.1],
                          [x, 1, -0.1]], dtype=np.float32)
        pial = white.copy()
        pial[:, 2] = 0.1
        hemispheres[hemi] = (white, faces, pial, faces, np.arange(3),
                             np.array([4, 4, 4]))
    result = label_cortex_voxels(aseg, np.eye(4), hemispheres)
    assert result[0, 0, 0] == 1004
    assert result[2, 0, 0] == 2004
    a2009s = label_cortex_voxels(aseg, np.eye(4), hemispheres, (11100, 12100))
    assert a2009s[0, 0, 0] == 11104
    assert a2009s[2, 0, 0] == 12104
    assert aseg[0, 0, 0] == 3
