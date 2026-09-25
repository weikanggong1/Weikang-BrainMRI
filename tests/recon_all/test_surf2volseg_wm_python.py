import numpy as np

from fnit.recon_all import surf2volseg_wm_python as wm


def test_label_wm_uses_side_and_five_mm_limit(monkeypatch):
    seg = np.array([2, 41, 77, 77, 87, 42], dtype=np.int32)[:, None, None]
    vox2ras = np.eye(4)
    vox2ras[0, 3] = -2.0
    calls = []

    def nearest(ras, xyz, faces, cortex_vertices, sign):
        calls.append((len(ras), sign))
        distance = np.where(ras[:, 0] == 0, 6.0, 4.0)
        return distance, np.zeros(len(ras), dtype=np.int64)

    monkeypatch.setattr(wm, "_nearest_with_dot", nearest)
    hemispheres = {
        "lh": (np.zeros((1, 3)), np.zeros((0, 3), dtype=np.int64),
               np.array([0]), np.array([12])),
        "rh": (np.zeros((1, 3)), np.zeros((0, 3), dtype=np.int64),
               np.array([0]), np.array([9])),
    }

    result = wm.label_wm_voxels(seg, vox2ras, hemispheres)

    np.testing.assert_array_equal(result[:, 0, 0], [3012, 4009, 5001, 4009, 5002, 42])
    assert calls == [(2, -1), (2, -1)]
