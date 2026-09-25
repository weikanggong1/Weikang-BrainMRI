import numpy as np

from fnit.recon_all.fill_aseg_python import fill_with_aseg


def test_separates_labeled_hemispheres():
    wm = np.zeros((9, 9, 9), dtype=np.uint8)
    aseg = np.zeros(wm.shape, dtype=np.int32)
    wm[2, 4, 4] = wm[6, 4, 4] = 110
    aseg[2, 4, 4] = 2
    aseg[6, 4, 4] = 41

    filled = fill_with_aseg(wm, aseg)

    assert filled[2, 4, 4] == 255
    assert filled[6, 4, 4] == 127
    assert np.count_nonzero(filled) == 2


def test_cut_mask_removes_saved_wm_marker():
    wm = np.zeros((9, 9, 9), dtype=np.uint8)
    aseg = np.zeros(wm.shape, dtype=np.int32)
    cut = np.zeros(wm.shape, dtype=bool)
    wm[4, 4, 4] = 200
    aseg[4, 4, 4] = 2
    cut[4, 4, 4] = True

    assert fill_with_aseg(wm, aseg)[4, 4, 4] == 255
    assert fill_with_aseg(wm, aseg, cc_cut_mask=cut)[4, 4, 4] == 0
