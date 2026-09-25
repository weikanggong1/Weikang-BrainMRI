import numpy as np

from fnit.recon_all.edit_wm_aseg_late_python import (
    apply_late_wm_edits,
    fill_seg_wm_from_core,
    fix_subcortical_mass_ha,
)


def test_fix_scm_zeros_amygdala_ilv_and_undilated_hippocampus():
    aseg = np.zeros((9, 9, 9), dtype=np.int16)
    aseg[2:7, 2:7, 2:7] = 17
    aseg[1, 1, 1] = 18
    aseg[1, 1, 2] = 5
    wm = np.full(aseg.shape, 250, dtype=np.uint8)
    result = fix_subcortical_mass_ha(wm, aseg)
    assert result[4, 4, 4] == 0
    assert result[2, 2, 2] == 250
    assert result[1, 1, 1] == result[1, 1, 2] == 0


def test_fill_seg_wm_propagates_to_neighbor_labeled_wm():
    aseg = np.zeros((9, 9, 9), dtype=np.int16)
    aseg[4, 4, 4] = aseg[5, 4, 4] = 2
    aseg[6, 4, 4] = 3
    wm = np.zeros(aseg.shape, dtype=np.uint8)
    result = fill_seg_wm_from_core(wm, aseg)
    assert result[4, 4, 4] == 250
    assert result[5, 4, 4] == 250
    assert result[6, 4, 4] == 0


def test_late_edits_keep_in_then_ento_and_acj():
    aseg = np.zeros((9, 9, 9), dtype=np.int16)
    aseg[4, 4, 4] = 18
    aseg[4, 4, 5] = 3
    entowm = np.zeros(aseg.shape, dtype=np.int16)
    entowm[3, 3, 3] = 3006
    original = np.zeros(aseg.shape, dtype=np.uint8)
    original[2, 2, 2] = 1
    result = apply_late_wm_edits(np.zeros_like(original), aseg, entowm, original)
    assert result[4, 4, 5] == 255
    assert result[3, 3, 3] == 255
    assert result[2, 2, 2] == 1
