"""Check source-ordered ACJ writes where two amygdalae share cortex."""

import numpy as np

from fnit.recon_all.wm_edits_python import amygdala_cortex_junction


def test_amygdala_overlap_uses_last_source_voxel():
    aseg = np.zeros((5, 5, 5), dtype=np.int16)
    aseg[2, 2, 2] = 3
    aseg[2, 2, 1] = 18
    aseg[2, 2, 3] = 54
    assert amygdala_cortex_junction(aseg)[2, 2, 2] == 7031
    aseg[2, 2, 1], aseg[2, 2, 3] = 54, 18
    assert amygdala_cortex_junction(aseg)[2, 2, 2] == 7030


def test_boundary_amygdala_is_not_scanned():
    aseg = np.zeros((5, 5, 5), dtype=np.int16)
    aseg[0, 2, 2] = 18
    aseg[1, 2, 2] = 3
    assert amygdala_cortex_junction(aseg)[1, 2, 2] == 0
