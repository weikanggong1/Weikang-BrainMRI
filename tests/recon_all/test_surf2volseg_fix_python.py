import numpy as np

from fnit.recon_all.surf2volseg_fix_python import fix_presurf_with_ribbon


def test_ribbon_rules_and_nearest_cortex_mask():
    aseg = np.zeros((5, 5, 5), dtype=np.int32)
    ribbon = np.zeros_like(aseg)
    aseg[1, 1, 1] = aseg[3, 1, 1] = 2
    ribbon[1, 2, 1] = ribbon[3, 2, 1] = 3
    aseg[1, 3, 1], ribbon[1, 3, 1] = 3, 2
    aseg[2, 3, 1], ribbon[2, 3, 1] = 77, 3
    aseg[3, 3, 1] = 8
    xyz = np.array([[1, 1, 1], [3, 1, 1]], dtype=np.float32)
    surfaces = [(xyz, np.array([True, False])) for _ in range(4)]
    result = fix_presurf_with_ribbon(aseg, ribbon, np.eye(4), surfaces)
    assert result[1, 1, 1] == 0
    assert result[3, 1, 1] == 2
    assert result[1, 2, 1] == 3
    assert result[3, 2, 1] == 0
    assert result[1, 3, 1] == 2
    assert result[2, 3, 1] == 3
    assert result[3, 3, 1] == 8
    assert aseg[1, 1, 1] == 2
