import numpy as np

from fnit.recon_all.segstats_wmparc_python import _statistics


def test_uniform_intensity_has_integer_partial_volumes():
    seg = np.zeros((8, 8, 8), dtype=np.int32)
    seg[2:5, 2:6, 2:6] = 3001
    seg[5:7, 2:6, 2:6] = 4001
    intensity = np.full(seg.shape, 100, dtype=np.uint8)
    rows = _statistics(seg, intensity, [3001, 4001], 1.0)
    assert [(r[0], r[1], r[2]) for r in rows] == [
        (3001, 48, 48.0), (4001, 32, 32.0)]
    assert all((r[3], r[4], r[5], r[6]) == (100.0, 0.0, 100.0, 100.0)
               for r in rows)


def test_partial_volume_uses_neighbor_on_other_intensity_side():
    seg = np.zeros((20, 20, 20), dtype=np.int32)
    seg[2:10, 2:18, 2:18] = 3001
    seg[10:18, 2:18, 2:18] = 4001
    intensity = np.full(seg.shape, 20, dtype=np.uint8)
    intensity[seg == 3001] = 100
    intensity[seg == 4001] = 60
    intensity[9, 2:18, 2:18] = 75
    rows = _statistics(seg, intensity, [3001, 4001], 1.0)
    volumes = {label: volume for label, _, volume, *_ in rows}
    assert volumes[3001] < np.count_nonzero(seg == 3001)
    assert volumes[4001] > np.count_nonzero(seg == 4001)
