"""Topology and intensity tie behavior of the fixed pretess calls."""

import numpy as np

from fnit.recon_all.pretess_python import pretess_values


def test_xy_edge_selects_brighter_bridge():
    segmentation = np.zeros((4, 4, 4), dtype=np.uint8)
    segmentation[1, 1, 1] = 255
    segmentation[2, 2, 1] = 255
    norm = np.zeros_like(segmentation)
    norm[1, 2, 1] = 200
    norm[2, 1, 1] = 100
    result, edits = pretess_values(segmentation, norm, 255)
    assert edits == 1
    assert result[1, 2, 1] == 255
    assert result[2, 1, 1] == 0


def test_wm_threshold_and_equal_intensity_tie():
    segmentation = np.zeros((4, 4, 4), dtype=np.uint8)
    segmentation[1, 1, 1] = 110
    segmentation[2, 2, 1] = 255
    norm = np.zeros_like(segmentation)
    result, edits = pretess_values(segmentation, norm, "wm")
    assert edits == 1
    assert result[1, 1, 1] == 110
    assert result[2, 2, 1] == 255
    assert result[1, 2, 1] == 0
    assert result[2, 1, 1] == 215
