"""Focused checks for the native gray/white statistics translation."""

import numpy as np

from fnit.recon_all.autodet_gwstats_python import _class_sigmas, format_autodet_stats


def test_white_clipping_precedes_border_sigma():
    brain = np.full((5, 5, 5), 50, dtype=np.uint8)
    wm = np.zeros_like(brain)
    wm[2, 2, 2:4] = 255
    brain[2, 2, 2] = 80
    brain[2, 2, 3] = 200
    white_std, gray_std = _class_sigmas(brain, wm)
    assert white_std == 15.0
    assert gray_std == 0.0


def test_native_text_spacing():
    stats = {"hemicode": 1, "white_border_hi": 108.0, "pial_border_hi": 40.0, "white_mode": 98.0, "use_mode": 1}
    assert format_autodet_stats(stats) == (
        "hemicode           1\n"
        "white_border_hi    108.000000\n"
        "pial_border_hi    40.000000\n"
        "white_mode 98.000000\n"
        "use_mode 1\n"
    )
