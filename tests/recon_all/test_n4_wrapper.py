"""Checks for the FreeSurfer N4 wrapper's numeric and metadata contracts."""

import gzip
from decimal import Decimal

import nibabel as nib
import numpy as np

from fnit.recon_all.n4_wrapper import (
    _UNKNOWN_TAG_NATIVE,
    _UNKNOWN_TAG_WITH_NUL,
    global_mean_scale,
    normalize_n4_footer,
)


def test_global_mean_uses_segstats_five_decimal_values():
    original = np.full((1, 1, 1), 7.0559239983558655, dtype=np.float64)
    corrected = np.full((1, 1, 1), 6.202682375907898, dtype=np.float64)
    assert global_mean_scale(original, corrected) == float(Decimal("7.05592") / Decimal("6.20268"))


def test_n4_footer_matches_native_unknown_tag_encoding(tmp_path):
    path = tmp_path / "nu0.mgz"
    nib.save(nib.MGHImage(np.arange(8, dtype=np.uint8).reshape(2, 2, 2), np.eye(4)), str(path))
    raw = gzip.decompress(path.read_bytes()) + _UNKNOWN_TAG_WITH_NUL
    path.write_bytes(gzip.compress(raw, mtime=0))

    assert normalize_n4_footer(path)
    assert gzip.decompress(path.read_bytes()) == raw.removesuffix(_UNKNOWN_TAG_WITH_NUL) + _UNKNOWN_TAG_NATIVE
    assert not normalize_n4_footer(path)
