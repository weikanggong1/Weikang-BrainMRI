import gzip
import struct

import nibabel as nib
import numpy as np

from fnit.recon_all.fill_cutting_plane_python import (
    _colortable_tag,
    compute_cc_cut_mask,
    read_vox_to_tal_lta,
    save_filled_mgz,
)


def test_talairach_midline_cut_extends_behind_cc():
    wm = np.zeros((21, 21, 21), dtype=np.uint8)
    wm[7:14, 4:13, 4:13] = 110
    aseg = np.full(wm.shape, 41, dtype=np.int32)
    aseg[:11] = 2

    mask, seed = compute_cc_cut_mask(wm, aseg, np.eye(4))

    expected = np.zeros(wm.shape, dtype=bool)
    expected[10, 4:, 4:13] = True
    assert seed[0] == 10
    np.testing.assert_array_equal(mask, expected)


def test_colortable_is_embedded_in_mgz(tmp_path):
    wm = np.zeros((4, 4, 4), dtype=np.uint8)
    source = tmp_path / "wm.mgz"
    output = tmp_path / "filled.mgz"
    lut = tmp_path / "SubCorticalMassLUT.txt"
    lut.write_text("0 Unknown 0 0 0 0\n127 Right-SubCorticalMass 0 180 0 0\n"
                   "255 Left-SubCorticalMass 0 0 180 0\n")
    nib.save(nib.MGHImage(wm, np.eye(4)), str(source))
    filled = wm.copy()
    filled[1, 1, 1] = 255

    save_filled_mgz(source, output, filled, lut)

    actual = nib.load(str(output))
    np.testing.assert_array_equal(np.asanyarray(actual.dataobj), filled)
    np.testing.assert_array_equal(actual.affine, nib.load(str(source)).affine)
    raw = gzip.decompress(output.read_bytes())
    tag = _colortable_tag(lut)
    assert tag in raw[284 + wm.size:]
    assert struct.unpack_from(">iiii", tag) == (1, -2, 256, len(str(lut)) + 1)


def test_lta_reads_type_zero_and_destination_cras(tmp_path):
    lta = tmp_path / "talairach.lta"
    lta.write_text("type = 0 # LINEAR_VOX_TO_VOX\n1 4 4\n"
                   "1 0 0 2\n0 1 0 3\n0 0 1 4\n0 0 0 1\n"
                   "dst volume info\nvalid = 1\ncras = 5 6 7\n")

    matrix, cras = read_vox_to_tal_lta(lta)

    np.testing.assert_array_equal(matrix[:3, 3], [2, 3, 4])
    np.testing.assert_array_equal(cras, [5, 6, 7])
