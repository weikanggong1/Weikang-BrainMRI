"""Small checks for the isolated FreeSurfer N4 CPU replay."""

import gzip

import nibabel as nib
import numpy as np
import pytest

pytest.importorskip("SimpleITK")
from fnit.recon_all.n4_sitk import _to_uchar, correct_volume


def test_uchar_rounding_matches_freesurfer():
    values = np.array([-1, 0, 0.49, 0.5, 1.5, 254.5, 255, 300], dtype=np.float32)
    np.testing.assert_array_equal(_to_uchar(values), [0, 0, 0, 1, 2, 255, 255, 255])


def test_replay_writes_uchar_mgz_with_input_geometry(tmp_path):
    xyz = np.indices((32, 32, 32), dtype=np.float32)
    radius = np.sqrt(np.square(xyz - 15.5).sum(axis=0))
    data = np.where(radius < 13, np.where(radius < 7, 110, 75), 0).astype(np.uint8)
    affine = np.diag([1.2, 1.2, 1.2, 1.0])
    source, output = tmp_path / "orig.mgz", tmp_path / "nu0.mgz"
    nib.save(nib.MGHImage(data, affine), source)

    correct_volume(source, output)

    result = nib.load(output)
    assert result.shape == data.shape
    assert result.get_data_dtype() == np.dtype("uint8")
    np.testing.assert_allclose(result.affine, nib.load(source).affine)
    assert np.count_nonzero(np.asarray(result.dataobj)[data == 0]) == 0
    source_raw, output_raw = (gzip.decompress(path.read_bytes()) for path in (source, output))
    first, last = int(result.header.get_data_offset()), int(result.header.get_data_offset()) + data.size
    assert output_raw[:first] == source_raw[:first]
    assert output_raw[last:] == source_raw[last:]
