import gzip
import struct

import nibabel as nib
import numpy as np
import pytest
import torch

from fnit.recon_all.mri_binarize_gpu import binarize_labels, binarize_volume


def test_match_and_inv_for_vsinus_cortical_mask():
    labels = torch.tensor([0, 3, 42, 2, 3, 41]).reshape(2, 3, 1)
    assert binarize_labels(labels, (3, 42)).flatten().tolist() == [0, 1, 1, 0, 1, 0]
    inverted = binarize_labels(labels, (3, 42), invert=True)
    assert inverted.flatten().tolist() == [1, 0, 0, 1, 0, 1]
    assert inverted.dtype == torch.int32


def test_mgz_roundtrip_uses_int32_and_keeps_geometry(tmp_path):
    affine = np.array([[-1, 0, 0, 123], [0, 1, 0, -17], [0, 0, 1, 31], [0, 0, 0, 1]], dtype=float)
    labels = np.array([0, 3, 42, 2, 3, 41], dtype=np.int32).reshape(2, 3, 1)
    source, target = tmp_path / "synthseg.mgz", tmp_path / "mask.mgz"
    nib.save(nib.MGHImage(labels, affine), str(source))
    original_bytes = gzip.decompress(source.read_bytes())
    voxel_end = 284 + labels.nbytes
    grey = struct.pack(">ii", 3, 5) + b"Grey\0" + struct.pack(">iiii", 100, 100, 100, 0)
    white = struct.pack(">ii", 1, 6) + b"White\0" + struct.pack(">iiii", 255, 255, 255, 0)
    colortable = struct.pack(">iiiii", 1, -2, 43, 0, 2) + grey + white
    other_tags = struct.pack(">iq", 41, 7) + b"UNKNOWN" + struct.pack(">iqi", 43, 4, 0)
    original_bytes += colortable + other_tags
    source.write_bytes(gzip.compress(original_bytes, mtime=0))
    binarize_volume(source, target, (3, 42), invert=True, device="cpu")
    result = nib.load(str(target))
    np.testing.assert_array_equal(np.asarray(result.dataobj), np.isin(labels, (3, 42), invert=True))
    np.testing.assert_array_equal(result.affine, affine)
    assert result.get_data_dtype() == np.dtype(">i4")
    output_bytes = gzip.decompress(target.read_bytes())
    assert output_bytes[:284] == original_bytes[:284]
    canonical_colortable = (colortable[:12] + struct.pack(">i", 1) + b"\0" +
                            colortable[16:20] + white + grey)
    assert output_bytes[voxel_end:] == original_bytes[voxel_end:voxel_end + 20] + other_tags + canonical_colortable


def test_rejects_invalid_input(tmp_path):
    with pytest.raises(ValueError, match="at least one"):
        binarize_labels(torch.zeros(2, 2, 2), ())
    source = tmp_path / "source.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((2, 2, 2), dtype=np.uint8), np.eye(4)), str(source))
    with pytest.raises(ValueError, match="MGH/MGZ"):
        binarize_volume(source, tmp_path / "mask.mgz", (3,), device="cpu")
