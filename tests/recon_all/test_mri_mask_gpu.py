import gzip

import nibabel as nib
import numpy as np
import pytest
import torch

from fnit.recon_all.mri_mask_gpu import mask_tensor, mask_volume


def test_threshold_and_inversion_follow_freesurfer_boundaries():
    image = torch.tensor([1, 2, 3, 4], dtype=torch.uint8).reshape(2, 2, 1)
    mask = torch.tensor([0, 1, 5, 6], dtype=torch.uint8).reshape(2, 2, 1)
    assert mask_tensor(image, mask).flatten().tolist() == [0, 2, 3, 4]
    assert mask_tensor(image, mask, threshold=5).flatten().tolist() == [0, 0, 0, 4]
    assert mask_tensor(image, mask, threshold=5, invert=True,
                       outside_value=7).flatten().tolist() == [1, 2, 7, 7]
    assert mask_tensor(image, mask, invert=True,
                       outside_value=7).flatten().tolist() == [1, 7, 7, 7]


def test_mgz_roundtrip_preserves_voxels_dtype_and_geometry(tmp_path):
    affine = np.diag([1, 1, 1, 1]).astype(float)
    values = np.arange(24, dtype=np.uint8).reshape(2, 3, 4)
    mask = np.where(values > 7, 10, 0).astype(np.uint8)
    image_path, mask_path, out_path = (tmp_path / name for name in
                                       ("input.mgz", "mask.mgz", "out.mgz"))
    nib.save(nib.MGHImage(values, affine), str(image_path))
    nib.save(nib.MGHImage(mask, affine), str(mask_path))
    mask_volume(image_path, mask_path, out_path, threshold=5, device="cpu")
    output = nib.load(str(out_path))
    np.testing.assert_array_equal(np.asarray(output.dataobj), np.where(mask > 5, values, 0))
    np.testing.assert_array_equal(output.affine, affine)
    assert output.get_data_dtype() == np.dtype("uint8")
    source_raw, output_raw = (gzip.decompress(path.read_bytes())
                              for path in (image_path, out_path))
    end = 284 + values.nbytes
    assert source_raw[:284] == output_raw[:284]
    assert source_raw[end:] == output_raw[end:]


def test_rejects_misaligned_volumes(tmp_path):
    image_path, mask_path = tmp_path / "input.mgz", tmp_path / "mask.mgz"
    nib.save(nib.MGHImage(np.zeros((2, 3, 4), dtype=np.uint8), np.eye(4)), str(image_path))
    affine = np.eye(4)
    affine[0, 3] = 1
    nib.save(nib.MGHImage(np.ones((2, 3, 4), dtype=np.uint8), affine), str(mask_path))
    with pytest.raises(ValueError, match="share a 3D voxel grid"):
        mask_volume(image_path, mask_path, tmp_path / "out.mgz", device="cpu")
