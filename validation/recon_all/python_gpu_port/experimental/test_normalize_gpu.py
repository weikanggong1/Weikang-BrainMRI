import nibabel as nib
import numpy as np
import torch

from normalize_gpu import normalize_tensor, run


def test_first_pass_scales_white_matter_mode():
    image = torch.full((9, 9, 9), 120, dtype=torch.uint8)
    result, details = normalize_tensor(image)
    assert details == {"phase": "first", "mode": 120, "control_voxels": 0}
    assert result.dtype == torch.uint8
    assert torch.all(result == 110)


def test_second_pass_uses_wm_ridge_and_brain_mask():
    image = torch.full((9, 9, 9), 100, dtype=torch.uint8)
    aseg = torch.zeros_like(image)
    aseg[2:7, 2:7, 2:7] = 2
    mask = torch.ones_like(image)
    mask[0] = 0
    result, details = normalize_tensor(image, aseg=aseg, mask=mask)
    assert details["phase"] == "second"
    assert details["control_voxels"] > 0
    assert torch.all(result[0] == 0)
    assert torch.all(result[1:] == 110)


def test_second_pass_reads_big_endian_mgh_aseg(tmp_path):
    image = np.full((9, 9, 9), 100, dtype=np.uint8)
    aseg = np.zeros(image.shape, dtype=np.int32)
    aseg[2:7, 2:7, 2:7] = 2
    for name, values in (("source", image), ("aseg", aseg),
                         ("mask", np.ones_like(image))):
        nib.save(nib.MGHImage(values, np.eye(4)), str(tmp_path / f"{name}.mgz"))
    run(tmp_path / "source.mgz", tmp_path / "output.mgz", phase="second",
        aseg_file=tmp_path / "aseg.mgz", mask_file=tmp_path / "mask.mgz",
        device="cpu")
    np.testing.assert_array_equal(np.asarray(nib.load(str(tmp_path / "output.mgz")).dataobj), 110)
