"""Checks for the isolated experimental Torch bias-correction candidate."""

import pytest
import nibabel as nib
import numpy as np

torch = pytest.importorskip("torch")
from fnit.recon_all.n4_gpu import correct, estimate_log_bias, run


def test_rejects_empty_and_negative_image():
    with pytest.raises(ValueError, match="no foreground"):
        estimate_log_bias(torch.zeros(16, 16, 16), sigma=1)
    with pytest.raises(ValueError, match="nonnegative"):
        estimate_log_bias(-torch.ones(16, 16, 16), sigma=1)


def test_smooth_bias_correction_reduces_within_tissue_variance():
    axis = torch.linspace(-1, 1, 64)
    x, y, z = torch.meshgrid(axis, axis, axis, indexing="ij")
    radius = (x.square() + y.square() + z.square()).sqrt()
    tissue = torch.where(radius < 0.38, 110., 75.)
    tissue = torch.where(radius < 0.75, tissue, 0.)
    biased = tissue * torch.exp(0.35 * x)
    result = correct(biased)
    assert result.shape == biased.shape
    assert torch.isfinite(result).all()
    assert torch.count_nonzero(result[~(radius < 0.75)]) == 0
    mask = radius < 0.37
    before = biased[mask].std() / biased[mask].mean()
    after = result[mask].std() / result[mask].mean()
    assert after < before, (before.item(), after.item())


def test_mgz_output_keeps_float_precision(tmp_path):
    data = np.full((16, 16, 16), 25.5, dtype=np.float32)
    source, output = tmp_path / "orig.mgz", tmp_path / "nu0.mgz"
    nib.save(nib.MGHImage(data, np.eye(4)), source)
    run(source, output, device="cpu")
    result = nib.load(output)
    assert result.get_data_dtype() == np.dtype(">f4")
    assert np.isfinite(np.asarray(result.dataobj)).all()
