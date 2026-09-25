"""Numerical checks for three-tissue segmentation and bias correction."""

import numpy as np
import pytest
import torch

from fnit.fast.algorithm import FASTConfig, segment_t1


def _phantom(device="cpu"):
    shape = (18, 16, 14)
    coordinates = torch.meshgrid(
        *(torch.linspace(-1, 1, size, device=device) for size in shape),
        indexing="ij",
    )
    x, y, z = coordinates
    mask = x.square() + (1.1 * y).square() + (1.2 * z).square() < 0.85
    classes = torch.where(x < -0.22, 0, torch.where(x < 0.28, 1, 2))
    base = torch.tensor((35.0, 75.0, 120.0), device=device)[classes]
    bias = torch.exp(0.28 * x - 0.12 * y + 0.04 * z)
    noise = 0.4 * torch.sin(7 * x + 3 * y - 2 * z)
    image = torch.where(mask, base * bias + noise, torch.zeros_like(base))
    return image, mask, classes


def _quick_config():
    return FASTConfig(
        init_iterations=4,
        bias_iterations=2,
        fixed_iterations=2,
        bias_fwhm_mm=8,
        pve_steps=20,
        mean_field_iterations=3,
        pve_chunk_size=5,
    )


def test_fast_defaults_match_single_channel_t1_settings():
    config = FASTConfig()
    assert (config.init_iterations, config.bias_iterations,
            config.fixed_iterations) == (15, 4, 4)
    assert (config.bias_fwhm_mm, config.init_mrf, config.mrf,
            config.mixel_mrf, config.pve_steps) == (20, 0.02, 0.1, 0.3, 100)


def test_pve_bias_and_class_order_are_well_formed():
    image, mask, classes = _phantom()
    result = segment_t1(image, mask, voxel_size=(1.0, 1.2, 1.5),
                        config=_quick_config())

    assert result.pve.shape == (3, *image.shape)
    assert torch.isfinite(result.pve).all()
    assert torch.all((result.pve >= 0) & (result.pve <= 1))
    torch.testing.assert_close(result.pve[:, mask].sum(dim=0),
                               torch.ones(int(mask.sum())), atol=1e-6, rtol=0)
    assert torch.count_nonzero(result.pve[:, ~mask]) == 0
    assert torch.all(result.bias_field > 0)
    assert torch.all(result.bias_field[~mask] == 1)
    torch.testing.assert_close(result.restored[mask],
                               image[mask] / result.bias_field[mask])
    assert abs(float(torch.log(result.bias_field[mask]).mean())) < 1e-5
    assert torch.all(result.tissue_means[1:] > result.tissue_means[:-1])

    predicted = result.hard_segmentation[mask] - 1
    assert float((predicted == classes[mask]).float().mean()) > 0.9
    before = torch.stack([image[mask & (classes == index)].std()
                          / image[mask & (classes == index)].mean()
                          for index in range(3)]).mean()
    after = torch.stack([result.restored[mask & (classes == index)].std()
                         / result.restored[mask & (classes == index)].mean()
                         for index in range(3)]).mean()
    assert after < before


def test_cpu_result_is_repeatable():
    image, mask, _ = _phantom()
    first = segment_t1(image, mask, config=_quick_config())
    second = segment_t1(image, mask, config=_quick_config())
    for name in ("pve", "hard_segmentation", "pve_segmentation", "mixel_type",
                 "bias_field", "restored"):
        torch.testing.assert_close(getattr(first, name), getattr(second, name),
                                   atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cpu_and_cuda_are_close():
    image, mask, _ = _phantom()
    cpu = segment_t1(image, mask, voxel_size=(1.0, 1.2, 1.5),
                     config=_quick_config())
    cuda = segment_t1(image.cuda(), mask.cuda(), voxel_size=(1.0, 1.2, 1.5),
                      config=_quick_config())
    torch.testing.assert_close(cpu.pve, cuda.pve.cpu(), atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(cpu.bias_field, cuda.bias_field.cpu(),
                               atol=2e-4, rtol=2e-4)


@pytest.mark.parametrize("bad_config", [
    {"pve_steps": 0}, {"bias_fwhm_mm": -1}, {"mrf": float("nan")},
])
def test_invalid_config_is_rejected(bad_config):
    with pytest.raises(ValueError):
        FASTConfig(**bad_config)


def test_invalid_image_or_mask_is_rejected():
    image = torch.ones((4, 4, 4))
    with pytest.raises(ValueError, match="3D"):
        segment_t1(image[0])
    image[0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="NaN"):
        segment_t1(image)
    with pytest.raises(ValueError, match="same shape"):
        segment_t1(torch.ones((4, 4, 4)), torch.ones((3, 3, 3)))
    with pytest.raises(ValueError, match="separable"):
        segment_t1(torch.ones((4, 4, 4)))


def test_sparse_negative_values_are_clamped_before_masking():
    image, mask, _ = _phantom()
    image = image.clone()
    image[0, 0, 0] = -1
    result = segment_t1(image, mask, config=_quick_config())
    assert result.restored[0, 0, 0] == 0
