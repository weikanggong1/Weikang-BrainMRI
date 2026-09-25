"""Small exact arithmetic checks for the isolated normalization port."""

import numpy as np
import torch

from fnit.recon_all.normalization.normalize_gaussian_source import (
    apply_gentle_bias, smooth_bias, smooth_bias_torch)
from fnit.recon_all.normalization.normalize_voronoi_source import (
    voronoi_fill, voronoi_fill_torch)


def test_wavefront_and_gaussian_match_tensor_path():
    source = np.zeros((7, 8, 9), dtype=np.float32)
    control = np.zeros(source.shape, dtype=np.uint8)
    control[1, 2, 3] = control[5, 6, 7] = 1
    source[1, 2, 3], source[5, 6, 7] = 99, 129
    reference, _ = voronoi_fill(source, control)
    tensor, _ = voronoi_fill_torch(torch.from_numpy(source), torch.from_numpy(control))
    np.testing.assert_array_equal(tensor.numpy(), reference)
    expected, _ = smooth_bias(reference, source, control)
    smoothed, _ = smooth_bias_torch(tensor, torch.from_numpy(source), torch.from_numpy(control))
    np.testing.assert_array_equal(smoothed.numpy(), expected)


def test_gentle_application_uses_freesurfer_half_up_rounding():
    source = torch.tensor([[[3., 5., 7.]]])
    bias = torch.tensor([[[110., 110., 110.]]])
    assert apply_gentle_bias(source, bias).flatten().tolist() == [3, 5, 7]
    source = torch.tensor([[[5., 7.]]])
    bias = torch.tensor([[[4.9, 6.9]]])
    assert apply_gentle_bias(source, bias).flatten().tolist() == [138, 128]
