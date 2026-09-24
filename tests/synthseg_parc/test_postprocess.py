"""Compare pure-torch component filtering with the official SciPy procedure."""

import numpy as np
import pytest
import torch
from scipy.ndimage import label as scipy_label

from freesurfer_torch.synthseg_parc.postprocess import (
    largest_connected_component, postprocess_segmentation,
)


def _reference_largest(mask):
    components, count = scipy_label(mask)
    if count == 0:
        return mask.copy()
    return components == np.bincount(components.flat)[1:].argmax() + 1


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_largest_component_matches_scipy_six_neighbors(device):
    rng = np.random.default_rng(12)
    masks = [rng.random((12, 11, 10)) < 0.24,
             np.zeros((8, 8, 8), dtype=bool),
             np.ones((8, 8, 8), dtype=bool)]
    tied = np.zeros((5, 5, 5), dtype=bool)
    tied[2, 1, 1] = tied[3, 2, 2] = True  # shifted diagonal is disconnected
    masks.append(tied)
    for mask in masks:
        actual = largest_connected_component(torch.as_tensor(mask, device=device))
        np.testing.assert_array_equal(actual.cpu().numpy(), _reference_largest(mask))


@pytest.mark.parametrize("fast", [False, True])
def test_synthseg_posterior_filter_matches_official_operations(fast):
    torch.manual_seed(13)
    posterior = torch.softmax(torch.randn(4, 10, 9, 8), dim=0)
    labels = torch.tensor([0, 2, 3, 42])
    classes = torch.tensor([0, 1, 1, 2])
    content = (slice(1, 9), slice(2, 8), slice(1, 7))

    reference = posterior.permute(1, 2, 3, 0).numpy().copy()
    if fast:
        reference = reference[content]
    reference[..., 1:] *= _reference_largest(reference[..., 1:].sum(-1) > 0.25)[..., None]
    if fast:
        reference[..., 1:] *= reference[..., 1:] > 0.2
    else:
        for group in (1, 2):
            channels = np.where(classes.numpy() == group)[0]
            mask = _reference_largest((reference[..., channels] > 0.25).any(-1))
            reference[..., channels] *= mask[..., None]
        reference = reference[content]
    reference /= reference.sum(-1, keepdims=True)
    expected = labels.numpy()[reference.argmax(-1)]

    actual_labels, actual_posterior = postprocess_segmentation(
        posterior, labels, classes, content, fast=fast)
    np.testing.assert_array_equal(actual_labels.numpy(), expected)
    np.testing.assert_allclose(actual_posterior.permute(1, 2, 3, 0).numpy(),
                               reference, rtol=0, atol=2e-7)
