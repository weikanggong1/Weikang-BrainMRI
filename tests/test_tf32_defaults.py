"""CUDA entry points enable TF32 while retaining float32 tensors."""

import pytest
import torch

from freesurfer_torch.applywarp import TorchApplyWarp
from freesurfer_torch.fast import TorchFAST
from freesurfer_torch.flirt import TorchFLIRT
from freesurfer_torch.fnirt import TorchFNIRT


@pytest.mark.parametrize(
    "constructor",
    (TorchApplyWarp, TorchFAST, TorchFLIRT, TorchFNIRT),
)
def test_cuda_registration_components_enable_tf32(monkeypatch, constructor):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)

    constructor(device="cuda:0")

    assert torch.backends.cuda.matmul.allow_tf32 is True
    assert torch.backends.cudnn.allow_tf32 is True
