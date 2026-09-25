"""Importable PyTorch implementation of FreeSurfer SynthStrip.

The official mri_synthstrip implementation already uses PyTorch. This module
preserves its network parameter names and Surfa image geometry operations.
Reference: FreeSurfer 8.2.0, build 20260314-d932c45, python/scripts/mri_synthstrip.
Source: https://github.com/freesurfer/freesurfer/blob/d932c45/mri_synthstrip/mri_synthstrip
SynthStrip: Hoopes et al., NeuroImage (2022), doi:10.1016/j.neuroimage.2022.119474.
"""

from dataclasses import dataclass
import os
from pathlib import Path

import numpy as np
import surfa as sf
import torch

from ..weights import resolve_weights
from .model import StripModel


def extend_sdt(sdt, border=1):
    """Preserve the official outer-distance extension for large mask borders."""
    if border < int(sdt.max()):
        return sdt
    mask = sdt < 1
    keep = np.nonzero(mask)
    low = np.min(keep, axis=-1)
    upp = np.max(keep, axis=-1)
    gap = int(border + 0.5)
    low = (max(i - gap, 0) for i in low)
    upp = (min(i + gap, d - 1) for i, d in zip(upp, mask.shape))
    ind = tuple(slice(a, b + 1) for a, b in zip(low, upp))
    out = np.full_like(sdt, fill_value=100)
    out[ind] = sf.Volume(mask[ind]).distance()
    out[keep] = sdt[keep]
    return sdt.new(out)


@dataclass
class StripResult:
    image: sf.Volume
    mask: sf.Volume
    distance: sf.Volume


class SynthStrip:
    """Load an official checkpoint once and process one or more image volumes.

    Images retain their original voxel grid and geometry. A 4D input is processed
    one frame at a time. ``distance`` contains millimetre signed distances;
    the binary mask uses ``distance < border`` followed by connected components.
    """

    def __init__(self, weights=None, device="cpu", no_csf=False, threads=None):
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        if threads is not None:
            torch.set_num_threads(threads)
        # Match the executable's convolution backend settings.
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        name = "synthstrip.nocsf.1.pt" if no_csf else "synthstrip.1.pt"
        self.model_path = Path(resolve_weights(name, explicit=weights))
        self.model = StripModel().to(self.device).eval()
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    @torch.no_grad()
    def __call__(self, image, border=1, fill=None):
        if isinstance(image, (str, os.PathLike)):
            image = sf.load_volume(str(image))
        distances, masks = [], []
        for frame_index in range(image.nframes):
            frame = image.new(image.framed_data[..., frame_index])
            conformed = frame.conform(
                voxsize=1.0, dtype="float32", method="nearest", orientation="LIA"
            )
            conformed = conformed.crop_to_bbox()
            shape = np.clip(np.ceil(np.array(conformed.shape[:3]) / 64).astype(int) * 64, 192, 320)
            conformed = conformed.reshape(shape)
            conformed -= conformed.min()
            conformed = (conformed / conformed.percentile(99)).clip(0, 1)
            tensor = torch.from_numpy(conformed.data[np.newaxis, np.newaxis]).to(self.device)
            distance = self.model(tensor).squeeze().cpu()
            distance = extend_sdt(conformed.new(distance), border=border)
            distance = distance.resample_like(image, fill=100)
            distances.append(distance)
            masks.append((distance < border).connected_component_mask(k=1, fill=True))
        distance = sf.stack(distances)
        mask = sf.stack(masks)
        output = image.copy()
        background = np.min([image.min(), 0]) if fill is None else fill
        output[mask == 0] = background
        return StripResult(output, image.new(mask), image.new(distance))
