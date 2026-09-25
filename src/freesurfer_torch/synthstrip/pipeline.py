"""Importable PyTorch implementation of FreeSurfer SynthStrip.

The official mri_synthstrip implementation already uses PyTorch. This module
preserves its network parameter names and Surfa image geometry operations.
Reference: FreeSurfer 8.2.0, build 20260314-d932c45, python/scripts/mri_synthstrip.
Source: https://github.com/freesurfer/freesurfer/blob/d932c45/mri_synthstrip/mri_synthstrip
SynthStrip: Hoopes et al., NeuroImage (2022), doi:10.1016/j.neuroimage.2022.119474.
"""

import argparse
from dataclasses import dataclass
import os
from pathlib import Path

import numpy as np
import surfa as sf
import torch

from ..weights import resolve_weights
from .._batch_table import cases_from_table
from .._parallel_table import run_parallel
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

    def predict_batch(self, table, border=1, fill=None, workers=1, threads_per_worker=1):
        """Save brain image, mask and distance map for each output prefix."""
        cases = cases_from_table(table)
        def paths_for(prefix):
            return {"image": Path(f"{prefix}_brain.nii.gz"),
                    "mask": Path(f"{prefix}_mask.nii.gz"),
                    "distance": Path(f"{prefix}_sdt.nii.gz")}

        def run_local(case):
            source, prefix = case
            result = self(source, border=border, fill=fill)
            paths = paths_for(prefix)
            prefix.parent.mkdir(parents=True, exist_ok=True)
            for name, path in paths.items():
                getattr(result, name).save(path)
            return paths

        def make_job(case):
            source, prefix = case
            return {"task": "synthstrip", "model": {"weights": str(self.model_path.resolve())},
                    "kwargs": {"image": source, "border": border, "fill": fill},
                    "outputs": paths_for(prefix)}

        return run_parallel(cases, self, 'synthstrip', workers, threads_per_worker,
                            make_job, run_local)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--image", required=True)
    parser.add_argument("-o", "--out", help="stripped image")
    parser.add_argument("-m", "--mask", help="binary brain mask")
    parser.add_argument("-d", "--sdt", help="signed distance transform")
    parser.add_argument("-g", "--gpu", action="store_true")
    parser.add_argument("-b", "--border", type=float, default=1)
    parser.add_argument("-t", "--threads", type=int)
    parser.add_argument("-f", "--fill", type=float)
    parser.add_argument("--no-csf", action="store_true")
    parser.add_argument("--model", help="official .pt checkpoint")
    parser.add_argument("-v", "--version", action="version", version="freesurfer-torch SynthStrip 1")
    args = parser.parse_args(argv)
    if not any((args.out, args.mask, args.sdt)):
        parser.error("provide at least one -o, -m, or -d output")
    model = SynthStrip(
        weights=args.model,
        device="cuda" if args.gpu else "cpu",
        no_csf=args.no_csf,
        threads=args.threads,
    )
    result = model(args.image, border=args.border, fill=args.fill)
    for volume, path in ((result.image, args.out), (result.mask, args.mask), (result.distance, args.sdt)):
        if path:
            volume.save(path)
            print(path)


if __name__ == "__main__":
    main()
