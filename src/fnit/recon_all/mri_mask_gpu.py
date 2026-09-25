"""GPU implementation of the aligned-volume mri_mask calls in single-T1 recon-all."""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from .mgh_compat import save_same_dtype_mgh


def mask_tensor(image: torch.Tensor, mask: torch.Tensor, *,
                threshold: float | None = None, invert: bool = False,
                outside_value: float = 0) -> torch.Tensor:
    """Apply FreeSurfer's mask, threshold and inverted-mask voxel rules."""
    if image.shape != mask.shape or image.ndim != 3:
        raise ValueError("input and mask must have the same 3D shape")
    if invert:
        selected = mask < (0.5 if threshold is None else threshold)
    elif threshold is None:
        selected = mask != 0
    else:
        selected = mask > threshold
    return torch.where(selected, image, torch.full_like(image, outside_value))


def mask_volume(input_file: str | Path, mask_file: str | Path, output_file: str | Path,
                *, threshold: float | None = None, invert: bool = False,
                outside_value: float = 0, device: str = "cuda:0") -> None:
    """Read, mask on the selected device, and write an aligned MGH/MGZ volume."""
    source = nib.load(str(input_file))
    mask = nib.load(str(mask_file))
    if not isinstance(source, nib.MGHImage) or not isinstance(mask, nib.MGHImage):
        raise ValueError("this recon-all replacement requires MGH/MGZ input volumes")
    if source.shape != mask.shape or len(source.shape) != 3 or not np.allclose(
            source.affine, mask.affine, rtol=0, atol=1e-4):
        raise ValueError("input and mask must share a 3D voxel grid and geometry")
    source_data = np.asarray(source.dataobj).astype(source.get_data_dtype().newbyteorder("="), copy=True)
    mask_data = np.asarray(mask.dataobj).astype(mask.get_data_dtype().newbyteorder("="), copy=True)
    result = mask_tensor(torch.from_numpy(source_data).to(device),
                         torch.from_numpy(mask_data).to(device),
                         threshold=threshold, invert=invert,
                         outside_value=outside_value).cpu().numpy()
    save_same_dtype_mgh(input_file, output_file, result)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-T", type=float, dest="threshold")
    parser.add_argument("-invert", action="store_true")
    parser.add_argument("-oval", type=float, default=0, dest="outside_value")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("input_file")
    parser.add_argument("mask_file")
    parser.add_argument("output_file")
    args = parser.parse_args(argv)
    mask_volume(args.input_file, args.mask_file, args.output_file,
                threshold=args.threshold, invert=args.invert,
                outside_value=args.outside_value, device=args.device)


if __name__ == "__main__":
    main()
