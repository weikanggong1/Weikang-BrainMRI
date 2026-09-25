"""Python ANTs replacement for the fixed ``AntsDenoiseImageFs`` step."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import time

import nibabel as nib
import numpy as np

from .mgh_compat import save_same_dtype_mgh


def _to_uchar(values: np.ndarray) -> np.ndarray:
    """FreeSurfer ``MRIsetVoxVal`` clipping and positive half-up ``nint``."""
    return np.floor(np.clip(values, 0, 255) + 0.5).astype(np.uint8)


def denoise_volume(input_file: str | Path, output_file: str | Path) -> dict:
    """Match ``AntsDenoiseImageFs -i brain.mgz -o antsdn.brain.mgz``."""
    started = time.perf_counter()
    source = nib.load(str(input_file))
    if not isinstance(source, nib.MGHImage) or len(source.shape) != 3:
        raise ValueError("expected a 3D MGH/MGZ image")
    if source.get_data_dtype() != np.dtype("uint8"):
        raise ValueError("the fixed brain.mgz profile requires uint8 voxels")

    # The FreeSurfer wrapper fixes ITK to one thread before running the filter.
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "1"
    import ants

    image = ants.from_numpy(np.asarray(source.dataobj, dtype=np.float32),
                            spacing=tuple(float(v) for v in source.header.get_zooms()[:3]))
    result = ants.denoise_image(image, mask=None, shrink_factor=1, p=1, r=2,
                                noise_model="Gaussian").numpy()
    save_same_dtype_mgh(input_file, output_file, _to_uchar(result))
    return {"voxels": int(result.size), "total_seconds": time.perf_counter() - started}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--i", required=True, dest="input_file")
    parser.add_argument("--o", required=True, dest="output_file")
    args = parser.parse_args(argv)
    print(denoise_volume(args.input_file, args.output_file))


if __name__ == "__main__":
    main()
