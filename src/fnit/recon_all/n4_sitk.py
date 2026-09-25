"""CPU Python replay of FreeSurfer's isolated ANTs N4 correction step.

This uses SimpleITK's ITK N4 filter. It produces the ``nu0.mgz`` image from
``orig.mgz``; ``mri_nu_correct.mni`` performs further scaling and uchar mapping.
The filter executes on CPU and is not part of the GPU recon-all pipeline.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np

from fnit.recon_all.mgh_compat import save_same_dtype_mgh


def _to_uchar(values: np.ndarray) -> np.ndarray:
    """Match MRIsetVoxVal's clipping and nint for nonnegative N4 output."""
    return np.floor(np.clip(values, 0, 255) + 0.5).astype(np.uint8)


def correct_volume(input_file: str | Path, output_file: str | Path) -> None:
    """Replay ``AntsN4BiasFieldCorrectionFs -i orig -o nu0 --dtype uchar``."""
    import SimpleITK as sitk

    source = nib.load(str(input_file))
    if not isinstance(source, nib.MGHImage) or len(source.shape) != 3:
        raise ValueError("expected a 3D MGH/MGZ input")

    array = np.asarray(source.dataobj, dtype=np.float32)
    image = sitk.GetImageFromArray(np.ascontiguousarray(array.transpose(2, 1, 0)))
    # FreeSurfer's MRI::toITKImage transfers voxel spacing, but uses the ITK
    # default origin and direction. Its no-mask path includes every voxel.
    image.SetSpacing(tuple(float(v) for v in source.header.get_zooms()[:3]))
    mask = sitk.Image(image.GetSize(), sitk.sitkUInt8)
    mask.CopyInformation(image)
    mask += 1

    previous_threads = sitk.ProcessObject.GetGlobalDefaultNumberOfThreads()
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    try:
        small = sitk.Shrink(image, [4, 4, 4])
        small_mask = sitk.Shrink(mask, [4, 4, 4])
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrector.SetMaximumNumberOfIterations([50, 50, 50, 50])
        corrector.SetConvergenceThreshold(0)
        corrector.Execute(small, small_mask)
        log_field = corrector.GetLogBiasFieldAsImage(image)
        corrected = image / sitk.Exp(log_field)
        # SimpleITK promotes division to float64; FreeSurfer's ITKImageType
        # and DivideImageFilter output are float32 before uchar conversion.
        result = np.asarray(sitk.GetArrayFromImage(corrected).transpose(2, 1, 0),
                            dtype=np.float32)
    finally:
        sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(previous_threads)

    uchar = _to_uchar(result)
    if source.get_data_dtype() == np.dtype("uint8"):
        save_same_dtype_mgh(input_file, output_file, uchar)
    else:
        header = source.header.copy()
        header.set_data_dtype(np.uint8)
        nib.save(nib.MGHImage(uchar, source.affine, header), str(output_file))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--i", required=True, dest="input_file")
    parser.add_argument("--o", required=True, dest="output_file")
    args = parser.parse_args(argv)
    correct_volume(args.input_file, args.output_file)


if __name__ == "__main__":
    main()
