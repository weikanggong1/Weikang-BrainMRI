"""Import a 3D float32 T1 NIfTI for recon-all's first ``mri_convert`` call.

This is an image I/O step: nibabel handles NIfTI and MGH serialization;
there is no voxel computation to accelerate with PyTorch.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
from nibabel.freesurfer.mghformat import MGHHeader
import numpy as np


def import_t1(input_file: str | Path, output_file: str | Path) -> None:
    """Match ``mri_convert input.nii.gz mri/orig/001.mgz`` for a float T1."""
    source = nib.load(str(input_file))
    if not isinstance(source, nib.Nifti1Image) or len(source.shape) != 3:
        raise ValueError("expected a 3D NIfTI-1 input")
    if source.get_data_dtype().newbyteorder("=") != np.dtype("float32"):
        raise ValueError("this recon-all import expects float32 voxels")
    if source.header["sform_code"] == 0:
        raise ValueError("this recon-all import expects a valid NIfTI sform")
    if source.dataobj.slope != 1 or source.dataobj.inter != 0:
        raise ValueError("scaled NIfTI voxels are not supported by this import")
    spatial_unit, time_unit = source.header.get_xyzt_units()
    if spatial_unit != "mm" or time_unit not in ("sec", "msec", "usec"):
        raise ValueError("expected millimeter and known temporal NIfTI units")

    # FreeSurfer's MRIsetVox2RASFromMatrix normalizes each sform column in
    # double precision, then stores MGH geometry as float32. Its center
    # calculation accumulates the three voxel terms before adding offset.
    sform = np.asarray(source.header.get_sform(), dtype=np.float32)
    columns = sform[:3, :3]
    sizes = np.linalg.norm(columns.astype(np.float64), axis=0)
    directions = (columns / sizes).T.astype(np.float32)
    midpoint = np.asarray(source.shape, dtype=np.float32) / 2
    center = np.empty(3, dtype=np.float32)
    for row in range(3):
        center[row] = np.float32(
            np.float32(np.float32(columns[row, 0] * midpoint[0])
                       + np.float32(columns[row, 1] * midpoint[1]))
            + np.float32(columns[row, 2] * midpoint[2])) + sform[row, 3]

    header = MGHHeader()
    header.set_data_shape(source.shape)
    header.set_data_dtype(np.float32)
    header["dof"] = 1
    header["delta"] = source.header["pixdim"][1:4]
    header["Mdc"] = directions
    header["Pxyz_c"] = center
    header["tr"] = np.float32(source.header["pixdim"][4] *
                              {"sec": 1000, "msec": 1, "usec": 0.001}[time_unit])
    header["fov"] = np.float32(header["delta"][0] * source.shape[0])
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.MGHImage(np.asarray(source.dataobj), None, header), str(output_path))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_file")
    parser.add_argument("output_file")
    args = parser.parse_args(argv)
    import_t1(args.input_file, args.output_file)


if __name__ == "__main__":
    main()
