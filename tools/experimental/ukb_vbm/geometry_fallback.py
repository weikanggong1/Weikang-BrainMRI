"""Derive the cropped-T1 to original-T1 FLIRT matrix from NIfTI geometry.

The two images contain the same scan when UKB gradient unwarping is skipped.
This substitutes the failed xyztrans.sch fit in non-UKB acquisitions.
"""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np


def voxel_to_fsl(image):
    matrix = np.diag([*image.header.get_zooms()[:3], 1.0])
    if np.linalg.det(image.affine[:3, :3]) > 0:
        flip = np.eye(4)
        flip[0, 0] = -1
        flip[0, 3] = image.shape[0] - 1
        matrix = matrix @ flip
    return matrix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cropped", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cropped = nib.load(args.cropped)
    original = nib.load(args.original)
    matrix = (voxel_to_fsl(original) @ np.linalg.inv(original.affine)
              @ cropped.affine @ np.linalg.inv(voxel_to_fsl(cropped)))
    np.savetxt(args.output, matrix, fmt="%.12g")
    print(np.round(matrix, 4))


if __name__ == "__main__":
    main()
