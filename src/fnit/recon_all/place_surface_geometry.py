"""Native-order surface RAS to MRI voxel map for FreeSurfer placement."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .mri_em_register import _vnl_affine_inverse


def _vox2ras(
    directions: np.ndarray,
    spacing: np.ndarray,
    center: np.ndarray,
    dimensions: np.ndarray,
) -> np.ndarray:
    result = np.eye(4, dtype=np.float32)
    result[:3, :3] = np.float32(np.asarray(directions, np.float32).T * np.asarray(spacing, np.float32)[None, :])
    offset = np.array([
        sum(float(result[row, column]) * float(np.float32(dimensions[column] / 2)) for column in range(3))
        for row in range(3)
    ], dtype=np.float32)
    result[:3, 3] = np.float32(np.asarray(center, np.float32) - offset)
    return result


def _multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    result = np.empty((4, 4), dtype=np.float32)
    for row in range(4):
        for column in range(4):
            value = np.float32(0)
            for index in range(4):
                value = np.float32(value + np.float32(left[row, index] * right[index, column]))
            result[row, column] = value
    return result


def surface_ras_to_voxel(volume_header: Mapping, surface_metadata: Mapping) -> np.ndarray:
    """Construct ``MRIS_makeRAS2VoxelMap`` from MGH and surface metadata."""
    scanner = _vox2ras(
        volume_header["Mdc"], volume_header["delta"], volume_header["Pxyz_c"], volume_header["dims"][:3],
    )
    surface = _vox2ras(
        np.stack([surface_metadata["xras"], surface_metadata["yras"], surface_metadata["zras"]]),
        surface_metadata["voxelsize"], surface_metadata["cras"], surface_metadata["volume"],
    )
    width, height, depth = np.asarray(surface_metadata["volume"], np.float32)
    xsize, ysize, zsize = np.asarray(surface_metadata["voxelsize"], np.float32)
    tkreg = np.zeros((4, 4), dtype=np.float32)
    tkreg[0, 0], tkreg[0, 3] = -xsize, np.float32(width * xsize / 2)
    tkreg[1, 2], tkreg[1, 3] = zsize, np.float32(-depth * zsize / 2)
    tkreg[2, 1], tkreg[2, 3] = -ysize, np.float32(height * ysize / 2)
    tkreg[3, 3] = 1
    return _multiply(_vnl_affine_inverse(scanner), _multiply(surface, _vnl_affine_inverse(tkreg)))
