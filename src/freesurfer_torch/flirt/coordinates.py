"""FSL scaled-mm and world-RAS affine coordinate conversions."""

import numpy as np


WORLD_FORWARD_CONVENTION = "moving-to-fixed-world-ras"
WORLD_PULL_CONVENTION = "fixed-to-moving-world-ras"


def _numpy_affine(value, name):
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], (0, 0, 0, 1), atol=1e-8, rtol=0):
        raise ValueError(f"{name} must be a homogeneous affine matrix")
    if abs(float(np.linalg.det(matrix[:3, :3]))) < 1e-8:
        raise ValueError(f"{name} must be invertible")
    return matrix


def voxel_to_fsl_scaled_mm(vox2world, shape, voxel_sizes=None):
    """Return FSL's voxel-index to scaled-mm transform.

    FSL scaled-mm coordinates are neither NIfTI world-RAS nor voxel indices.
    FSL applies voxel sizes and flips the first axis for neurological input
    storage (a positive voxel-to-world determinant). Pass the NIfTI header
    ``pixdim`` values as ``voxel_sizes`` when they cannot be recovered from a
    sheared sform; otherwise affine column norms are used.
    """
    affine = _numpy_affine(vox2world, "vox2world")
    shape = tuple(int(size) for size in shape)
    if len(shape) != 3 or any(size < 1 for size in shape):
        raise ValueError("shape must contain three positive dimensions")
    if voxel_sizes is None:
        voxel_sizes = np.linalg.norm(affine[:3, :3], axis=0)
    else:
        voxel_sizes = np.asarray(voxel_sizes, dtype=np.float64)
        if voxel_sizes.shape != (3,):
            raise ValueError("voxel_sizes must contain three values")
    if not np.isfinite(voxel_sizes).all() or np.any(voxel_sizes <= 0):
        raise ValueError("vox2world must define positive voxel sizes")
    scaled = np.diag([*voxel_sizes, 1.0])
    if np.linalg.det(affine[:3, :3]) > 0:
        scaled[0, 0] *= -1
        scaled[0, 3] = voxel_sizes[0] * (shape[0] - 1)
    return scaled


def flirt_to_world_affine(
    flirt_matrix,
    moving_vox2world,
    fixed_vox2world,
    moving_shape,
    fixed_shape,
    moving_voxel_sizes=None,
    fixed_voxel_sizes=None,
):
    """Convert an FSL FLIRT matrix to moving-to-fixed world-RAS.

    A FLIRT ``.mat`` maps moving/input FSL scaled-mm coordinates to
    fixed/reference FSL scaled-mm coordinates. It must not be used directly
    as a NIfTI or Surfa world transform. The optional voxel-size arguments are
    the stored NIfTI ``pixdim`` values and are needed for sheared sforms.
    """
    flirt = _numpy_affine(flirt_matrix, "flirt_matrix")
    moving_world = _numpy_affine(moving_vox2world, "moving_vox2world")
    fixed_world = _numpy_affine(fixed_vox2world, "fixed_vox2world")
    moving_fsl = voxel_to_fsl_scaled_mm(
        moving_world, moving_shape, moving_voxel_sizes
    )
    fixed_fsl = voxel_to_fsl_scaled_mm(
        fixed_world, fixed_shape, fixed_voxel_sizes
    )
    forward = (
        fixed_world
        @ np.linalg.inv(fixed_fsl)
        @ flirt
        @ moving_fsl
        @ np.linalg.inv(moving_world)
    )
    forward[3] = (0, 0, 0, 1)
    return forward


def world_to_flirt_affine(
    world_matrix,
    moving_vox2world,
    fixed_vox2world,
    moving_shape,
    fixed_shape,
    moving_voxel_sizes=None,
    fixed_voxel_sizes=None,
):
    """Convert moving-to-fixed world-RAS to an FSL FLIRT matrix.

    The returned matrix maps moving/input FSL scaled-mm coordinates to
    fixed/reference FSL scaled-mm coordinates, matching the coordinate and
    direction contract of a FLIRT ``-omat`` file. The optional voxel-size
    arguments are the stored NIfTI ``pixdim`` values.
    """
    forward = _numpy_affine(world_matrix, "world_matrix")
    moving_world = _numpy_affine(moving_vox2world, "moving_vox2world")
    fixed_world = _numpy_affine(fixed_vox2world, "fixed_vox2world")
    moving_fsl = voxel_to_fsl_scaled_mm(
        moving_world, moving_shape, moving_voxel_sizes
    )
    fixed_fsl = voxel_to_fsl_scaled_mm(
        fixed_world, fixed_shape, fixed_voxel_sizes
    )
    flirt = (
        fixed_fsl
        @ np.linalg.inv(fixed_world)
        @ forward
        @ moving_world
        @ np.linalg.inv(moving_fsl)
    )
    flirt[3] = (0, 0, 0, 1)
    return flirt


def flirt_to_world_pull(
    flirt_matrix,
    moving_vox2world,
    fixed_vox2world,
    moving_shape,
    fixed_shape,
    moving_voxel_sizes=None,
    fixed_voxel_sizes=None,
):
    """Convert FLIRT scaled-mm forward coordinates to a world-RAS pull."""
    forward = flirt_to_world_affine(
        flirt_matrix,
        moving_vox2world,
        fixed_vox2world,
        moving_shape,
        fixed_shape,
        moving_voxel_sizes,
        fixed_voxel_sizes,
    )
    pull = np.linalg.inv(forward)
    pull[3] = (0, 0, 0, 1)
    return pull



__all__ = [
    "WORLD_FORWARD_CONVENTION",
    "WORLD_PULL_CONVENTION",
    "flirt_to_world_affine",
    "flirt_to_world_pull",
    "voxel_to_fsl_scaled_mm",
    "world_to_flirt_affine",
]
