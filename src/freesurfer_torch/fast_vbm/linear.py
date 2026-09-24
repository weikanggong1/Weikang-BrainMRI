"""Independent PyTorch 12-DOF affine registration for FastVBM.

The parameterization follows FLIRT's useful conventions: rotations, translations
in world millimetres, scales, and shears are applied about the moving image's
weighted centre of gravity.  Optimization is coarse-to-fine normalized cross
correlation with Adam.  This is therefore FLIRT-compatible, but it is not a
bitwise reimplementation of FSL FLIRT's default correlation-ratio cost and
Brent search schedule.
"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F


WORLD_FORWARD_CONVENTION = "moving-to-fixed-world-ras"
WORLD_PULL_CONVENTION = "fixed-to-moving-world-ras"


@dataclass(frozen=True)
class LinearRegistrationResult:
    """Affine transforms, fixed-grid moving image, and fit diagnostics."""

    moving_to_fixed_world: np.ndarray
    fixed_to_moving_world: np.ndarray
    warped: torch.Tensor
    qc: dict

    @property
    def forward_world_affine(self):
        """Alias for the moving-to-fixed world affine."""
        return self.moving_to_fixed_world

    @property
    def pull_world_affine(self):
        """Alias for the fixed-to-moving world pull affine."""
        return self.fixed_to_moving_world


def _volume(value, name, device):
    if isinstance(value, torch.Tensor):
        tensor = value.detach().to(device=device, dtype=torch.float32)
    else:
        tensor = torch.as_tensor(
            np.asarray(value, dtype=np.float32).copy(), device=device
        )
    if tensor.ndim == 5 and tuple(tensor.shape[:2]) == (1, 1):
        tensor = tensor[0, 0]
    if tensor.ndim != 3:
        raise ValueError(f"{name} must be a 3D array or a [1, 1, D, H, W] tensor")
    if any(size < 2 for size in tensor.shape):
        raise ValueError(f"{name} dimensions must each contain at least two voxels")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must contain only finite values")
    return tensor.contiguous()


def _affine(value, name, device):
    matrix = torch.as_tensor(value, dtype=torch.float32, device=device)
    if matrix.shape != (4, 4) or not bool(torch.isfinite(matrix).all()):
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    if abs(float(torch.linalg.det(matrix[:3, :3]))) < 1e-8:
        raise ValueError(f"{name} must be invertible")
    return matrix


def _numpy_affine(value, name):
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], (0, 0, 0, 1), atol=1e-8, rtol=0):
        raise ValueError(f"{name} must be a homogeneous affine matrix")
    if abs(float(np.linalg.det(matrix[:3, :3]))) < 1e-8:
        raise ValueError(f"{name} must be invertible")
    return matrix


def voxel_to_fsl_scaled_mm(vox2world, shape):
    """Return FSL's voxel-index to scaled-mm transform.

    FSL scaled-mm coordinates are neither NIfTI world-RAS nor voxel indices.
    FSL applies voxel sizes and flips the first axis for neurological input
    storage (a positive voxel-to-world determinant).
    """
    affine = _numpy_affine(vox2world, "vox2world")
    shape = tuple(int(size) for size in shape)
    if len(shape) != 3 or any(size < 1 for size in shape):
        raise ValueError("shape must contain three positive dimensions")
    voxel_sizes = np.linalg.norm(affine[:3, :3], axis=0)
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
):
    """Convert an FSL FLIRT matrix to moving-to-fixed world-RAS.

    A FLIRT ``.mat`` maps moving/input FSL scaled-mm coordinates to
    fixed/reference FSL scaled-mm coordinates. It must not be used directly
    as a NIfTI or Surfa world transform.
    """
    flirt = _numpy_affine(flirt_matrix, "flirt_matrix")
    moving_world = _numpy_affine(moving_vox2world, "moving_vox2world")
    fixed_world = _numpy_affine(fixed_vox2world, "fixed_vox2world")
    moving_fsl = voxel_to_fsl_scaled_mm(moving_world, moving_shape)
    fixed_fsl = voxel_to_fsl_scaled_mm(fixed_world, fixed_shape)
    forward = (
        fixed_world
        @ np.linalg.inv(fixed_fsl)
        @ flirt
        @ moving_fsl
        @ np.linalg.inv(moving_world)
    )
    forward[3] = (0, 0, 0, 1)
    return forward


def flirt_to_world_pull(
    flirt_matrix,
    moving_vox2world,
    fixed_vox2world,
    moving_shape,
    fixed_shape,
):
    """Convert FLIRT scaled-mm forward coordinates to a world-RAS pull."""
    forward = flirt_to_world_affine(
        flirt_matrix,
        moving_vox2world,
        fixed_vox2world,
        moving_shape,
        fixed_shape,
    )
    pull = np.linalg.inv(forward)
    pull[3] = (0, 0, 0, 1)
    return pull


def _weighted_center(data, affine):
    values = data.clamp_min(0)
    positive = values[values > 0]
    if positive.numel() == 0:
        raise ValueError("registration image is empty")
    threshold = torch.quantile(positive, 0.25)
    weights = torch.where(values >= threshold, values, 0)
    mass = weights.sum()
    coordinates = []
    for axis, size in enumerate(data.shape):
        marginal = weights.sum(dim=tuple(i for i in range(3) if i != axis))
        index = torch.arange(size, dtype=data.dtype, device=data.device)
        coordinates.append((marginal * index).sum() / mass)
    voxel = torch.stack(coordinates)
    return affine[:3, :3] @ voxel + affine[:3, 3]


def affine_from_parameters(parameters, moving_center_world, fixed_center_world):
    """Build a moving-to-fixed world affine from FLIRT-style 12 parameters.

    Parameters are three Euler angles in radians, three translations in world
    millimetres, three log scales, and three upper-triangular shears.
    """
    p = torch.as_tensor(parameters)
    if p.shape != (12,):
        raise ValueError("parameters must contain 12 values")
    moving_center = torch.as_tensor(
        moving_center_world, dtype=p.dtype, device=p.device
    )
    fixed_center = torch.as_tensor(fixed_center_world, dtype=p.dtype, device=p.device)
    if moving_center.shape != (3,) or fixed_center.shape != (3,):
        raise ValueError("centres must each contain three world coordinates")

    rx, ry, rz = p[:3]
    zero, one = p.new_zeros(()), p.new_ones(())
    cx, sx = torch.cos(rx), torch.sin(rx)
    cy, sy = torch.cos(ry), torch.sin(ry)
    cz, sz = torch.cos(rz), torch.sin(rz)
    rot_x = torch.stack(
        (one, zero, zero, zero, cx, -sx, zero, sx, cx)
    ).reshape(3, 3)
    rot_y = torch.stack(
        (cy, zero, sy, zero, one, zero, -sy, zero, cy)
    ).reshape(3, 3)
    rot_z = torch.stack(
        (cz, -sz, zero, sz, cz, zero, zero, zero, one)
    ).reshape(3, 3)
    sh_xy, sh_xz, sh_yz = p[9:12]
    shear = torch.stack(
        (one, sh_xy, sh_xz, zero, one, sh_yz, zero, zero, one)
    ).reshape(3, 3)
    linear = rot_z @ rot_y @ rot_x @ shear @ torch.diag(torch.exp(p[6:9]))
    offset = fixed_center + p[3:6] - linear @ moving_center
    upper = torch.cat((linear, offset[:, None]), dim=1)
    return torch.cat((upper, p.new_tensor([[0.0, 0.0, 0.0, 1.0]])), dim=0)


def _world_grid(affine, shape):
    axes = torch.meshgrid(
        *[
            torch.arange(size, dtype=affine.dtype, device=affine.device)
            for size in shape
        ],
        indexing="ij",
    )
    voxels = torch.stack(axes).reshape(3, -1)
    return affine[:3, :3] @ voxels + affine[:3, 3:4]


def _resample_world(moving, world_to_moving, target_world, fixed_shape, forward):
    pull = torch.linalg.inv(forward)
    source_world = pull[:3, :3] @ target_world + pull[:3, 3:4]
    source_voxels = (
        world_to_moving[:3, :3] @ source_world + world_to_moving[:3, 3:4]
    )
    valid = torch.ones(source_voxels.shape[1], dtype=torch.bool, device=moving.device)
    normalized = []
    for axis, size in enumerate(moving.shape):
        valid &= (source_voxels[axis] >= 0) & (source_voxels[axis] <= size - 1)
        normalized.append(2 * source_voxels[axis] / (size - 1) - 1)
    grid = torch.stack(normalized[::-1], dim=-1).reshape(1, *fixed_shape, 3)
    warped = F.grid_sample(
        moving[None, None],
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )[0, 0]
    return warped, valid.reshape(fixed_shape)


def _resample_tensor(moving, moving_affine, fixed_affine, fixed_shape, forward):
    return _resample_world(
        moving,
        torch.linalg.inv(moving_affine),
        _world_grid(fixed_affine, fixed_shape),
        fixed_shape,
        forward,
    )


def resample_to_fixed(
    moving,
    moving_affine,
    fixed_affine,
    fixed_shape,
    moving_to_fixed_world,
    *,
    device=None,
):
    """Resample a moving array onto a fixed grid using a forward world affine."""
    if device is None:
        device = moving.device if isinstance(moving, torch.Tensor) else "cpu"
    device = torch.device(device)
    moving = _volume(moving, "moving", device)
    moving_affine = _affine(moving_affine, "moving_affine", device)
    fixed_affine = _affine(fixed_affine, "fixed_affine", device)
    forward = _affine(moving_to_fixed_world, "moving_to_fixed_world", device)
    shape = tuple(int(size) for size in fixed_shape)
    if len(shape) != 3 or any(size < 1 for size in shape):
        raise ValueError("fixed_shape must contain three positive dimensions")
    return _resample_tensor(moving, moving_affine, fixed_affine, shape, forward)


def _normcorr(warped, fixed, valid):
    weight = valid.to(warped.dtype)
    mass = weight.sum().clamp_min(1)
    a = (warped - (warped * weight).sum() / mass) * weight
    b = (fixed - (fixed * weight).sum() / mass) * weight
    return (a * b).sum() / torch.sqrt(a.square().sum() * b.square().sum()).clamp_min(
        1e-8
    )


def _per_level(value, count, name, cast):
    if np.isscalar(value):
        result = (cast(value),) * count
    else:
        result = tuple(cast(item) for item in value)
    if len(result) != count:
        raise ValueError(f"{name} must have one value per stride")
    return result


def register_affine(
    moving,
    fixed,
    moving_affine,
    fixed_affine,
    *,
    device=None,
    strides: Sequence[int] = (4, 2, 1),
    steps: Sequence[int] | int = (80, 60, 50),
    learning_rates: Sequence[float] | float = (0.05, 0.025, 0.0125),
    cost="normcorr",
):
    """Fit a 12-DOF moving-to-fixed affine with coarse-to-fine PyTorch NCC."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    moving = _volume(moving, "moving", device)
    fixed = _volume(fixed, "fixed", device)
    moving_affine = _affine(moving_affine, "moving_affine", device)
    fixed_affine = _affine(fixed_affine, "fixed_affine", device)
    strides = tuple(int(stride) for stride in strides)
    if not strides or any(stride < 1 for stride in strides):
        raise ValueError("strides must contain positive integers")
    steps = _per_level(steps, len(strides), "steps", int)
    learning_rates = _per_level(
        learning_rates, len(strides), "learning_rates", float
    )
    if any(value < 0 for value in steps):
        raise ValueError("steps must be non-negative")
    if any(value <= 0 for value in learning_rates):
        raise ValueError("learning_rates must be positive")
    if cost != "normcorr":
        raise ValueError("the independent PyTorch backend currently supports cost='normcorr'")

    moving_center = _weighted_center(moving, moving_affine)
    fixed_center = _weighted_center(fixed, fixed_affine)
    world_to_moving = torch.linalg.inv(moving_affine)
    parameters = torch.nn.Parameter(torch.zeros(12, dtype=torch.float32, device=device))
    levels = []
    for stride, iterations, learning_rate in zip(strides, steps, learning_rates):
        fixed_level = fixed[::stride, ::stride, ::stride]
        scale = torch.diag(
            torch.tensor(
                [stride, stride, stride, 1], dtype=torch.float32, device=device
            )
        )
        level_affine = fixed_affine @ scale
        target_world = _world_grid(level_affine, fixed_level.shape)
        optimizer = torch.optim.Adam((parameters,), lr=learning_rate)
        for _ in range(iterations):
            optimizer.zero_grad(set_to_none=True)
            forward = affine_from_parameters(
                parameters, moving_center, fixed_center
            )
            warped, valid = _resample_world(
                moving, world_to_moving, target_world, fixed_level.shape, forward
            )
            correlation = _normcorr(warped, fixed_level, valid)
            loss = (
                1 - correlation
                + 1e-3 * parameters[6:9].square().sum()
                + 1e-3 * parameters[9:12].square().sum()
            )
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                parameters[:3].clamp_(-1.2, 1.2)
                parameters[3:6].clamp_(-80, 80)
                parameters[6:9].clamp_(-0.5, 0.5)
                parameters[9:12].clamp_(-0.5, 0.5)
        with torch.no_grad():
            forward = affine_from_parameters(parameters, moving_center, fixed_center)
            warped, valid = _resample_world(
                moving, world_to_moving, target_world, fixed_level.shape, forward
            )
            level_correlation = float(_normcorr(warped, fixed_level, valid))
            if not np.isfinite(level_correlation):
                raise RuntimeError("affine optimization produced a non-finite loss")
            levels.append(
                {
                    "stride": stride,
                    "steps": iterations,
                    "learning_rate": learning_rate,
                    "correlation": level_correlation,
                    "overlap_fraction": float(valid.float().mean()),
                }
            )

    with torch.no_grad():
        forward = affine_from_parameters(parameters, moving_center, fixed_center)
        pull = torch.linalg.inv(forward)
        pull[3] = pull.new_tensor([0, 0, 0, 1])
        warped, valid = _resample_world(
            moving,
            world_to_moving,
            _world_grid(fixed_affine, fixed.shape),
            fixed.shape,
            forward,
        )
        correlation = float(_normcorr(warped, fixed, valid))
    qc = {
        "backend": "independent-pytorch-flirt-compatible",
        "equivalent_to_fsl_flirt": False,
        "forward_transform_convention": WORLD_FORWARD_CONVENTION,
        "pull_transform_convention": WORLD_PULL_CONVENTION,
        "accepts_fsl_flirt_matrix_directly": False,
        "degrees_of_freedom": 12,
        "parameter_center": "moving weighted centre of gravity",
        "translation_units": "world millimetres",
        "cost": "normalized correlation",
        "optimizer": "Adam",
        "schedule": "coarse-to-fine",
        "random_initialization": False,
        "fixed_iteration_count": True,
        "strides": list(strides),
        "correlation": correlation,
        "overlap_fraction": float(valid.float().mean()),
        "parameters": parameters.detach().cpu().tolist(),
        "levels": levels,
    }
    return LinearRegistrationResult(
        moving_to_fixed_world=forward.cpu().numpy().astype(np.float64),
        fixed_to_moving_world=pull.cpu().numpy().astype(np.float64),
        warped=warped.detach(),
        qc=qc,
    )


__all__ = [
    "LinearRegistrationResult",
    "WORLD_FORWARD_CONVENTION",
    "WORLD_PULL_CONVENTION",
    "affine_from_parameters",
    "flirt_to_world_affine",
    "flirt_to_world_pull",
    "register_affine",
    "resample_to_fixed",
    "voxel_to_fsl_scaled_mm",
]
