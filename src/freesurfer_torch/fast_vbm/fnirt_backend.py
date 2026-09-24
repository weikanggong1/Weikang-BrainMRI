"""CUDA-capable FNIRT-style cubic B-spline registration for FastVBM.

This module follows the parts of FNIRT that define the VBM transform: an
affine initialization, a cubic B-spline displacement field, an SSD data term,
bending-energy regularization, and Jacobian-range projection.  It uses PyTorch
autograd and Adam rather than FSL FNIRT's C++ optimizer and intensity model, so
the algorithm is not numerically equivalent to FSL FNIRT.
"""

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
import surfa as sf
import torch
import torch.nn.functional as F

from .legacy_registration import pull_jacobian
from .linear import voxel_to_fsl_scaled_mm, world_to_flirt_affine
from .synthmorph_backend import _world_affine, pull_jacobian_determinants


@dataclass
class FNIRTVBMResult:
    """Template-space images and the fixed-to-moving world-RAS pull warp."""

    moved: sf.Volume
    pull_transform: sf.Warp
    full_pull_jacobian: sf.Volume
    nonlinear_jacobian: sf.Volume
    modulated_gm: sf.Volume
    affine_pull_determinant: float
    qc: dict


def _per_level(value, count, name, cast):
    if np.isscalar(value):
        result = (cast(value),) * count
    else:
        result = tuple(cast(item) for item in value)
    if len(result) != count:
        raise ValueError(f"{name} must have one value per stride")
    return result


def _normalise(data):
    positive = data[data > 0]
    if positive.size == 0:
        raise ValueError("GM image is empty")
    high = float(np.percentile(positive, 99.5))
    return np.clip(np.maximum(data, 0) / max(high, 1e-6), 0, 1).astype(
        np.float32
    )


def _gaussian_blur(volume, fwhm_mm, voxel_sizes):
    if fwhm_mm <= 0:
        return volume
    result = volume
    for axis, voxel_size in enumerate(voxel_sizes):
        sigma = float(fwhm_mm) / (2.354820045 * float(voxel_size))
        if sigma < 0.05:
            continue
        radius = max(1, int(math.ceil(3 * sigma)))
        coordinate = torch.arange(
            -radius, radius + 1, dtype=volume.dtype, device=volume.device
        )
        kernel = torch.exp(-0.5 * (coordinate / sigma).square())
        kernel = kernel / kernel.sum()
        shape = [1, 1, 1, 1, 1]
        shape[axis + 2] = kernel.numel()
        padding = [0, 0, 0]
        padding[axis] = radius
        result = F.conv3d(result, kernel.reshape(shape), padding=tuple(padding))
    return result


def _bspline_axis(positions, knot_spacing_voxels, control_points):
    """Return a sparse cubic cardinal B-spline basis as a dense matrix."""
    if knot_spacing_voxels < 1 or control_points < 4:
        raise ValueError("invalid B-spline grid")
    coordinate = positions / float(knot_spacing_voxels)
    integer = torch.floor(coordinate).to(torch.long)
    fraction = coordinate - integer.to(coordinate.dtype)
    one_minus = 1 - fraction
    weights = torch.stack(
        (
            one_minus.pow(3) / 6,
            (3 * fraction.pow(3) - 6 * fraction.square() + 4) / 6,
            (-3 * fraction.pow(3) + 3 * fraction.square() + 3 * fraction + 1)
            / 6,
            fraction.pow(3) / 6,
        ),
        dim=1,
    )
    indices = integer[:, None] + torch.arange(4, device=positions.device)[None]
    if int(indices.min()) < 0 or int(indices.max()) >= control_points:
        raise ValueError("B-spline basis exceeds its control grid")
    basis = positions.new_zeros((positions.numel(), control_points))
    basis.scatter_add_(1, indices, weights)
    return basis


def cubic_bspline_field(coefficients, shape, knot_spacing_voxels, positions=None):
    """Expand FSL scaled-mm B-spline coefficients on a requested voxel grid.

    ``coefficients`` has shape ``[1, 3, Cx, Cy, Cz]``. ``positions`` contains
    full-resolution voxel indices for each output axis and defaults to every
    voxel.  The returned field has shape ``[1, 3, X, Y, Z]``.
    """
    if coefficients.ndim != 5 or tuple(coefficients.shape[:2]) != (1, 3):
        raise ValueError("coefficients must have shape [1, 3, Cx, Cy, Cz]")
    shape = tuple(int(size) for size in shape)
    spacing = tuple(int(value) for value in knot_spacing_voxels)
    if len(shape) != 3 or len(spacing) != 3:
        raise ValueError("shape and knot_spacing_voxels must contain three values")
    if positions is None:
        positions = tuple(
            torch.arange(size, dtype=coefficients.dtype, device=coefficients.device)
            for size in shape
        )
    if len(positions) != 3:
        raise ValueError("positions must contain one coordinate vector per axis")
    bases = tuple(
        _bspline_axis(axis, step, control)
        for axis, step, control in zip(positions, spacing, coefficients.shape[2:])
    )
    field = torch.einsum("xi,bcijk->bcxjk", bases[0], coefficients)
    field = torch.einsum("yj,bcxjk->bcxyk", bases[1], field)
    return torch.einsum("zk,bcxyk->bcxyz", bases[2], field)


def _coordinate_grid(affine, positions):
    axes = torch.meshgrid(*positions, indexing="ij")
    voxels = torch.stack(axes)
    return (
        torch.einsum("ij,jxyz->ixyz", affine[:3, :3], voxels)
        + affine[:3, 3, None, None, None]
    )[None]


def _sample(moving, moving_fsl2vox, target_fsl, fsl_pull, residual):
    source_fsl = (
        torch.einsum("ij,bjxyz->bixyz", fsl_pull[:3, :3], target_fsl)
        + fsl_pull[:3, 3][None, :, None, None, None]
        + residual
    )
    source_voxels = (
        torch.einsum("ij,bjxyz->bixyz", moving_fsl2vox[:3, :3], source_fsl)
        + moving_fsl2vox[:3, 3][None, :, None, None, None]
    )
    valid = torch.ones(
        source_voxels.shape[0:1] + source_voxels.shape[2:],
        dtype=torch.bool,
        device=moving.device,
    )
    normalised = []
    for axis, size in enumerate(moving.shape[2:]):
        valid &= (source_voxels[:, axis] >= 0) & (source_voxels[:, axis] <= size - 1)
        normalised.append(2 * source_voxels[:, axis] / (size - 1) - 1)
    grid = torch.stack(normalised[::-1], dim=-1)
    warped = F.grid_sample(
        moving, grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )
    return warped, valid, source_fsl


def _constrain_fnirt_field(
    affine_pull,
    field,
    fixed_voxel_to_fsl,
    jacobian_range,
    minimum_scale=1e-4,
):
    """Scale residual coefficients until the full FSL pull Jacobian is valid."""
    low, high = jacobian_range

    def evaluate(scale):
        scaled = field * scale
        full = pull_jacobian(
            affine_pull[:3, :3], scaled, fixed_voxel_to_fsl[:3, :3]
        )
        nonlinear = pull_jacobian(
            torch.eye(3, dtype=field.dtype, device=field.device),
            scaled,
            fixed_voxel_to_fsl[:3, :3],
        )
        valid = (
            bool(torch.isfinite(full).all())
            and bool(torch.isfinite(nonlinear).all())
            and float(full.min()) >= low + 1e-4
            and float(full.max()) <= high - 1e-4
        )
        return full, nonlinear, valid

    raw_full, raw_nonlinear, valid = evaluate(1.0)
    qc = {
        "raw_full_jacobian_min": float(raw_full.min()),
        "raw_full_jacobian_max": float(raw_full.max()),
        "raw_full_nonpositive_jacobian_voxels": int((raw_full <= 0).sum()),
        "raw_nonlinear_jacobian_min": float(raw_nonlinear.min()),
        "raw_nonlinear_jacobian_max": float(raw_nonlinear.max()),
        "raw_nonlinear_nonpositive_jacobian_voxels": int(
            (raw_nonlinear <= 0).sum()
        ),
        "jacobian_allowed_range": [low, high],
        "jacobian_constraint_scope": "full pull including affine",
    }
    if valid:
        qc["deformation_scale"] = 1.0
        return field, raw_full, raw_nonlinear, qc

    accepted = 0.0
    rejected = 1.0
    scale = 0.5
    while scale >= minimum_scale:
        _, _, valid = evaluate(scale)
        if valid:
            accepted = scale
            break
        rejected = scale
        scale *= 0.5
    if accepted == 0:
        zero_full, zero_nonlinear, valid = evaluate(0.0)
        if not valid:
            raise RuntimeError(
                "affine initialization violates the requested full Jacobian range"
            )
        accepted = 0.0
        full, nonlinear = zero_full, zero_nonlinear
    else:
        for _ in range(12):
            midpoint = (accepted + rejected) / 2
            _, _, valid = evaluate(midpoint)
            if valid:
                accepted = midpoint
            else:
                rejected = midpoint
        full, nonlinear, valid = evaluate(accepted)
        if not valid:
            raise RuntimeError("Jacobian became invalid after deformation backtracking")
    qc["deformation_scale"] = accepted
    return field * accepted, full, nonlinear, qc


def _linear_intensity_ssd(warped, fixed, mask):
    weight = mask.to(warped.dtype)
    mass = weight.sum().clamp_min(1)
    moving_mean = (warped * weight).sum() / mass
    fixed_mean = (fixed * weight).sum() / mass
    moving_centered = (warped - moving_mean) * weight
    fixed_centered = (fixed - fixed_mean) * weight
    scale = (
        (moving_centered * fixed_centered).sum()
        / moving_centered.square().sum().clamp_min(1e-8)
    ).clamp(0.25, 4.0)
    offset = fixed_mean - scale * moving_mean
    residual = (scale * warped + offset - fixed) * weight
    return residual.square().sum() / mass, scale, offset


def _bending_energy(coefficients, spacing_mm):
    terms = []
    for axis, step in enumerate(spacing_mm):
        if coefficients.shape[axis + 2] >= 3:
            terms.append(
                (coefficients.diff(n=2, dim=axis + 2) / (step * step))
                .square()
                .mean()
            )
    for first, second in ((0, 1), (0, 2), (1, 2)):
        if coefficients.shape[first + 2] >= 2 and coefficients.shape[second + 2] >= 2:
            mixed = coefficients.diff(dim=first + 2).diff(dim=second + 2)
            terms.append(
                2
                * (mixed / (spacing_mm[first] * spacing_mm[second]))
                .square()
                .mean()
            )
    return sum(terms, coefficients.new_zeros(()))


def _normalised_correlation(first, second, mask):
    weight = mask.to(first.dtype)
    mass = weight.sum().clamp_min(1)
    a = (first - (first * weight).sum() / mass) * weight
    b = (second - (second * weight).sum() / mass) * weight
    return (a * b).sum() / torch.sqrt(
        a.square().sum() * b.square().sum()
    ).clamp_min(1e-8)


class PyTorchFNIRTRegistration:
    """FNIRT-style cubic B-spline registration implemented with PyTorch.

    The public transform is a fixed-grid, target-to-source, world-RAS relative
    displacement.  This matches the internal sampling role required by
    FastVBM, while FSL coefficient files and FSL dense warp files remain
    separate file formats.
    """

    def __init__(
        self,
        *,
        device="cpu",
        strides: Sequence[int] = (4, 2, 1, 1),
        steps: Sequence[int] | int = (20, 20, 30, 20),
        learning_rates: Sequence[float] | float = (0.5, 0.25, 0.1, 0.05),
        input_fwhm_mm: Sequence[float] | float = (6.0, 4.0, 2.0, 2.0),
        reference_fwhm_mm: Sequence[float] | float = (4.0, 2.0, 0.0, 0.0),
        warp_resolution_mm=10.0,
        regularization: Sequence[float] | float = (150.0, 75.0, 50.0, 30.0),
        jacobian_penalty=1.0,
        jacobian_range=(0.2, 5.0),
        mask_threshold=1e-4,
    ):
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        self.strides = tuple(int(value) for value in strides)
        if not self.strides or any(value < 1 for value in self.strides):
            raise ValueError("strides must contain positive integers")
        count = len(self.strides)
        self.steps = _per_level(steps, count, "steps", int)
        self.learning_rates = _per_level(
            learning_rates, count, "learning_rates", float
        )
        self.input_fwhm_mm = _per_level(
            input_fwhm_mm, count, "input_fwhm_mm", float
        )
        self.reference_fwhm_mm = _per_level(
            reference_fwhm_mm, count, "reference_fwhm_mm", float
        )
        self.regularization = _per_level(
            regularization, count, "regularization", float
        )
        if any(value < 0 for value in self.steps):
            raise ValueError("steps must be non-negative")
        if any(value <= 0 for value in self.learning_rates):
            raise ValueError("learning_rates must be positive")
        if any(value < 0 for value in self.input_fwhm_mm + self.reference_fwhm_mm):
            raise ValueError("FWHM values must be non-negative")
        if any(value < 0 for value in self.regularization):
            raise ValueError("regularization must be non-negative")
        self.warp_resolution_mm = float(warp_resolution_mm)
        self.jacobian_penalty = float(jacobian_penalty)
        self.jacobian_range = tuple(float(value) for value in jacobian_range)
        self.mask_threshold = (
            None if mask_threshold is None else float(mask_threshold)
        )
        if self.warp_resolution_mm <= 0:
            raise ValueError("warp_resolution_mm must be positive")
        if self.jacobian_penalty < 0:
            raise ValueError("jacobian_penalty must be non-negative")
        if (
            len(self.jacobian_range) != 2
            or self.jacobian_range[0] <= 0
            or self.jacobian_range[0] >= self.jacobian_range[1]
        ):
            raise ValueError("jacobian_range must be an increasing positive pair")
        if self.mask_threshold is not None and self.mask_threshold < 0:
            raise ValueError("mask_threshold must be non-negative")

    def __call__(self, moving, fixed, moving_to_fixed):
        """Register GM after a moving-to-fixed affine initialization."""
        if not isinstance(moving, sf.Volume) or not isinstance(fixed, sf.Volume):
            raise TypeError("moving and fixed must be surfa.Volume objects")
        initial = _world_affine(moving_to_fixed, moving, fixed)
        moving_data = np.asarray(moving.data, dtype=np.float32)
        fixed_data = np.asarray(fixed.data, dtype=np.float32)
        if moving_data.ndim == 4 and moving_data.shape[-1] == 1:
            moving_data = moving_data[..., 0]
        if fixed_data.ndim == 4 and fixed_data.shape[-1] == 1:
            fixed_data = fixed_data[..., 0]
        if moving_data.ndim != 3 or fixed_data.ndim != 3:
            raise ValueError("moving and fixed must each contain one 3D frame")
        if min(moving_data.shape) < 2 or min(fixed_data.shape) < 2:
            raise ValueError("moving and fixed spatial dimensions must be at least 2")
        if not np.isfinite(moving_data).all() or not np.isfinite(fixed_data).all():
            raise ValueError("moving and fixed must contain only finite values")
        if not np.any(moving_data > 0) or not np.any(fixed_data > 0):
            raise ValueError("moving and fixed GM images must be nonempty")

        device = self.device
        moving_tensor = torch.from_numpy(moving_data.copy())[None, None].to(device)
        moving_normalised = torch.from_numpy(_normalise(moving_data))[None, None].to(
            device
        )
        fixed_normalised = torch.from_numpy(_normalise(fixed_data))[None, None].to(
            device
        )
        moving_affine = torch.as_tensor(
            np.array(moving.geom.vox2world.matrix, dtype=np.float32, copy=True),
            dtype=torch.float32,
            device=device,
        )
        fixed_affine = torch.as_tensor(
            np.array(fixed.geom.vox2world.matrix, dtype=np.float32, copy=True),
            dtype=torch.float32,
            device=device,
        )
        moving_fsl_array = voxel_to_fsl_scaled_mm(
            moving.geom.vox2world.matrix, moving_data.shape, moving.geom.voxsize
        )
        fixed_fsl_array = voxel_to_fsl_scaled_mm(
            fixed.geom.vox2world.matrix, fixed_data.shape, fixed.geom.voxsize
        )
        fsl_forward_array = world_to_flirt_affine(
            initial.matrix,
            moving.geom.vox2world.matrix,
            fixed.geom.vox2world.matrix,
            moving_data.shape,
            fixed_data.shape,
            moving.geom.voxsize,
            fixed.geom.voxsize,
        )
        moving_fsl = torch.as_tensor(
            moving_fsl_array, dtype=torch.float32, device=device
        )
        fixed_fsl = torch.as_tensor(
            fixed_fsl_array, dtype=torch.float32, device=device
        )
        fsl_forward = torch.as_tensor(
            fsl_forward_array, dtype=torch.float32, device=device
        )
        fsl_pull = torch.linalg.inv(fsl_forward)
        moving_fsl2vox = torch.linalg.inv(moving_fsl)

        voxel_sizes = torch.as_tensor(
            np.asarray(fixed.geom.voxsize, dtype=np.float32),
            dtype=torch.float32,
            device=device,
        )
        knot_voxels = tuple(
            max(1, int(math.floor(self.warp_resolution_mm / float(size))))
            for size in voxel_sizes
        )
        actual_spacing = tuple(
            float(step * size) for step, size in zip(knot_voxels, voxel_sizes)
        )
        fixed_shape = tuple(int(value) for value in fixed_data.shape)
        control_shape = tuple(
            math.floor((size - 1) / step) + 4
            for size, step in zip(fixed_shape, knot_voxels)
        )
        coefficients = torch.nn.Parameter(
            torch.zeros((1, 3, *control_shape), dtype=torch.float32, device=device)
        )

        levels = []
        affine_det = torch.linalg.det(fsl_pull[:3, :3])
        if not bool(torch.isfinite(affine_det)) or float(affine_det) <= 0:
            raise ValueError("the affine pull must have a positive finite determinant")
        moving_voxel_sizes = torch.as_tensor(
            np.asarray(moving.geom.voxsize, dtype=np.float32),
            dtype=torch.float32,
            device=device,
        )
        for (
            stride,
            iterations,
            learning_rate,
            regularization,
            input_fwhm,
            reference_fwhm,
        ) in zip(
            self.strides,
            self.steps,
            self.learning_rates,
            self.regularization,
            self.input_fwhm_mm,
            self.reference_fwhm_mm,
        ):
            positions = tuple(
                torch.arange(
                    0, size, stride, dtype=torch.float32, device=device
                )
                for size in fixed_shape
            )
            moving_level_source = _gaussian_blur(
                moving_normalised, input_fwhm, moving_voxel_sizes
            )
            fixed_level = _gaussian_blur(
                fixed_normalised, reference_fwhm, voxel_sizes
            )[
                :, :, ::stride, ::stride, ::stride
            ]
            target_fsl = _coordinate_grid(fixed_fsl, positions)
            level_linear = fixed_fsl[:3, :3] @ torch.diag(
                torch.full((3,), float(stride), device=device)
            )
            optimizer = torch.optim.Adam((coefficients,), lr=learning_rate)
            last_data = last_bending = last_penalty = None
            for _ in range(iterations):
                optimizer.zero_grad(set_to_none=True)
                field = cubic_bspline_field(
                    coefficients, fixed_shape, knot_voxels, positions
                )
                warped, valid, _ = _sample(
                    moving_level_source,
                    moving_fsl2vox,
                    target_fsl,
                    fsl_pull,
                    field,
                )
                mask = valid[:, None]
                if self.mask_threshold is not None:
                    mask = mask & (fixed_level > self.mask_threshold)
                mask = mask.expand_as(fixed_level)
                if int(mask.sum()) < 8:
                    raise RuntimeError("registration mask has fewer than eight voxels")
                data_loss, _, _ = _linear_intensity_ssd(
                    warped, fixed_level, mask
                )
                bending = _bending_energy(coefficients, actual_spacing)
                if self.jacobian_penalty:
                    full_jacobian = pull_jacobian(
                        fsl_pull[:3, :3], field, level_linear
                    )
                    low, high = self.jacobian_range
                    jacobian_loss = (
                        F.relu(low - full_jacobian).square().mean()
                        + F.relu(full_jacobian - high).square().mean()
                    )
                else:
                    jacobian_loss = coefficients.new_zeros(())
                loss = (
                    data_loss
                    + regularization * bending
                    + self.jacobian_penalty * jacobian_loss
                )
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("FNIRT-style optimization produced a non-finite loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_((coefficients,), max_norm=10.0)
                optimizer.step()
                last_data = data_loss.detach()
                last_bending = bending.detach()
                last_penalty = jacobian_loss.detach()

            with torch.no_grad():
                field = cubic_bspline_field(
                    coefficients, fixed_shape, knot_voxels, positions
                )
                warped, valid, _ = _sample(
                    moving_level_source,
                    moving_fsl2vox,
                    target_fsl,
                    fsl_pull,
                    field,
                )
                mask = valid[:, None]
                if self.mask_threshold is not None:
                    mask = mask & (fixed_level > self.mask_threshold)
                mask = mask.expand_as(fixed_level)
                correlation = float(
                    _normalised_correlation(warped, fixed_level, mask)
                )
                levels.append(
                    {
                        "stride": stride,
                        "steps": iterations,
                        "learning_rate": learning_rate,
                        "regularization": regularization,
                        "input_fwhm_mm": input_fwhm,
                        "reference_fwhm_mm": reference_fwhm,
                        "correlation": correlation,
                        "mask_fraction": float(mask.float().mean()),
                        "ssd": None if last_data is None else float(last_data),
                        "bending_energy": (
                            None if last_bending is None else float(last_bending)
                        ),
                        "jacobian_penalty": (
                            None if last_penalty is None else float(last_penalty)
                        ),
                    }
                )

        with torch.no_grad():
            positions = tuple(
                torch.arange(size, dtype=torch.float32, device=device)
                for size in fixed_shape
            )
            target_fsl = _coordinate_grid(fixed_fsl, positions)
            target_world = _coordinate_grid(fixed_affine, positions)
            raw_field = cubic_bspline_field(
                coefficients, fixed_shape, knot_voxels, positions
            )
            field, constrained_full, nonlinear_jacobian, constraint_qc = (
                _constrain_fnirt_field(
                    fsl_pull,
                    raw_field,
                    fixed_fsl,
                    self.jacobian_range,
                )
            )
            moved, valid, source_fsl = _sample(
                moving_tensor,
                moving_fsl2vox,
                target_fsl,
                fsl_pull,
                field,
            )
            normalised_moved, _, _ = _sample(
                moving_normalised,
                moving_fsl2vox,
                target_fsl,
                fsl_pull,
                field,
            )
            final_mask = valid[:, None]
            if self.mask_threshold is not None:
                final_mask = final_mask & (
                    fixed_normalised > self.mask_threshold
                )
            final_mask = final_mask.expand_as(fixed_normalised)
            _, final_scale, final_offset = _linear_intensity_ssd(
                normalised_moved, fixed_normalised, final_mask
            )
            final_correlation = float(
                _normalised_correlation(
                    normalised_moved, fixed_normalised, final_mask
                )
            )
            source_voxels = (
                torch.einsum(
                    "ij,bjxyz->bixyz", moving_fsl2vox[:3, :3], source_fsl
                )
                + moving_fsl2vox[:3, 3][None, :, None, None, None]
            )
            source_world = (
                torch.einsum(
                    "ij,bjxyz->bixyz", moving_affine[:3, :3], source_voxels
                )
                + moving_affine[:3, 3][None, :, None, None, None]
            )
            total_displacement = (source_world - target_world)[0].movedim(0, -1)

        displacement_array = total_displacement.cpu().numpy().astype(
            np.float32, copy=False
        )
        pull = sf.Warp(
            displacement_array,
            source=moving,
            target=fixed,
            format=sf.Warp.Format.disp_ras,
        )
        full, divided_affine_jacobian, affine_pull = pull_jacobian_determinants(
            displacement_array,
            fixed.geom.vox2world.matrix,
            initial.matrix,
            device=device,
        )
        fsl_to_world_determinant = (
            torch.linalg.det(
                moving_affine[:3, :3] @ torch.linalg.inv(moving_fsl[:3, :3])
            )
            * torch.linalg.det(
                fixed_fsl[:3, :3] @ torch.linalg.inv(fixed_affine[:3, :3])
            )
        )
        expected_world_full = constrained_full * fsl_to_world_determinant
        if not torch.allclose(full, expected_world_full, atol=2e-4, rtol=2e-4):
            raise RuntimeError("FSL-coordinate and world-RAS full Jacobians disagree")
        moved_array = moved[0, 0].cpu().numpy().astype(np.float32, copy=False)
        full_array = full.cpu().numpy().astype(np.float32, copy=False)
        nonlinear_array = nonlinear_jacobian.cpu().numpy().astype(
            np.float32, copy=False
        )
        moved_volume = fixed.new(moved_array)
        full_volume = fixed.new(full_array)
        nonlinear_volume = fixed.new(nonlinear_array)
        modulated = fixed.new(moved_array * nonlinear_array)
        qc = {
            "backend": "pytorch-fnirt-style-cubic-bspline",
            "fsl_fnirt_numerically_equivalent": False,
            "fast_vbm_output_role_compatible": True,
            "accepts_fsl_coefficient_file": False,
            "exports_fsl_coefficient_file": False,
            "data_term": "SSD after per-level global linear intensity fit",
            "optimizer": "Adam",
            "regularizer": "cubic B-spline coefficient bending energy",
            "internal_warp_convention": (
                "fixed-grid fixed-to-moving FSL scaled-mm residual displacement"
            ),
            "warp_convention": (
                "fixed-grid target-to-source disp-ras: "
                "source_world(target)-target_world"
            ),
            "output_jacobian_convention": (
                "FSL FNIRT nonlinear-only det(I + d residual_fsl / d fixed_fsl)"
            ),
            "full_pull_jacobian_convention": (
                "world-RAS determinant of the affine plus nonlinear pull"
            ),
            "requested_warp_resolution_mm": self.warp_resolution_mm,
            "actual_control_point_spacing_mm": list(actual_spacing),
            "control_grid_shape": list(control_shape),
            "strides": list(self.strides),
            "steps": list(self.steps),
            "learning_rates": list(self.learning_rates),
            "input_fwhm_mm": list(self.input_fwhm_mm),
            "reference_fwhm_mm": list(self.reference_fwhm_mm),
            "regularization": list(self.regularization),
            "jacobian_penalty_weight": self.jacobian_penalty,
            "jacobian_allowed_range": list(self.jacobian_range),
            "reference_mask_threshold": self.mask_threshold,
            "levels": levels,
            "final_correlation": final_correlation,
            "final_intensity_scale": float(final_scale),
            "final_intensity_offset": float(final_offset),
            "final_mask_fraction": float(final_mask.float().mean()),
            "affine_pull_jacobian_determinant": float(affine_pull),
            "full_pull_jacobian_min": float(full.min()),
            "full_pull_jacobian_max": float(full.max()),
            "full_divided_by_affine_jacobian_min": float(
                divided_affine_jacobian.min()
            ),
            "full_divided_by_affine_jacobian_max": float(
                divided_affine_jacobian.max()
            ),
            "nonlinear_jacobian_min": float(nonlinear_jacobian.min()),
            "nonlinear_jacobian_max": float(nonlinear_jacobian.max()),
            "nonpositive_full_pull_jacobian_voxels": int((full <= 0).sum()),
            "nonpositive_nonlinear_jacobian_voxels": int(
                (nonlinear_jacobian <= 0).sum()
            ),
            **constraint_qc,
        }
        return FNIRTVBMResult(
            moved=moved_volume,
            pull_transform=pull,
            full_pull_jacobian=full_volume,
            nonlinear_jacobian=nonlinear_volume,
            modulated_gm=modulated,
            affine_pull_determinant=float(affine_pull),
            qc=qc,
        )


__all__ = [
    "FNIRTVBMResult",
    "PyTorchFNIRTRegistration",
    "cubic_bspline_field",
]
