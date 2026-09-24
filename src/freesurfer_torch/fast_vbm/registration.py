"""PyTorch gray-matter registration, Jacobian estimation, and modulation."""

from dataclasses import dataclass
import math

import numpy as np
import surfa as sf
import torch
import torch.nn.functional as F


@dataclass
class VBMRegistrationResult:
    """Gray-matter maps on the template grid and registration diagnostics."""

    warped_gm: sf.Volume
    jacobian: sf.Volume
    modulated_gm: sf.Volume
    pull_world_affine: np.ndarray
    fit_score: float
    maximum_displacement_mm: float
    qc: dict


def _normalize(data):
    positive = data[data > 0]
    if positive.size == 0:
        raise ValueError("GM image is empty")
    high = float(np.percentile(positive, 99.5))
    return np.clip(np.maximum(data, 0) / max(high, 1e-6), 0, 1).astype(np.float32)


def _center_of_mass(volume, affine):
    values = volume[0, 0]
    mass = values.sum().clamp_min(1e-8)
    coordinates = []
    for axis, size in enumerate(values.shape):
        marginal = values.sum(dim=tuple(i for i in range(3) if i != axis))
        coordinates.append(
            (marginal * torch.arange(size, device=values.device)).sum() / mass
        )
    voxel = torch.stack(coordinates)
    return affine[:3, :3] @ voxel + affine[:3, 3]


def _level(fixed, affine, scale):
    full_shape = fixed.shape[2:]
    shape = tuple(max(4, math.ceil(n / scale)) for n in full_shape)
    small = F.interpolate(fixed, size=shape, mode="trilinear", align_corners=True)
    factors = torch.tensor(
        [(n - 1) / (m - 1) for n, m in zip(full_shape, shape)],
        device=fixed.device,
        dtype=fixed.dtype,
    )
    axes = torch.meshgrid(
        *[
            torch.arange(n, device=fixed.device, dtype=fixed.dtype)
            for n in shape
        ],
        indexing="ij",
    )
    voxels = torch.stack(axes, dim=0) * factors[:, None, None, None]
    world = torch.einsum("ij,jxyz->ixyz", affine[:3, :3], voxels)
    world = (world + affine[:3, 3, None, None, None])[None]
    level_linear = affine[:3, :3] @ torch.diag(factors)
    return small, world, level_linear


def _sample(
    moving,
    moving_world2vox,
    target_world,
    base,
    delta_linear,
    delta_offset,
    target_center,
    displacement,
):
    centered = target_world - target_center[None, :, None, None, None]
    source_world = (
        torch.einsum("ij,bjxyz->bixyz", base[:3, :3], target_world)
        + base[:3, 3][None, :, None, None, None]
        + torch.einsum("ij,bjxyz->bixyz", delta_linear, centered)
        + delta_offset[None, :, None, None, None]
        + displacement
    )
    voxels = (
        torch.einsum("ij,bjxyz->bixyz", moving_world2vox[:3, :3], source_world)
        + moving_world2vox[:3, 3][None, :, None, None, None]
    )
    normalized = [
        2 * voxels[:, i] / (size - 1) - 1
        for i, size in enumerate(moving.shape[2:])
    ]
    grid = torch.stack(normalized[::-1], dim=-1)
    return F.grid_sample(
        moving, grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )


def _image_loss(warped, fixed):
    a, b = warped.flatten(), fixed.flatten()
    a, b = a - a.mean(), b - b.mean()
    correlation = (a * b).mean() / (
        a.square().mean() * b.square().mean()
    ).sqrt().clamp_min(1e-8)
    return 1 - correlation + 0.2 * (warped - fixed).square().mean()


def pull_jacobian(linear, displacement, target_vox2world_linear):
    """Return the full pull-map determinant in world coordinates."""
    derivatives = []
    for component in range(3):
        derivatives.append(
            torch.stack(
                torch.gradient(displacement[0, component], dim=(0, 1, 2)), dim=0
            )
        )
    voxel_gradient = torch.stack(derivatives, dim=0)
    world_gradient = torch.einsum(
        "abxyz,bc->acxyz",
        voxel_gradient,
        torch.linalg.inv(target_vox2world_linear),
    )
    j = linear[:, :, None, None, None] + world_gradient
    return (
        j[0, 0] * (j[1, 1] * j[2, 2] - j[1, 2] * j[2, 1])
        - j[0, 1] * (j[1, 0] * j[2, 2] - j[1, 2] * j[2, 0])
        + j[0, 2] * (j[1, 0] * j[2, 1] - j[1, 1] * j[2, 0])
    )


def constrain_deformation(
    linear,
    field,
    target_vox2world_linear,
    min_jac=0.2,
    max_jac=5.0,
    min_scale=0.05,
):
    """Scale a deformation field until its nonlinear Jacobian is admissible."""
    affine_det = torch.linalg.det(linear)
    if (
        not torch.isfinite(affine_det)
        or affine_det <= 0
        or not torch.isfinite(field).all()
    ):
        raise RuntimeError("non-finite field or nonpositive affine determinant")

    def evaluate(scale):
        jac = (
            pull_jacobian(linear, field * scale, target_vox2world_linear)
            / affine_det
        )
        valid = (
            bool(torch.isfinite(jac).all())
            and float(jac.min()) >= min_jac + 1e-4
            and float(jac.max()) <= max_jac - 1e-4
        )
        return jac, valid

    raw, valid = evaluate(1.0)
    report = {
        "raw_jacobian_min": float(raw.min()),
        "raw_jacobian_max": float(raw.max()),
        "raw_nonpositive_jacobian_voxels": int((raw <= 0).sum()),
        "jacobian_allowed_range": [min_jac, max_jac],
    }
    if valid:
        report["deformation_scale"] = 1.0
        return field, raw, report

    rejected = 1.0
    accepted = None
    candidate = 0.5
    while candidate >= min_scale:
        jac, valid = evaluate(candidate)
        if valid:
            accepted = candidate
            break
        rejected = candidate
        candidate *= 0.5
    if accepted is None:
        jac, valid = evaluate(min_scale)
        if valid:
            accepted = min_scale
        else:
            raise RuntimeError(
                f"Jacobian range [{min_jac}, {max_jac}] cannot be met with "
                f"deformation scale >= {min_scale}; raw min="
                f"{report['raw_jacobian_min']:.4g}, raw max="
                f"{report['raw_jacobian_max']:.4g}"
            )

    for _ in range(10):
        midpoint = (accepted + rejected) / 2
        _, valid = evaluate(midpoint)
        if valid:
            accepted = midpoint
        else:
            rejected = midpoint
    scaled = field * accepted
    jac, valid = evaluate(accepted)
    if not valid:
        raise RuntimeError("Jacobian became invalid after deformation backtracking")
    report["deformation_scale"] = accepted
    return scaled, jac, report


def register(
    moving,
    fixed,
    moving_affine,
    fixed_affine,
    *,
    initial_pull=None,
    scales=(4, 2, 1),
    affine_steps=50,
    deform_steps=40,
    smoothness=0.5,
    qc=None,
):
    """Register N,C,X,Y,Z tensors and return warped GM, Jacobian, and transform."""
    device = moving.device
    if any(n < 4 for n in moving.shape[2:] + fixed.shape[2:]):
        raise ValueError("all image dimensions must be at least four voxels")
    if affine_steps < 0 or deform_steps < 0 or smoothness < 0:
        raise ValueError("steps and smoothness must be non-negative")
    if not scales or any(not isinstance(scale, int) or scale < 1 for scale in scales):
        raise ValueError("scales must contain positive integers")
    m_np = moving[0, 0].detach().cpu().numpy()
    f_np = fixed[0, 0].detach().cpu().numpy()
    m_norm = torch.from_numpy(_normalize(m_np))[None, None].to(device)
    f_norm = torch.from_numpy(_normalize(f_np))[None, None].to(device)
    source_center = _center_of_mass(m_norm, moving_affine)
    target_center = _center_of_mass(f_norm, fixed_affine)
    base = torch.eye(4, device=device, dtype=torch.float32)
    if initial_pull is None:
        base[:3, 3] = source_center - target_center
    else:
        if initial_pull.shape != (4, 4):
            raise ValueError("initial_pull must have shape 4x4")
        base[:] = initial_pull
    moving_world2vox = torch.linalg.inv(moving_affine)
    delta_linear = torch.nn.Parameter(torch.zeros((3, 3), device=device))
    delta_offset = torch.nn.Parameter(torch.zeros(3, device=device))
    for scale in (scale for scale in scales if scale >= 2):
        target, world, _ = _level(f_norm, fixed_affine, scale)
        optimizer = torch.optim.Adam(
            [
                {"params": [delta_linear], "lr": 0.002},
                {"params": [delta_offset], "lr": 0.5},
            ]
        )
        for _ in range(affine_steps):
            optimizer.zero_grad(set_to_none=True)
            warped = _sample(
                m_norm,
                moving_world2vox,
                world,
                base,
                delta_linear,
                delta_offset,
                target_center,
                0,
            )
            loss = (
                _image_loss(warped, target)
                + 0.05 * delta_linear.square().sum()
                + 1e-4 * delta_offset.square().sum()
            )
            loss.backward()
            optimizer.step()

    control = None
    for scale in scales:
        target, world, level_linear = _level(f_norm, fixed_affine, scale)
        shape = target.shape[2:]
        control_shape = tuple(max(4, math.ceil(n / 4)) for n in shape)
        initial = (
            torch.zeros((1, 3, *control_shape), device=device)
            if control is None
            else F.interpolate(
                control.detach(),
                size=control_shape,
                mode="trilinear",
                align_corners=True,
            )
        )
        control = torch.nn.Parameter(initial)
        optimizer = torch.optim.Adam([control], lr=0.3)
        spacing = torch.linalg.vector_norm(level_linear, dim=0)
        for _ in range(deform_steps):
            optimizer.zero_grad(set_to_none=True)
            field = F.interpolate(
                control, size=shape, mode="trilinear", align_corners=True
            )
            warped = _sample(
                m_norm,
                moving_world2vox,
                world,
                base,
                delta_linear.detach(),
                delta_offset.detach(),
                target_center,
                field,
            )
            gradients = [
                (field.diff(dim=axis + 2) / spacing[axis]).square().mean()
                for axis in range(3)
            ]
            loss = (
                _image_loss(warped, target)
                + smoothness * sum(gradients) / 3
                + 1e-4 * field.square().mean()
            )
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        _, world, _ = _level(f_norm, fixed_affine, 1)
        field = F.interpolate(
            control,
            size=fixed.shape[2:],
            mode="trilinear",
            align_corners=True,
        ).detach()
        original_norm = _sample(
            m_norm,
            moving_world2vox,
            world,
            base,
            delta_linear,
            delta_offset,
            target_center,
            field,
        )
        original_score = float(1 - _image_loss(original_norm, f_norm).item())
        field, jacobian, deformation_qc = constrain_deformation(
            base[:3, :3] + delta_linear,
            field,
            fixed_affine[:3, :3],
        )
        warped = _sample(
            moving,
            moving_world2vox,
            world,
            base,
            delta_linear,
            delta_offset,
            target_center,
            field,
        )
        fitted_norm = _sample(
            m_norm,
            moving_world2vox,
            world,
            base,
            delta_linear,
            delta_offset,
            target_center,
            field,
        )
        affine = base.clone()
        affine[:3, :3] += delta_linear
        affine[:3, 3] += delta_offset - delta_linear @ target_center
        score = float(1 - _image_loss(fitted_norm, f_norm).item())
        if qc is not None:
            qc.update(deformation_qc)
            qc["fit_score_before_jacobian_constraint"] = original_score
    return warped[0, 0], jacobian, field[0], affine, score


def _array(volume, name):
    if not isinstance(volume, sf.Volume):
        raise TypeError(f"{name} must be a surfa.Volume")
    data = np.asarray(volume.data)
    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"{name} must contain one 3D frame")
    if not np.isfinite(data).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return np.asarray(data, dtype=np.float32)


def _affine(volume, name):
    affine = np.asarray(volume.geom.vox2world.matrix, dtype=np.float64)
    if affine.shape != (4, 4) or not np.isfinite(affine).all():
        raise ValueError(f"{name} affine must be a finite 4x4 matrix")
    linear = affine[:3, :3]
    voxel_size = np.linalg.norm(linear, axis=0)
    if np.any(voxel_size <= 0) or abs(np.linalg.det(linear)) < 1e-8:
        raise ValueError(f"{name} affine must be invertible with positive voxel sizes")
    return affine.copy()


def register_gm(
    moving,
    fixed,
    *,
    device="cpu",
    initial_pull=None,
    scales=(4, 2, 1),
    affine_steps=50,
    deform_steps=40,
    smoothness=10.0,
):
    """Register a GM PVE volume to a GM template and compute modulation."""
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    moving_data = _array(moving, "moving")
    fixed_data = _array(fixed, "fixed")
    moving_affine_array = _affine(moving, "moving")
    fixed_affine_array = _affine(fixed, "fixed")
    moving_tensor = torch.from_numpy(moving_data.copy())[None, None].to(device)
    fixed_tensor = torch.from_numpy(fixed_data.copy())[None, None].to(device)
    moving_affine = torch.as_tensor(
        moving_affine_array, dtype=torch.float32, device=device
    )
    fixed_affine = torch.as_tensor(
        fixed_affine_array, dtype=torch.float32, device=device
    )
    if initial_pull is not None:
        initial_pull = torch.as_tensor(
            initial_pull, dtype=torch.float32, device=device
        )
    qc = {}
    warped, jacobian, field, affine, score = register(
        moving_tensor,
        fixed_tensor,
        moving_affine,
        fixed_affine,
        initial_pull=initial_pull,
        scales=scales,
        affine_steps=affine_steps,
        deform_steps=deform_steps,
        smoothness=smoothness,
        qc=qc,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    def volume(tensor):
        return fixed.new(tensor.detach().cpu().numpy().astype(np.float32, copy=False))

    return VBMRegistrationResult(
        warped_gm=volume(warped),
        jacobian=volume(jacobian),
        modulated_gm=volume(warped * jacobian),
        pull_world_affine=affine.detach().cpu().numpy(),
        fit_score=score,
        maximum_displacement_mm=float(torch.linalg.vector_norm(field, dim=0).max()),
        qc={
            **qc,
            "affine_jacobian_determinant": float(
                torch.linalg.det(affine[:3, :3])
            ),
            "jacobian_min": float(jacobian.min()),
            "jacobian_max": float(jacobian.max()),
            "nonpositive_jacobian_voxels": int((jacobian <= 0).sum()),
        },
    )


__all__ = [
    "VBMRegistrationResult",
    "constrain_deformation",
    "pull_jacobian",
    "register",
    "register_gm",
]
