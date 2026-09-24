#!/usr/bin/env python3
"""Experimental GPU GM-to-template VBM registration.

This is an intensity-based PyTorch alternative to UKB bb_vbm's fsl_reg/FNIRT
step, not a port of FNIRT. The input must already be a GM probability or PVE
image; it can come from FAST or the experimental SynthSeg-derived estimator.
"""

import argparse
import json
import math
from pathlib import Path
import time

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F


def _v2fsl(image):
    """Voxel -> FSL scaled-mm coordinates (including FSL's handedness flip)."""
    matrix = np.diag([*image.header.get_zooms()[:3], 1.0])
    if np.linalg.det(image.affine[:3, :3]) > 0:
        flip = np.eye(4)
        flip[0, 0] = -1
        flip[0, 3] = image.shape[0] - 1
        matrix = matrix @ flip
    return matrix


def flirt_pull_world(path, moving_image, fixed_image):
    """Convert a moving->fixed FLIRT .mat to fixed-world->moving-world."""
    flirt = np.loadtxt(path)
    if flirt.shape != (4, 4):
        raise ValueError("FLIRT matrix must have shape 4x4")
    moving_to_fixed = (fixed_image.affine @ np.linalg.inv(_v2fsl(fixed_image))
                       @ flirt @ _v2fsl(moving_image)
                       @ np.linalg.inv(moving_image.affine))
    return np.linalg.inv(moving_to_fixed)


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
        coordinates.append((marginal * torch.arange(size, device=values.device)).sum() / mass)
    voxel = torch.stack(coordinates)
    return affine[:3, :3] @ voxel + affine[:3, 3]


def _level(fixed, affine, scale):
    full_shape = fixed.shape[2:]
    shape = tuple(max(4, math.ceil(n / scale)) for n in full_shape)
    small = F.interpolate(fixed, size=shape, mode="trilinear", align_corners=True)
    factors = torch.tensor([(n - 1) / (m - 1) for n, m in zip(full_shape, shape)],
                           device=fixed.device, dtype=fixed.dtype)
    axes = torch.meshgrid(*[torch.arange(n, device=fixed.device, dtype=fixed.dtype)
                            for n in shape], indexing="ij")
    voxels = torch.stack(axes, dim=0) * factors[:, None, None, None]
    world = torch.einsum("ij,jxyz->ixyz", affine[:3, :3], voxels)
    world = (world + affine[:3, 3, None, None, None])[None]
    level_linear = affine[:3, :3] @ torch.diag(factors)
    return small, world, level_linear


def _sample(moving, moving_world2vox, target_world, base, delta_linear, delta_offset,
            target_center, displacement):
    centered = target_world - target_center[None, :, None, None, None]
    source_world = (torch.einsum("ij,bjxyz->bixyz", base[:3, :3], target_world)
                    + base[:3, 3][None, :, None, None, None]
                    + torch.einsum("ij,bjxyz->bixyz", delta_linear, centered)
                    + delta_offset[None, :, None, None, None] + displacement)
    voxels = (torch.einsum("ij,bjxyz->bixyz", moving_world2vox[:3, :3], source_world)
              + moving_world2vox[:3, 3][None, :, None, None, None])
    normalized = [2 * voxels[:, i] / (size - 1) - 1
                  for i, size in enumerate(moving.shape[2:])]
    grid = torch.stack(normalized[::-1], dim=-1)
    return F.grid_sample(moving, grid, mode="bilinear", padding_mode="zeros",
                         align_corners=True)


def _image_loss(warped, fixed):
    a, b = warped.flatten(), fixed.flatten()
    a, b = a - a.mean(), b - b.mean()
    correlation = (a * b).mean() / (a.square().mean() * b.square().mean()).sqrt().clamp_min(1e-8)
    return 1 - correlation + 0.2 * (warped - fixed).square().mean()


def pull_jacobian(linear, displacement, target_vox2world_linear):
    """Full det(d moving-world / d fixed-world), including the affine."""
    derivatives = []
    for component in range(3):
        derivatives.append(torch.stack(torch.gradient(displacement[0, component],
                                                       dim=(0, 1, 2)), dim=0))
    voxel_gradient = torch.stack(derivatives, dim=0)
    world_gradient = torch.einsum("abxyz,bc->acxyz", voxel_gradient,
                                  torch.linalg.inv(target_vox2world_linear))
    j = linear[:, :, None, None, None] + world_gradient
    return (j[0, 0] * (j[1, 1] * j[2, 2] - j[1, 2] * j[2, 1])
            - j[0, 1] * (j[1, 0] * j[2, 2] - j[1, 2] * j[2, 0])
            + j[0, 2] * (j[1, 0] * j[2, 1] - j[1, 1] * j[2, 0]))


def constrain_deformation(linear, field, target_vox2world_linear,
                          min_jac=0.2, max_jac=5.0, min_scale=0.05):
    """Backtrack the whole field until its nonlinear Jacobian is admissible.

    Scaling a field keeps its spatial pattern but weakens registration. A very
    small accepted scale is treated as failed registration, not a valid VBM map.
    No Jacobian voxel is clipped or replaced independently.
    """
    affine_det = torch.linalg.det(linear)
    if not torch.isfinite(affine_det) or affine_det <= 0 or not torch.isfinite(field).all():
        raise RuntimeError("non-finite field or nonpositive affine determinant")

    def evaluate(scale):
        jac = pull_jacobian(linear, field * scale, target_vox2world_linear) / affine_det
        valid = (bool(torch.isfinite(jac).all()) and
                 float(jac.min()) >= min_jac + 1e-4 and
                 float(jac.max()) <= max_jac - 1e-4)
        return jac, valid

    raw, valid = evaluate(1.0)
    report = {"raw_jacobian_min": float(raw.min()),
              "raw_jacobian_max": float(raw.max()),
              "raw_nonpositive_jacobian_voxels": int((raw <= 0).sum()),
              "jacobian_allowed_range": [min_jac, max_jac]}
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
                f"deformation scale >= {min_scale}; raw min={report['raw_jacobian_min']:.4g}, "
                f"raw max={report['raw_jacobian_max']:.4g}")

    # Search towards the original field. Always retain the last valid scale;
    # determinant bounds need not vary monotonically over the entire interval.
    for _ in range(10):
        midpoint = (accepted + rejected) / 2
        jac, valid = evaluate(midpoint)
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


def register(moving, fixed, moving_affine, fixed_affine, *, initial_pull=None,
             scales=(4, 2, 1), affine_steps=50, deform_steps=40, smoothness=0.5,
             qc=None):
    """Return warped original PVE, pull Jacobian, displacement, affine, scores.

    Arrays are torch N,C,X,Y,Z tensors. Affines and initial_pull are torch 4x4
    matrices in RAS millimetres. Optimization and sampling run on their device.
    """
    device = moving.device
    if any(n < 4 for n in moving.shape[2:] + fixed.shape[2:]):
        raise ValueError("all image dimensions must be at least four voxels")
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
        base[:] = initial_pull
    moving_world2vox = torch.linalg.inv(moving_affine)
    delta_linear = torch.nn.Parameter(torch.zeros((3, 3), device=device))
    delta_offset = torch.nn.Parameter(torch.zeros(3, device=device))
    affine_scales = tuple(s for s in scales if s >= 2)
    for scale in affine_scales:
        target, world, _ = _level(f_norm, fixed_affine, scale)
        optimizer = torch.optim.Adam([{"params": [delta_linear], "lr": 0.002},
                                      {"params": [delta_offset], "lr": 0.5}])
        for _ in range(affine_steps):
            optimizer.zero_grad(set_to_none=True)
            warped = _sample(m_norm, moving_world2vox, world, base, delta_linear,
                             delta_offset, target_center, 0)
            loss = (_image_loss(warped, target) + 0.05 * delta_linear.square().sum()
                    + 1e-4 * delta_offset.square().sum())
            loss.backward()
            optimizer.step()

    control = None
    for scale in scales:
        target, world, level_linear = _level(f_norm, fixed_affine, scale)
        shape = target.shape[2:]
        control_shape = tuple(max(4, math.ceil(n / 4)) for n in shape)
        initial = (torch.zeros((1, 3, *control_shape), device=device) if control is None
                   else F.interpolate(control.detach(), size=control_shape,
                                      mode="trilinear", align_corners=True))
        control = torch.nn.Parameter(initial)
        optimizer = torch.optim.Adam([control], lr=0.3)
        spacing = torch.linalg.vector_norm(level_linear, dim=0)
        for _ in range(deform_steps):
            optimizer.zero_grad(set_to_none=True)
            field = F.interpolate(control, size=shape, mode="trilinear", align_corners=True)
            warped = _sample(m_norm, moving_world2vox, world, base, delta_linear.detach(),
                             delta_offset.detach(), target_center, field)
            gradients = [(field.diff(dim=axis + 2) / spacing[axis]).square().mean()
                         for axis in range(3)]
            loss = (_image_loss(warped, target) + smoothness * sum(gradients) / 3
                    + 1e-4 * field.square().mean())
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        _, world, _ = _level(f_norm, fixed_affine, 1)
        field = F.interpolate(control, size=fixed.shape[2:], mode="trilinear",
                              align_corners=True).detach()
        original_norm = _sample(m_norm, moving_world2vox, world, base, delta_linear,
                                delta_offset, target_center, field)
        original_score = float(1 - _image_loss(original_norm, f_norm).item())
        field, jacobian, deformation_qc = constrain_deformation(
            base[:3, :3] + delta_linear, field, fixed_affine[:3, :3])
        warped = _sample(moving, moving_world2vox, world, base, delta_linear,
                         delta_offset, target_center, field)
        fitted_norm = _sample(m_norm, moving_world2vox, world, base, delta_linear,
                              delta_offset, target_center, field)
        affine = base.clone()
        affine[:3, :3] += delta_linear
        affine[:3, 3] += delta_offset - delta_linear @ target_center
        score = float(1 - _image_loss(fitted_norm, f_norm).item())
        if qc is not None:
            qc.update(deformation_qc)
            qc["fit_score_before_jacobian_constraint"] = original_score
    return warped[0, 0], jacobian, field[0], affine, score


def _save(data, fixed_image, path):
    image = nib.Nifti1Image(data.detach().cpu().numpy().astype(np.float32),
                            fixed_image.affine)
    image.set_sform(fixed_image.affine, int(fixed_image.header["sform_code"]) or 1)
    image.set_qform(fixed_image.affine, int(fixed_image.header["qform_code"]) or 1)
    nib.save(image, str(path))


def self_test(device):
    shape = (32, 32, 32)
    axes = torch.meshgrid(*[torch.arange(n, device=device) for n in shape], indexing="ij")
    coords = torch.stack(axes).float()
    target = torch.exp(-((coords - torch.tensor([15, 16, 17], device=device)[:, None, None, None])
                         .square().sum(0) / 32))[None, None]
    source = torch.exp(-((coords - torch.tensor([18, 16, 17], device=device)[:, None, None, None])
                         .square().sum(0) / 32))[None, None]
    identity = torch.eye(4, device=device)
    zeros = torch.zeros((1, 3, *shape), device=device)
    j0 = pull_jacobian(identity[:3, :3], zeros, identity[:3, :3])
    scaled = identity[:3, :3].clone()
    scaled[0, 0] = 1.2
    js = pull_jacobian(scaled, zeros, identity[:3, :3])
    assert torch.allclose(j0, torch.ones_like(j0), atol=1e-6)
    assert torch.allclose(js, torch.full_like(js, 1.2), atol=1e-6)
    strong_field = zeros.clone()
    strong_field[:, 0] = 24 * torch.sin(coords[0] * (2 * math.pi / (shape[0] - 1)))
    safe_field, safe_jac, constraint = constrain_deformation(
        identity[:3, :3], strong_field, identity[:3, :3])
    assert constraint["raw_jacobian_min"] < 0.2
    assert constraint["raw_jacobian_max"] > 5
    assert 0.05 <= constraint["deformation_scale"] < 1
    assert float(safe_jac.min()) >= 0.2 and float(safe_jac.max()) <= 5
    assert torch.allclose(safe_field, strong_field * constraint["deformation_scale"])
    try:
        constrain_deformation(identity[:3, :3], strong_field, identity[:3, :3],
                              min_scale=0.8)
    except RuntimeError:
        pass
    else:
        raise AssertionError("an inadmissible minimum scale must fail")
    scaled_affine = identity.clone()
    scaled_affine[0, 0] = 1.2
    _, jac_affine_only, _, _, _ = register(
        source, target, identity, identity, initial_pull=scaled_affine,
        scales=(1,), affine_steps=0, deform_steps=0)
    assert torch.allclose(jac_affine_only, torch.ones_like(jac_affine_only), atol=1e-6)
    warped, jac, _, affine, _ = register(source, target, identity, identity,
                                         scales=(2, 1), affine_steps=12,
                                         deform_steps=6, smoothness=0.5)
    before = _image_loss(source, target).item()
    after = _image_loss(warped[None, None], target).item()
    assert after < before * 0.2, (before, after)
    assert abs(affine[0, 3].item() - 3) < 0.75, affine[0, 3].item()
    assert torch.isfinite(jac).all() and (jac > 0).all()
    print(json.dumps({"translation_mm": float(affine[0, 3]),
                      "loss_before": before, "loss_after": after,
                      "jacobian_identity": float(j0.mean()),
                      "full_jacobian_scale_1p2": float(js.mean()),
                      "modulation_jacobian_affine_only": float(jac_affine_only.mean()),
                      "raw_fold_min": constraint["raw_jacobian_min"],
                      "raw_expansion_max": constraint["raw_jacobian_max"],
                      "accepted_deformation_scale": constraint["deformation_scale"],
                      "constrained_jacobian_min": float(safe_jac.min()),
                      "constrained_jacobian_max": float(safe_jac.max()),
                      "device": str(device)}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--moving", help="moving GM probability/PVE NIfTI")
    parser.add_argument("--fixed", help="UKB template_GM.nii.gz")
    parser.add_argument("--output-prefix", help="prefix for three NIfTI files and JSON")
    parser.add_argument("--init-flirt", help="optional moving-to-fixed FSL FLIRT .mat")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int,
                        help="PyTorch CPU threads used by this registration process")
    parser.add_argument("--affine-steps", type=int, default=50)
    parser.add_argument("--deform-steps", type=int, default=40)
    parser.add_argument("--smoothness", type=float, default=0.5)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.threads is not None and args.threads < 1:
        parser.error("--threads must be positive")
    if args.threads is not None:
        torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use --device cpu for the self-test")
    if args.self_test:
        self_test(device)
        return
    if not all((args.moving, args.fixed, args.output_prefix)):
        parser.error("--moving, --fixed and --output-prefix are required")
    if args.affine_steps < 0 or args.deform_steps < 0 or args.smoothness < 0:
        parser.error("steps and smoothness must be non-negative")
    started = time.perf_counter()
    moving_image, fixed_image = nib.load(args.moving), nib.load(args.fixed)
    if len(moving_image.shape) != 3 or len(fixed_image.shape) != 3:
        parser.error("both inputs must be 3D NIfTI images")
    moving = torch.from_numpy(moving_image.get_fdata(dtype=np.float32).copy())[None, None].to(device)
    fixed = torch.from_numpy(fixed_image.get_fdata(dtype=np.float32).copy())[None, None].to(device)
    if not torch.isfinite(moving).all() or not torch.isfinite(fixed).all():
        parser.error("inputs contain NaN or infinity")
    m_aff = torch.as_tensor(moving_image.affine, dtype=torch.float32, device=device)
    f_aff = torch.as_tensor(fixed_image.affine, dtype=torch.float32, device=device)
    initial = None if args.init_flirt is None else torch.as_tensor(
        flirt_pull_world(args.init_flirt, moving_image, fixed_image),
        dtype=torch.float32, device=device)
    deformation_qc = {}
    warped, jac, field, affine, score = register(
        moving, fixed, m_aff, f_aff, initial_pull=initial,
        affine_steps=args.affine_steps, deform_steps=args.deform_steps,
        smoothness=args.smoothness, qc=deformation_qc)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    paths = {name: str(prefix) + suffix for name, suffix in {
        "warped_gm": "_warped_gm.nii.gz", "jacobian": "_jacobian.nii.gz",
        "modulated_gm": "_modulated_gm.nii.gz"}.items()}
    for name, volume in (("warped_gm", warped), ("jacobian", jac),
                         ("modulated_gm", warped * jac)):
        _save(volume, fixed_image, paths[name])
    report = {"method": ("PyTorch multiscale global normalized correlation + 0.2 MSE, "
                         "sparse displacement control grid; not FNIRT"),
              "device": str(device), "seconds_including_io": time.perf_counter() - started,
              "moving": str(Path(args.moving).resolve()),
              "fixed": str(Path(args.fixed).resolve()),
              "init_flirt": args.init_flirt,
              "settings": {"affine_steps_per_scale": args.affine_steps,
                           "deform_steps_per_scale": args.deform_steps,
                           "smoothness": args.smoothness, "scales": [4, 2, 1],
                           "torch_threads": torch.get_num_threads()},
              "fit_score_1_minus_loss": score,
              "pull_world_affine": affine.cpu().numpy().tolist(),
              "affine_jacobian_determinant": float(torch.linalg.det(affine[:3, :3])),
              "jacobian_convention": "nonlinear-only: full pull determinant divided by affine determinant",
              "maximum_displacement_mm": float(torch.linalg.vector_norm(field, dim=0).max()),
              "jacobian_min": float(jac.min()), "jacobian_max": float(jac.max()),
              "nonpositive_jacobian_voxels": int((jac <= 0).sum()),
              **deformation_qc,
              "outputs": paths}
    report_path = str(prefix) + "_report.json"
    Path(report_path).write_text(json.dumps(report, indent=2) + "\n")
    print(report_path)


if __name__ == "__main__":
    main()
