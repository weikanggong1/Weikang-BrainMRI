#!/usr/bin/env python3
"""Compatibility CLI for the retired experimental VBM registration.

This command preserves the earlier research workflow's file names and private
report schema. The stable package uses ``FastVBM`` and does not import this
implementation.
"""

import argparse
import json
import math
from pathlib import Path
import time

import nibabel as nib
import numpy as np
import torch

from _legacy_registration import (
    constrain_deformation,
    pull_jacobian,
    register,
)


def _v2fsl(image):
    """Voxel to FSL scaled-mm coordinates, including FSL's handedness flip."""
    matrix = np.diag([*image.header.get_zooms()[:3], 1.0])
    if np.linalg.det(image.affine[:3, :3]) > 0:
        flip = np.eye(4)
        flip[0, 0] = -1
        flip[0, 3] = image.shape[0] - 1
        matrix = matrix @ flip
    return matrix


def flirt_pull_world(path, moving_image, fixed_image):
    """Convert a moving-to-fixed FLIRT matrix to fixed-world-to-moving-world."""
    flirt = np.loadtxt(path)
    if flirt.shape != (4, 4):
        raise ValueError("FLIRT matrix must have shape 4x4")
    moving_to_fixed = (
        fixed_image.affine
        @ np.linalg.inv(_v2fsl(fixed_image))
        @ flirt
        @ _v2fsl(moving_image)
        @ np.linalg.inv(moving_image.affine)
    )
    return np.linalg.inv(moving_to_fixed)


def _save(data, fixed_image, path):
    image = nib.Nifti1Image(
        data.detach().cpu().numpy().astype(np.float32), fixed_image.affine
    )
    image.set_sform(fixed_image.affine, int(fixed_image.header["sform_code"]) or 1)
    image.set_qform(fixed_image.affine, int(fixed_image.header["qform_code"]) or 1)
    nib.save(image, str(path))


def _image_loss_for_test(warped, fixed):
    a, b = warped.flatten(), fixed.flatten()
    a, b = a - a.mean(), b - b.mean()
    correlation = (a * b).mean() / (
        a.square().mean() * b.square().mean()
    ).sqrt().clamp_min(1e-8)
    return float((1 - correlation + 0.2 * (warped - fixed).square().mean()).item())


def self_test(device):
    """Exercise affine, nonlinear Jacobian, and registration invariants."""
    shape = (32, 32, 32)
    axes = torch.meshgrid(
        *[torch.arange(n, device=device) for n in shape], indexing="ij"
    )
    coords = torch.stack(axes).float()
    target = torch.exp(
        -(
            (
                coords
                - torch.tensor([15, 16, 17], device=device)[:, None, None, None]
            )
            .square()
            .sum(0)
            / 32
        )
    )[None, None]
    source = torch.exp(
        -(
            (
                coords
                - torch.tensor([18, 16, 17], device=device)[:, None, None, None]
            )
            .square()
            .sum(0)
            / 32
        )
    )[None, None]
    identity = torch.eye(4, device=device)
    zeros = torch.zeros((1, 3, *shape), device=device)
    j0 = pull_jacobian(identity[:3, :3], zeros, identity[:3, :3])
    scaled = identity[:3, :3].clone()
    scaled[0, 0] = 1.2
    js = pull_jacobian(scaled, zeros, identity[:3, :3])
    assert torch.allclose(j0, torch.ones_like(j0), atol=1e-6)
    assert torch.allclose(js, torch.full_like(js, 1.2), atol=1e-6)
    strong_field = zeros.clone()
    strong_field[:, 0] = 24 * torch.sin(
        coords[0] * (2 * math.pi / (shape[0] - 1))
    )
    safe_field, safe_jac, constraint = constrain_deformation(
        identity[:3, :3], strong_field, identity[:3, :3]
    )
    assert constraint["raw_jacobian_min"] < 0.2
    assert constraint["raw_jacobian_max"] > 5
    assert 0.05 <= constraint["deformation_scale"] < 1
    assert float(safe_jac.min()) >= 0.2 and float(safe_jac.max()) <= 5
    assert torch.allclose(
        safe_field, strong_field * constraint["deformation_scale"]
    )
    try:
        constrain_deformation(
            identity[:3, :3], strong_field, identity[:3, :3], min_scale=0.8
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("an inadmissible minimum scale must fail")
    scaled_affine = identity.clone()
    scaled_affine[0, 0] = 1.2
    _, jac_affine_only, _, _, _ = register(
        source,
        target,
        identity,
        identity,
        initial_pull=scaled_affine,
        scales=(1,),
        affine_steps=0,
        deform_steps=0,
    )
    assert torch.allclose(
        jac_affine_only, torch.ones_like(jac_affine_only), atol=1e-6
    )
    warped, jac, _, affine, _ = register(
        source,
        target,
        identity,
        identity,
        scales=(2, 1),
        affine_steps=12,
        deform_steps=6,
        smoothness=0.5,
    )
    before = _image_loss_for_test(source, target)
    after = _image_loss_for_test(warped[None, None], target)
    assert after < before * 0.2, (before, after)
    assert abs(affine[0, 3].item() - 3) < 0.75, affine[0, 3].item()
    assert torch.isfinite(jac).all() and (jac > 0).all()
    print(
        json.dumps(
            {
                "translation_mm": float(affine[0, 3]),
                "loss_before": before,
                "loss_after": after,
                "jacobian_identity": float(j0.mean()),
                "full_jacobian_scale_1p2": float(js.mean()),
                "modulation_jacobian_affine_only": float(jac_affine_only.mean()),
                "raw_fold_min": constraint["raw_jacobian_min"],
                "raw_expansion_max": constraint["raw_jacobian_max"],
                "accepted_deformation_scale": constraint["deformation_scale"],
                "constrained_jacobian_min": float(safe_jac.min()),
                "constrained_jacobian_max": float(safe_jac.max()),
                "device": str(device),
            },
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--moving", help="moving GM probability/PVE NIfTI")
    parser.add_argument("--fixed", help="GM template")
    parser.add_argument("--output-prefix", help="prefix for three NIfTI files and JSON")
    parser.add_argument("--init-flirt", help="optional moving-to-fixed FSL FLIRT .mat")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int)
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
    moving = torch.from_numpy(
        moving_image.get_fdata(dtype=np.float32).copy()
    )[None, None].to(device)
    fixed = torch.from_numpy(
        fixed_image.get_fdata(dtype=np.float32).copy()
    )[None, None].to(device)
    if not torch.isfinite(moving).all() or not torch.isfinite(fixed).all():
        parser.error("inputs contain NaN or infinity")
    moving_affine = torch.as_tensor(
        moving_image.affine, dtype=torch.float32, device=device
    )
    fixed_affine = torch.as_tensor(
        fixed_image.affine, dtype=torch.float32, device=device
    )
    initial = None
    if args.init_flirt is not None:
        initial = torch.as_tensor(
            flirt_pull_world(args.init_flirt, moving_image, fixed_image),
            dtype=torch.float32,
            device=device,
        )
    deformation_qc = {}
    warped, jacobian, field, affine, score = register(
        moving,
        fixed,
        moving_affine,
        fixed_affine,
        initial_pull=initial,
        affine_steps=args.affine_steps,
        deform_steps=args.deform_steps,
        smoothness=args.smoothness,
        qc=deformation_qc,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    paths = {
        name: str(prefix) + suffix
        for name, suffix in {
            "warped_gm": "_warped_gm.nii.gz",
            "jacobian": "_jacobian.nii.gz",
            "modulated_gm": "_modulated_gm.nii.gz",
        }.items()
    }
    for name, volume in (
        ("warped_gm", warped),
        ("jacobian", jacobian),
        ("modulated_gm", warped * jacobian),
    ):
        _save(volume, fixed_image, paths[name])
    report = {
        "method": (
            "PyTorch multiscale global normalized correlation + 0.2 MSE, "
            "sparse displacement control grid; not FNIRT"
        ),
        "device": str(device),
        "seconds_including_io": time.perf_counter() - started,
        "moving": str(Path(args.moving).resolve()),
        "fixed": str(Path(args.fixed).resolve()),
        "init_flirt": args.init_flirt,
        "settings": {
            "affine_steps_per_scale": args.affine_steps,
            "deform_steps_per_scale": args.deform_steps,
            "smoothness": args.smoothness,
            "scales": [4, 2, 1],
            "torch_threads": torch.get_num_threads(),
        },
        "fit_score_1_minus_loss": score,
        "pull_world_affine": affine.cpu().numpy().tolist(),
        "affine_jacobian_determinant": float(torch.linalg.det(affine[:3, :3])),
        "jacobian_convention": (
            "nonlinear-only: full pull determinant divided by affine determinant"
        ),
        "maximum_displacement_mm": float(
            torch.linalg.vector_norm(field, dim=0).max()
        ),
        "jacobian_min": float(jacobian.min()),
        "jacobian_max": float(jacobian.max()),
        "nonpositive_jacobian_voxels": int((jacobian <= 0).sum()),
        **deformation_qc,
        "outputs": paths,
    }
    report_path = str(prefix) + "_report.json"
    Path(report_path).write_text(json.dumps(report, indent=2) + "\n")
    print(report_path)


if __name__ == "__main__":
    main()
