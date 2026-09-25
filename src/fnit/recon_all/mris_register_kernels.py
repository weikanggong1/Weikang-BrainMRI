"""Source-order tensor kernels used at the start of ``mris_register``.

These functions do not implement atlas matching or surface deformation.
"""

from __future__ import annotations

import math

import torch


def center_sphere(vertices: torch.Tensor) -> torch.Tensor:
    """Move the float32 bounding-box center to zero as ``MRIScenter`` does."""
    xyz = vertices.to(torch.float32)
    center = ((xyz.amin(dim=0).to(torch.float64)
               + xyz.amax(dim=0).to(torch.float64)) * 0.5).to(torch.float32)
    return xyz - center


def project_sphere(vertices: torch.Tensor, radius: float = 100.0) -> torch.Tensor:
    """One ``MRISprojectOntoSphereWkr`` radial update for a centered sphere."""
    xyz = vertices.to(torch.float64)
    distance = torch.sqrt(xyz[:, 0] * xyz[:, 0]
                          + xyz[:, 1] * xyz[:, 1]
                          + xyz[:, 2] * xyz[:, 2])
    scale = torch.where(distance < torch.finfo(torch.float32).eps,
                        torch.zeros_like(distance), 1.0 - radius / distance)
    displacement = scale.unsqueeze(1) * xyz
    return (xyz - displacement).to(torch.float32)


def rigid_grid_angle(half_degree_index: int) -> float:
    """An angle on FreeSurfer's default 64-degree, 128-cell search grid."""
    cell = torch.tensor(math.radians(64.0), dtype=torch.float32) / 128
    return float(cell * half_degree_index)


def rotate_sphere(vertices: torch.Tensor,
                  angles_radians: tuple[float, float, float]) -> torch.Tensor:
    """Apply ``MRISrotate`` Z-Y-X coefficients in source float32 order."""
    alpha, beta, gamma = (torch.tensor(a, dtype=torch.float32)
                          for a in angles_radians)
    sa, sb, sg = (torch.tensor(math.sin(float(a)), dtype=torch.float32)
                  for a in (alpha, beta, gamma))
    ca, cb, cg = (torch.tensor(math.cos(float(a)), dtype=torch.float32)
                  for a in (alpha, beta, gamma))

    cacb = ca * cb
    cacgsb = ca * cg * sb
    sasg = sa * sg
    cgsa = cg * sa
    casbsg = ca * sb * sg
    cbsa = cb * sa
    cgsasb = cg * sa * sb
    casg = ca * sg
    cacg = ca * cg
    sasbsg = sa * sb * sg
    cbcg = cb * cg
    cbsg = cb * sg

    x, y, z = vertices.to(torch.float32).unbind(dim=1)
    coefficients = [v.to(vertices.device) for v in
                    (cacb, cacgsb, sasg, cgsa, casbsg, cbsa,
                     cgsasb, casg, cacg, sasbsg, cbcg, cbsg, sb)]
    (cacb, cacgsb, sasg, cgsa, casbsg, cbsa,
     cgsasb, casg, cacg, sasbsg, cbcg, cbsg, sb) = coefficients
    xp = x * cacb + z * (-cacgsb - sasg) + y * (cgsa - casbsg)
    yp = -x * cbsa + z * (cgsasb - casg) + y * (cacg + sasbsg)
    zp = z * cbcg + x * sb + y * cbsg
    return torch.stack((xp, yp, zp), dim=1)


def normalize_mean_curvature(values: torch.Tensor,
                             ripped: torch.Tensor | None = None) -> torch.Tensor:
    """FreeSurfer ``NORM_MEAN`` with population variance and float32 output."""
    curv = values.to(torch.float32)
    active = curv if ripped is None else curv[~ripped]
    active64 = active.to(torch.float64)
    mean = active64.sum() / active64.numel()
    delta = active64 - mean
    std = torch.sqrt((delta * delta).sum() / active64.numel())
    result = curv.clone()
    if ripped is None:
        result = ((curv.to(torch.float64) - mean) / std).to(torch.float32)
    else:
        result[~ripped] = (delta / std).to(torch.float32)
    return result
