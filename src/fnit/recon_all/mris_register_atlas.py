"""Source-order atlas sampling for ``mris_register`` spherical surfaces."""

from __future__ import annotations

import math

import torch


def sample_atlas_on_canonical_sphere(vertices: torch.Tensor,
                                     atlas_frame: torch.Tensor) -> torch.Tensor:
    """Match ``MRISfromParameterization`` on a centered canonical sphere.

    ``atlas_frame`` has shape (azimuthal bins, polar bins) and float32 values.
    The fixed FreeSurfer atlas TIFF stores its float32 bits as int32, so the
    caller must reinterpret rather than numerically convert those bits.
    """
    xyz = vertices.float()
    center = (xyz.double().amin(dim=0) + xyz.double().amax(dim=0)) / 2
    radius = torch.sqrt(((xyz.double() - center) ** 2).sum(dim=1)).mean().float()
    x, y, z = xyz.unbind(dim=1)
    theta = torch.atan2((y / radius).double(), (x / radius).double()).float()
    theta = torch.where(theta < 0, (2 * math.pi + theta.double()).float(), theta)
    d = (radius * radius - z * z).clamp_min(0)
    phi = torch.atan2(torch.sqrt(d.double()), z.double()).float()
    u_dim, v_dim = atlas_frame.shape[1], atlas_frame.shape[0]
    uf = ((u_dim * phi).double() / math.pi).float()
    vf = ((v_dim * theta).double() / (2 * math.pi)).float()
    u0, u1 = torch.floor(uf).int(), torch.ceil(uf).int()
    v0, v1 = torch.floor(vf).int(), torch.ceil(vf).int()
    du, dv = uf - u0, vf - v0
    u0 = torch.where(u0 < 0, -u0, u0)
    u1 = torch.where(u1 < 0, -u1, u1)
    u0 = torch.where(u0 >= u_dim, u_dim - (u0 - u_dim + 1), u0)
    u1 = torch.where(u1 >= u_dim, u_dim - (u1 - u_dim + 1), u1)
    v0, v1 = v0 % v_dim, v1 % v_dim
    return (du * dv * atlas_frame[v1.long(), u1.long()]
            + (1 - du) * dv * atlas_frame[v1.long(), u0.long()]
            + (1 - du) * (1 - dv) * atlas_frame[v0.long(), u0.long()]
            + du * (1 - dv) * atlas_frame[v0.long(), u1.long()]).float()
