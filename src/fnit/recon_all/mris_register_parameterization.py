"""Surface curvature rasterization for spherical registration."""

from __future__ import annotations

import math

import torch


def parameterize_curvature(vertices: torch.Tensor, curvature: torch.Tensor,
                           u_dim: int = 256, v_dim: int = 512) -> torch.Tensor:
    """Map vertex values to a (azimuth, polar) grid as ``MRIStoParameterization``."""
    xyz = vertices.float()
    center = (xyz.double().amin(dim=0) + xyz.double().amax(dim=0)) / 2
    radius = torch.sqrt(((xyz.double() - center) ** 2).sum(dim=1)).mean().float()
    x, y, z = xyz.unbind(dim=1)
    theta = torch.atan2((y / radius).double(), (x / radius).double()).float()
    theta = torch.where(theta < 0, (2 * math.pi + theta.double()).float(), theta)
    d = (radius * radius - z * z).clamp_min(0)
    phi = torch.atan2(torch.sqrt(d.double()), z.double()).float()
    uf = ((u_dim * phi).double() / math.pi).float()
    vf = ((v_dim * theta).double() / (2 * math.pi)).float()
    u = torch.floor(uf + 0.5).long()
    v = torch.floor(vf + 0.5).long()
    u = torch.where(u < 0, -u, u)
    u = torch.where(u >= u_dim, u_dim - (u - u_dim + 1), u)
    v = v % v_dim

    flat = u * v_dim + v
    counts = torch.bincount(flat, minlength=u_dim * v_dim)
    values = curvature.float() / counts[flat].float()
    grid_flat = torch.zeros(u_dim * v_dim, dtype=torch.float32, device=vertices.device)
    grid_flat.index_add_(0, flat, values)
    grid = grid_flat.view(u_dim, v_dim)
    filled = counts.view(u_dim, v_dim) > 0
    u_axis = torch.arange(u_dim, device=vertices.device)
    v_axis = torch.arange(v_dim, device=vertices.device)
    while not bool(filled.all()):
        total = torch.zeros_like(grid)
        neighbors = torch.zeros_like(grid, dtype=torch.int32)
        for uk in (-1, 0, 1):
            u1 = u_axis + uk
            u1 = torch.where(u1 < 0, -u1, u1)
            u1 = torch.where(u1 >= u_dim, u_dim - (u1 - u_dim + 1), u1)
            for vk in (-1, 0, 1):
                v1 = (v_axis + vk) % v_dim
                available = filled[u1[:, None], v1[None, :]]
                sample = grid[u1[:, None], v1[None, :]]
                total += torch.where(available, sample, 0)
                neighbors += available.int()
        add = ~filled & (neighbors > 0)
        grid = torch.where(add, total / neighbors.clamp_min(1), grid)
        filled |= add
    return grid.T.contiguous()
