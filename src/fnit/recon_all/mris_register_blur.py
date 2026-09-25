"""Spherical parameterization blur used by ``mris_register``."""

from __future__ import annotations

import math

import torch


def blur_atlas_frame(frame: torch.Tensor, sigma: float) -> torch.Tensor:
    """Blur one float32 (azimuth, polar) frame in FreeSurfer source order."""
    v_dim, u_dim = frame.shape
    cart_klen = round(6 * sigma) + 1
    if cart_klen % 2 == 0:
        cart_klen += 1
    sigma_sq_inv = float((torch.tensor(1, dtype=torch.float32)
                          / torch.tensor(sigma, dtype=torch.float32).square()).item())
    output = torch.empty_like(frame)
    for u in range(u_dim):
        sin_sq = math.sin(u * math.pi / u_dim) ** 2
        if sin_sq < torch.finfo(torch.float32).eps:
            klen = 4 * cart_klen
        else:
            k = cart_klen * cart_klen
            klen = min(int(math.sqrt(k + k / sin_sq)), 4 * cart_klen)
        khalf = min(klen, u_dim - 1, v_dim - 1) // 2
        weights = {(uk, vk): math.exp(-(uk * uk + sin_sq * vk * vk) * sigma_sq_inv)
                   for uk in range(khalf + 1) for vk in range(khalf + 1)}
        total = torch.zeros(v_dim, dtype=torch.float64, device=frame.device)
        ktotal = 0.0
        for uk in range(-khalf, khalf + 1):
            u1 = u + uk
            voff = 0
            if u1 < 0:
                u1 = -u1
                voff = v_dim // 2
            elif u1 >= u_dim:
                u1 = u_dim - (u1 - u_dim + 1)
                voff = v_dim // 2
            column = frame[:, u1].to(torch.float64)
            for vk in range(-khalf, khalf + 1):
                weight = weights[abs(uk), abs(vk)]
                ktotal += weight
                total += weight * torch.roll(column, -(vk + voff))
        output[:, u] = (total / ktotal).float()
    return output
