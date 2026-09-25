"""Rigid-search curvature objective for spherical registration."""

from __future__ import annotations

import math
from functools import lru_cache

import torch


@lru_cache(maxsize=2)
def _atan_table(device_name: str) -> torch.Tensor:
    return torch.tensor([math.atan2(i, 100000) for i in range(100001)],
                        dtype=torch.float32, device=device_name)


def _fast_atan2(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    ax, ay = x.abs(), y.abs()
    major_x = ax >= ay
    lo = torch.where(major_x, ay, ax)
    hi = torch.where(major_x, ax, ay)
    index = torch.where(hi == 0, 0, (100000 * lo / hi).int()).clamp(0, 100000)
    table = _atan_table(str(x.device))
    sample = table[index].double()
    half_pi = float(torch.tensor(math.pi / 2, dtype=torch.float32))
    pi = float(torch.tensor(math.pi, dtype=torch.float32))
    value = torch.where(major_x, sample, half_pi - sample)
    value = torch.where((x >= 0) & (y < 0), -value, value)
    value = torch.where((x < 0) & (y >= 0), pi - value, value)
    value = torch.where((x < 0) & (y < 0), -pi + value, value)
    return value.float()


def _sample_mean_variance(mean: torch.Tensor, variance: torch.Tensor,
                          x: torch.Tensor, y: torch.Tensor, z: torch.Tensor,
                          alpha: float, radius: float) -> tuple[torch.Tensor, torch.Tensor]:
    v_dim, u_dim = mean.shape
    rad = torch.tensor(radius, dtype=torch.float32, device=x.device)
    d = (rad * rad - z * z).clamp_min(0)
    phi = torch.atan2(torch.sqrt(d), z)
    uf = ((u_dim * phi).double() / math.pi).float()
    u0, u1 = torch.floor(uf).long(), torch.ceil(uf).long()
    du = uf - u0.float()
    u0_voff = torch.where((u0 < 0) | (u0 >= u_dim), v_dim // 2, 0)
    u1_voff = torch.where((u1 < 0) | (u1 >= u_dim), v_dim // 2, 0)
    u0 = torch.where(u0 < 0, -u0, u0)
    u1 = torch.where(u1 < 0, -u1, u1)
    u0 = torch.where(u0 >= u_dim, u_dim - (u0 - u_dim + 1), u0)
    u1 = torch.where(u1 >= u_dim, u_dim - (u1 - u_dim + 1), u1)

    alpha32 = torch.tensor(alpha, dtype=torch.float32, device=x.device)
    theta = _fast_atan2(y, x) - alpha32
    theta = torch.where(theta < 0, (theta.double() + 2 * math.pi).float(), theta)
    theta = torch.where(theta >= 2 * math.pi,
                        (theta.double() - 2 * math.pi).float(), theta)
    vf = ((v_dim * theta).double() / (2 * math.pi)).float()
    v0, v1 = torch.floor(vf).long() % v_dim, torch.ceil(vf).long() % v_dim
    dv = vf - torch.floor(vf)

    def sample(frame: torch.Tensor) -> torch.Tensor:
        return (du * dv * frame[(v1 + u1_voff) % v_dim, u1]
                + (1 - du) * dv * frame[(v1 + u0_voff) % v_dim, u0]
                + (1 - du) * (1 - dv) * frame[(v0 + u0_voff) % v_dim, u0]
                + du * (1 - dv) * frame[(v0 + u1_voff) % v_dim, u1])

    return sample(mean), sample(variance)


def rigid_sse(vertices: torch.Tensor, curvature: torch.Tensor,
              target_mean: torch.Tensor, target_variance: torch.Tensor,
              angles_radians: tuple[float, float, float],
              radius: float = 100.0) -> float:
    """Compute the 16-partition native absolute standardized SSE at one angle."""
    xyz = vertices.float()
    x, y, z = xyz.unbind(dim=1)
    radius32 = torch.tensor(radius, dtype=torch.float32, device=xyz.device)
    length = torch.sqrt(x * x + y * y + z * z)
    scale = radius32 / length
    x, y, z = x * scale, y * scale, z * scale

    alpha, beta, gamma = angles_radians
    sg = torch.tensor(math.sin(float(torch.tensor(gamma, dtype=torch.float32))),
                      dtype=torch.float32, device=xyz.device)
    cg = torch.tensor(math.cos(float(torch.tensor(gamma, dtype=torch.float32))),
                      dtype=torch.float32, device=xyz.device)
    sb = torch.tensor(math.sin(float(torch.tensor(beta, dtype=torch.float32))),
                      dtype=torch.float32, device=xyz.device)
    cb = torch.tensor(math.cos(float(torch.tensor(beta, dtype=torch.float32))),
                      dtype=torch.float32, device=xyz.device)
    gamma_y = y * cg - z * sg
    gamma_z = y * sg + z * cg
    rotated_x = x * cb - gamma_z * sb
    rotated_z = x * sb + gamma_z * cb
    target, variance = _sample_mean_variance(
        target_mean, target_variance, rotated_x, gamma_y, rotated_z, alpha, radius)
    std = torch.sqrt(variance.double())
    std = torch.where(std.abs() < torch.finfo(torch.float32).eps, 4.0, std)
    delta = (curvature.double() - target.double()) / std
    abs_delta = delta.abs()
    partition_size = (len(vertices) + 15) // 16
    return sum(float(abs_delta[p * partition_size:(p + 1) * partition_size].sum())
               for p in range(16))


def rigid_search(vertices: torch.Tensor, curvature: torch.Tensor,
                 target_mean: torch.Tensor, target_variance: torch.Tensor,
                 *, max_degrees: float = 64.0, min_degrees: float = 0.5,
                 nangles: int = 8) -> tuple[tuple[float, float, float], float, int]:
    """Search the native one-center angle grid; return angles, SSE, evaluations."""
    max_radians = float(torch.tensor(math.radians(max_degrees), dtype=torch.float32))
    min_radians = float(torch.tensor(math.radians(min_degrees), dtype=torch.float32))
    grid_size = 1
    while max_radians / grid_size > min_radians:
        grid_size *= 2
    center = grid_size // 2
    radians_per_cell = float(torch.tensor(max_radians / grid_size, dtype=torch.float32))

    def angle(index: int) -> float:
        return float(torch.tensor((index - center) * radians_per_cell,
                                  dtype=torch.float32))

    stride = (grid_size + nangles - 1) // nangles - 1
    best_indices = (center, center, center)
    best_sse = -1.0
    done: set[tuple[int, int, int]] = set()
    changed = True
    with torch.no_grad():
        while True:
            if not changed:
                if stride == 1:
                    break
                stride //= 2
            changed = False
            old_indices = best_indices
            for gj in range(nangles + 1):
                gi = old_indices[2] + stride * (gj - nangles // 2)
                if not 0 <= gi < grid_size:
                    continue
                for bj in range(nangles + 1):
                    bi = old_indices[1] + stride * (bj - nangles // 2)
                    if not 0 <= bi < grid_size:
                        continue
                    for aj in range(nangles + 1):
                        ai = old_indices[0] + stride * (aj - nangles // 2)
                        if not 0 <= ai < grid_size:
                            continue
                        candidate = (ai, bi, gi)
                        if candidate in done:
                            continue
                        done.add(candidate)
                        sse = rigid_sse(vertices, curvature, target_mean,
                                        target_variance,
                                        (angle(ai), angle(bi), angle(gi)))
                        radius = (stride * nangles) / 3.0
                        nearby = all(abs(old - new) <= radius
                                     for old, new in zip(best_indices, candidate))
                        if (best_sse < 0 or
                                (nearby and best_sse >= sse) or
                                (not nearby and best_sse > sse)):
                            best_indices = candidate
                            best_sse = sse
                            changed = True
    return tuple(angle(i) for i in best_indices), best_sse, len(done)
