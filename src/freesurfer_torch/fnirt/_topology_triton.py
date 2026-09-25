"""Serial-order CUDA kernel for FSL's topology gradient limiter."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _det3(a00, a01, a02, a10, a11, a12, a20, a21, a22):
    return (
        a00 * (a11 * a22 - a12 * a21)
        - a01 * (a10 * a22 - a12 * a20)
        + a02 * (a10 * a21 - a11 * a20)
    )


@triton.jit
def _limit_gradient_serial_kernel(
    gradient,
    jacobians,
    nx: tl.constexpr,
    ny: tl.constexpr,
    nz: tl.constexpr,
    minimum: tl.constexpr,
    maximum: tl.constexpr,
):
    cube_x: tl.constexpr = nx - 1
    cube_y: tl.constexpr = ny - 1
    cube_z: tl.constexpr = nz - 1
    spatial: tl.constexpr = nx * ny * nz
    cubes: tl.constexpr = cube_x * cube_y * cube_z
    # One program deliberately preserves warpfns' z/y/x/corner update order.
    for linear in tl.range(0, cubes, num_stages=1):
        z = linear // (cube_x * cube_y)
        remainder = linear - z * cube_x * cube_y
        y = remainder // cube_x
        x = remainder - y * cube_x
        cube_index = x * cube_y * cube_z + y * cube_z + z
        for corner in tl.static_range(8):
            jacobian = tl.load(jacobians + corner * cubes + cube_index)
            invalid = (jacobian < minimum) | (jacobian > maximum)
            if invalid:
                if corner == 0:
                    x0: tl.constexpr = 0; y0: tl.constexpr = 0; z0: tl.constexpr = 0
                    x1: tl.constexpr = 0; y1: tl.constexpr = 0; z1: tl.constexpr = 0
                    x2: tl.constexpr = 0; y2: tl.constexpr = 0; z2: tl.constexpr = 0
                elif corner == 1:
                    x0: tl.constexpr = 0; y0: tl.constexpr = 0; z0: tl.constexpr = 0
                    x1: tl.constexpr = 1; y1: tl.constexpr = 0; z1: tl.constexpr = 0
                    x2: tl.constexpr = 1; y2: tl.constexpr = 0; z2: tl.constexpr = 0
                elif corner == 2:
                    x0: tl.constexpr = 0; y0: tl.constexpr = 1; z0: tl.constexpr = 0
                    x1: tl.constexpr = 0; y1: tl.constexpr = 0; z1: tl.constexpr = 0
                    x2: tl.constexpr = 0; y2: tl.constexpr = 1; z2: tl.constexpr = 0
                elif corner == 3:
                    x0: tl.constexpr = 0; y0: tl.constexpr = 0; z0: tl.constexpr = 1
                    x1: tl.constexpr = 0; y1: tl.constexpr = 0; z1: tl.constexpr = 1
                    x2: tl.constexpr = 0; y2: tl.constexpr = 0; z2: tl.constexpr = 0
                elif corner == 4:
                    x0: tl.constexpr = 0; y0: tl.constexpr = 1; z0: tl.constexpr = 1
                    x1: tl.constexpr = 0; y1: tl.constexpr = 0; z1: tl.constexpr = 1
                    x2: tl.constexpr = 0; y2: tl.constexpr = 1; z2: tl.constexpr = 0
                elif corner == 5:
                    x0: tl.constexpr = 0; y0: tl.constexpr = 0; z0: tl.constexpr = 1
                    x1: tl.constexpr = 1; y1: tl.constexpr = 0; z1: tl.constexpr = 1
                    x2: tl.constexpr = 1; y2: tl.constexpr = 0; z2: tl.constexpr = 0
                elif corner == 6:
                    x0: tl.constexpr = 0; y0: tl.constexpr = 1; z0: tl.constexpr = 0
                    x1: tl.constexpr = 1; y1: tl.constexpr = 0; z1: tl.constexpr = 0
                    x2: tl.constexpr = 1; y2: tl.constexpr = 1; z2: tl.constexpr = 0
                else:
                    x0: tl.constexpr = 0; y0: tl.constexpr = 1; z0: tl.constexpr = 1
                    x1: tl.constexpr = 1; y1: tl.constexpr = 0; z1: tl.constexpr = 1
                    x2: tl.constexpr = 1; y2: tl.constexpr = 1; z2: tl.constexpr = 0

                p0 = (x + x0) * ny * nz + (y + y0) * nz + z + z0
                p1 = (x + x1) * ny * nz + (y + y1) * nz + z + z1
                p2 = (x + x2) * ny * nz + (y + y2) * nz + z + z2
                a00 = tl.load(gradient + 0 * spatial + p0).to(tl.float64)
                a10 = tl.load(gradient + 3 * spatial + p0).to(tl.float64)
                a20 = tl.load(gradient + 6 * spatial + p0).to(tl.float64)
                a01 = tl.load(gradient + 1 * spatial + p1).to(tl.float64)
                a11 = tl.load(gradient + 4 * spatial + p1).to(tl.float64)
                a21 = tl.load(gradient + 7 * spatial + p1).to(tl.float64)
                a02 = tl.load(gradient + 2 * spatial + p2).to(tl.float64)
                a12 = tl.load(gradient + 5 * spatial + p2).to(tl.float64)
                a22 = tl.load(gradient + 8 * spatial + p2).to(tl.float64)
                alpha = tl.zeros((), tl.float32)
                determinant = _det3(
                    a00, a01, a02, a10, a11, a12, a20, a21, a22
                )
                for _ in tl.static_range(10):
                    outside = (determinant < minimum) | (determinant > maximum)
                    next_alpha = tl.minimum(alpha + 0.1, 1.0)
                    alpha = tl.where(outside, next_alpha, alpha)
                    one_minus = (1.0 - alpha).to(tl.float64)
                    alpha64 = alpha.to(tl.float64)
                    candidate = _det3(
                        one_minus * a00 + alpha64,
                        one_minus * a01,
                        one_minus * a02,
                        one_minus * a10,
                        one_minus * a11 + alpha64,
                        one_minus * a12,
                        one_minus * a20,
                        one_minus * a21,
                        one_minus * a22 + alpha64,
                    )
                    determinant = tl.where(outside, candidate, determinant)
                alpha = tl.minimum(alpha + 0.1, 1.0)
                one_minus = (1.0 - alpha).to(tl.float64)
                alpha64 = alpha.to(tl.float64)
                tl.store(gradient + 0 * spatial + p0, one_minus * a00 + alpha64)
                tl.store(gradient + 3 * spatial + p0, one_minus * a10)
                tl.store(gradient + 6 * spatial + p0, one_minus * a20)
                tl.store(gradient + 1 * spatial + p1, one_minus * a01)
                tl.store(gradient + 4 * spatial + p1, one_minus * a11 + alpha64)
                tl.store(gradient + 7 * spatial + p1, one_minus * a21)
                tl.store(gradient + 2 * spatial + p2, one_minus * a02)
                tl.store(gradient + 5 * spatial + p2, one_minus * a12)
                tl.store(gradient + 8 * spatial + p2, one_minus * a22 + alpha64)


def limit_gradient_serial_cuda(gradient, jacobians, minimum, maximum):
    if not gradient.is_cuda or not jacobians.is_cuda:
        raise ValueError("Triton topology limiter requires CUDA tensors")
    if gradient.dtype != torch.float32 or jacobians.dtype != torch.float32:
        raise ValueError("Triton topology limiter requires float32 tensors")
    gradient = gradient.contiguous()
    jacobians = jacobians.contiguous()
    nx, ny, nz = (int(value) for value in gradient.shape[1:])
    _limit_gradient_serial_kernel[(1,)](
        gradient,
        jacobians,
        nx=nx,
        ny=ny,
        nz=nz,
        minimum=float(minimum),
        maximum=float(maximum),
        num_warps=1,
    )
    return gradient


__all__ = ["limit_gradient_serial_cuda"]
