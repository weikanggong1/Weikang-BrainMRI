"""Check selected native spherical-blur variance pixels before vectorization."""

from __future__ import annotations

import argparse
import json
import math

import nibabel as nib
import numpy as np
import tifffile


def blur_pixel(frame: np.ndarray, sigma: float, u: int, v: int) -> float:
    v_dim, u_dim = frame.shape
    cart_klen = round(6 * sigma) + 1
    if cart_klen % 2 == 0:
        cart_klen += 1
    phi = u * math.pi / u_dim
    sin_sq = math.sin(phi) ** 2
    if sin_sq < np.finfo(np.float32).eps:
        klen = 4 * cart_klen
    else:
        k = cart_klen * cart_klen
        klen = int(math.sqrt(k + k / sin_sq))
        klen = min(klen, 4 * cart_klen)
    klen = min(klen, u_dim - 1, v_dim - 1)
    khalf = klen // 2
    sigma_sq_inv = np.float32(1.0 / np.float32(sigma * sigma))
    total = 0.0
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
        for vk in range(-khalf, khalf + 1):
            weight = math.exp(-((uk * uk) + sin_sq * (vk * vk)) * float(sigma_sq_inv))
            v1 = (v + vk + voff) % v_dim
            ktotal += weight
            total += weight * float(frame[v1, u1])
    return float(np.float32(total / ktotal))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("atlas")
    p.add_argument("native_mgz")
    p.add_argument("--frame", type=int, default=4)
    p.add_argument("--sigma", type=float, default=0.5)
    args = p.parse_args()
    atlas = tifffile.imread(args.atlas)[args.frame].view(np.float32)
    native = np.asanyarray(nib.load(args.native_mgz).dataobj)[:, :, args.frame]
    points = [(0, 0), (0, 137), (1, 31), (16, 200), (64, 100),
              (128, 0), (128, 300), (192, 420), (254, 501), (255, 77)]
    result = []
    for u, v in points:
        predicted = blur_pixel(atlas, args.sigma, u, v)
        reference = float(native[u, v])
        result.append({"u": u, "v": v, "predicted": predicted,
                       "native": reference, "exact": predicted == reference,
                       "error": predicted - reference})
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
