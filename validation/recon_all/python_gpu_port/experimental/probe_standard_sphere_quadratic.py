"""Inspect FreeSurfer's three-point float32 quadratic-fit arithmetic."""

from __future__ import annotations

import argparse
import json

import numpy as np


def mm(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.zeros((a.shape[0], b.shape[1]), np.float32)
    for row in range(a.shape[0]):
        for col in range(b.shape[1]):
            value = np.float32(0)
            for k in range(a.shape[1]):
                value = np.float32(value + np.float32(a[row, k] * b[k, col]))
            out[row, col] = value
    return out


def vnl_inverse(m: np.ndarray) -> np.ndarray:
    a, b, c = m[0]
    d, e, f = m[1]
    g, h, i = m[2]
    determinant = a * e * i - a * h * f - d * b * i + d * h * c + g * b * f - g * e * c
    inverse_det = np.float32(1) / determinant
    inverse = np.asarray([
        [e*i - f*h, h*c - i*b, b*f - c*e],
        [f*g - d*i, a*i - c*g, d*c - f*a],
        [d*h - e*g, b*g - a*h, a*e - b*d]], np.float32)
    return np.float32(inverse * inverse_det)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report")
    args = parser.parse_args()
    data = json.load(open(args.report))["search"]["candidates"][:3]
    x = np.asarray([[entry["dt"] ** 2, 2 * entry["dt"], 1] for entry in data], np.float32)
    y = np.asarray([entry["sse"] for entry in data], np.float32).reshape(3, 1)
    xtx = mm(x.T, x)
    xty = mm(x.T, y)
    print("x", x.tolist())
    print("y", y.ravel().tolist())
    print("xtx", xtx.tolist())
    print("xty", xty.ravel().tolist())
    for name, inverse in (("vnl_explicit", vnl_inverse(xtx)),
                          ("numpy_float32", np.linalg.inv(xtx)),
                          ("numpy_float64_cast", np.linalg.inv(xtx.astype(np.float64)).astype(np.float32))):
        p = mm(inverse, xty).ravel()
        print(name, p.tolist(), float(np.float32(-p[1] / p[0])))
    p64 = np.linalg.solve(x.astype(np.float64), y.astype(np.float64)).ravel()
    print("direct_float64_on_float32_data", p64.tolist(), float(-p64[1] / p64[0]))


if __name__ == "__main__":
    main()
