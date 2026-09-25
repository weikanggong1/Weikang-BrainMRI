"""Compare native curvature with several 5x5 inversion methods on outliers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import linalg

from probe_place_surface_curvature_numeric import STATE, neighbors


def multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    result = np.zeros((a.shape[0], b.shape[1]), dtype=np.float32)
    for row in range(a.shape[0]):
        for col in range(b.shape[1]):
            for item in range(a.shape[1]):
                result[row, col] = np.float32(result[row, col] + np.float32(a[row, item] * b[item, col]))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--vertices", type=int, nargs="+", required=True)
    args = parser.parse_args()
    state = np.fromfile(args.probe / "lh.gradient.normal_spring", dtype=STATE)
    xyz = state["floats"][:, :3]
    normal = state["floats"][:, 3:6]
    basis = np.fromfile(args.probe / "lh.curvature.tangent_basis", dtype="<f4").reshape(-1, 6)
    truth = np.fromfile(args.probe / "lh.curvature.scalar", dtype="<f4")
    offsets, flat = neighbors(args.probe / "lh.gradient.neighbors_total", len(state))
    results = []
    for vertex in args.vertices:
        group = flat[offsets[vertex]:offsets[vertex + 1]]
        design = np.zeros((len(group), 5), dtype=np.float32)
        height = np.zeros((len(group), 1), dtype=np.float32)
        for row, other in enumerate(group):
            edge = np.float32(xyz[other] - xyz[vertex])
            height[row, 0] = np.float32(np.float32(edge[0]*normal[vertex, 0] + edge[1]*normal[vertex, 1]) + edge[2]*normal[vertex, 2])
            u = np.float32(np.float32(edge[0]*basis[vertex, 0] + edge[1]*basis[vertex, 1]) + edge[2]*basis[vertex, 2])
            v = np.float32(np.float32(edge[0]*basis[vertex, 3] + edge[1]*basis[vertex, 4]) + edge[2]*basis[vertex, 5])
            design[row] = (np.float32(u*u), np.float32(v*v), u, v, np.float32(1))
        gram = multiply(design.T.copy(), design)
        raw = (args.probe / f"lh.curvature.matrix.{vertex}").read_bytes()
        native_count = int(np.frombuffer(raw, dtype="<i4", count=1)[0])
        if native_count != len(group):
            raise ValueError("native design row mismatch")
        values = np.frombuffer(raw, dtype="<f4", offset=4)
        cursor = 0
        native_design = values[cursor:cursor + native_count*5].reshape(native_count, 5)
        cursor += native_count*5
        native_height = values[cursor:cursor + native_count]
        cursor += native_count
        native_gram = values[cursor:cursor + 25].reshape(5, 5)
        cursor += 25
        native_inverse = values[cursor:cursor + 25].reshape(5, 5)
        cursor += 25
        native_pseudo = values[cursor:cursor + 5*native_count].reshape(5, native_count)
        cursor += 5*native_count
        native_parameter = values[cursor:cursor + 5]
        q, r = linalg.qr(gram, mode="economic")
        methods = {}
        for name, inverse in (
            ("numpy_inv32", np.linalg.inv(gram)),
            ("scipy_inv32", linalg.inv(gram)),
            ("qr32", linalg.solve_triangular(r, q.T)),
            ("numpy_inv64", np.linalg.inv(gram.astype(np.float64)).astype(np.float32)),
            ("native_inverse", native_inverse),
        ):
            pseudo = multiply(inverse.astype(np.float32), design.T.copy())
            value = multiply(pseudo[4:5], height)[0, 0]
            methods[name] = {"value": float(value), "error": float(abs(float(value) - float(truth[vertex])))}
        results.append({
            "vertex": vertex, "source": float(truth[vertex]), "condition": float(np.linalg.cond(gram)),
            "input_exact": {
                "design": int(np.count_nonzero(design == native_design)),
                "design_total": int(design.size),
                "height": int(np.count_nonzero(height[:, 0] == native_height)),
                "height_total": int(native_height.size),
                "gram": int(np.count_nonzero(gram == native_gram)),
                "gram_total": 25,
            },
            "inverse_max_abs": float(np.max(np.abs(np.linalg.inv(gram).astype(np.float64) - native_inverse.astype(np.float64)))),
            "native_pseudo_recomputed_exact": int(np.count_nonzero(multiply(native_inverse, native_design.T.copy()) == native_pseudo)),
            "native_parameter": native_parameter.tolist(),
            "methods": methods,
        })
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
