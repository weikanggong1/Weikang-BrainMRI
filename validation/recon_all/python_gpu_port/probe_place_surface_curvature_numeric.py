"""Numerical experiment for first pial quadratic curvature, not a runtime path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


STATE = np.dtype([("floats", "<f4", 9), ("flags", "<i4", 3)])


def neighbors(path: Path, vertices: int) -> tuple[np.ndarray, np.ndarray]:
    data = path.read_bytes()
    offsets = np.zeros(vertices + 1, dtype=np.int32)
    flat: list[int] = []
    cursor = 0
    for v in range(vertices):
        count = int(np.frombuffer(data, dtype="<i2", count=1, offset=cursor)[0])
        cursor += 2
        flat.extend(np.frombuffer(data, dtype="<i4", count=count, offset=cursor))
        cursor += 4 * count
        offsets[v + 1] = len(flat)
    if cursor != len(data):
        raise ValueError("neighbor dump has trailing entries")
    return offsets, np.asarray(flat, dtype=np.int32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=1000)
    args = parser.parse_args()
    state = np.fromfile(args.probe / "lh.gradient.normal_spring", dtype=STATE)
    xyz = state["floats"][:, :3]
    normal = state["floats"][:, 3:6]
    ripped = state["flags"][:, 0]
    basis = np.fromfile(args.probe / "lh.curvature.tangent_basis", dtype="<f4").reshape(-1, 6)
    truth = np.fromfile(args.probe / "lh.curvature.scalar", dtype="<f4")
    offsets, flat = neighbors(args.probe / "lh.gradient.neighbors_total", len(state))
    candidate = np.zeros(args.limit, dtype=np.float32)
    failed = 0
    for vertex in range(min(args.limit, len(state))):
        if ripped[vertex]:
            continue
        group = flat[offsets[vertex]:offsets[vertex + 1]]
        x = np.zeros((len(group), 5), dtype=np.float32)
        y = np.zeros(len(group), dtype=np.float32)
        for row, other in enumerate(group):
            edge = np.float32(xyz[other] - xyz[vertex])
            y[row] = np.float32(np.float32(edge[0]*normal[vertex, 0] + edge[1]*normal[vertex, 1]) + edge[2]*normal[vertex, 2])
            u = np.float32(np.float32(edge[0]*basis[vertex, 0] + edge[1]*basis[vertex, 1]) + edge[2]*basis[vertex, 2])
            v = np.float32(np.float32(edge[0]*basis[vertex, 3] + edge[1]*basis[vertex, 4]) + edge[2]*basis[vertex, 5])
            x[row] = (np.float32(u*u), np.float32(v*v), u, v, np.float32(1))
        gram = np.zeros((5, 5), dtype=np.float32)
        for row in range(5):
            for col in range(5):
                for item in range(len(group)):
                    gram[row, col] = np.float32(gram[row, col] + np.float32(x[item, row] * x[item, col]))
        try:
            inverse = np.linalg.inv(gram)
        except np.linalg.LinAlgError:
            failed += 1
            continue
        pseudo = np.zeros((5, len(group)), dtype=np.float32)
        for row in range(5):
            for col in range(len(group)):
                for item in range(5):
                    pseudo[row, col] = np.float32(pseudo[row, col] + np.float32(inverse[row, item] * x[col, item]))
        for col in range(len(group)):
            candidate[vertex] = np.float32(candidate[vertex] + np.float32(pseudo[4, col] * y[col]))
    valid = ripped[:args.limit] == 0
    error = np.abs(candidate[:len(valid)][valid].astype(np.float64) - truth[:len(valid)][valid].astype(np.float64))
    print(json.dumps({
        "vertices": args.limit, "active": int(valid.sum()), "failed": failed,
        "exact": int(np.count_nonzero(candidate[:len(valid)][valid] == truth[:len(valid)][valid])),
        "max_abs": float(error.max(initial=0)), "p99_abs": float(np.percentile(error, 99)),
        "first_native": truth[:10].tolist(), "first_candidate": candidate[:10].tolist(),
    }, indent=2))


if __name__ == "__main__":
    main()
