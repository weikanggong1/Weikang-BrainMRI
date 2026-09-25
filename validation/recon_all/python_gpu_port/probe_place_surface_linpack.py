"""Small float32 LINPACK-style QR inverse experiment on curvature matrices."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def snrm2(values: np.ndarray, start: int) -> np.float32:
    """LINPACK BLAS SNRM2 scaled float32 reduction."""
    if len(values) - start == 1:
        return np.float32(abs(values[start]))
    scale = np.float32(0)
    ssq = np.float32(1)
    for value in values[start:]:
        if value == 0:
            continue
        absolute = np.float32(abs(value))
        if scale < absolute:
            ratio = np.float32(scale / absolute)
            ssq = np.float32(np.float32(ssq * np.float32(ratio * ratio)) + np.float32(1))
            scale = absolute
        else:
            ratio = np.float32(absolute / scale)
            ssq = np.float32(ssq + np.float32(ratio * ratio))
    return np.float32(float(scale) * math.sqrt(float(ssq)))


def inverse5(gram: np.ndarray) -> np.ndarray:
    work = gram.copy()
    aux = np.zeros(5, dtype=np.float32)
    for col in range(5):
        if col == 4:
            continue
        norm = snrm2(work[:, col], col)
        if norm == 0:
            continue
        if work[col, col] < 0:
            norm = np.float32(-norm)
        reciprocal = np.float32(np.float32(1.0) / norm)
        for row in range(col, 5):
            work[row, col] = np.float32(work[row, col] * reciprocal)
        work[col, col] = np.float32(work[col, col] + np.float32(1))
        for other in range(col + 1, 5):
            dot = np.float32(0)
            for row in range(col, 5):
                dot = np.float32(dot + np.float32(work[row, col] * work[row, other]))
            scale = np.float32(-dot / work[col, col])
            for row in range(col, 5):
                work[row, other] = np.float32(work[row, other] + np.float32(scale * work[row, col]))
        aux[col] = work[col, col]
        work[col, col] = np.float32(-norm)

    inverse = np.zeros((5, 5), dtype=np.float32)
    for rhs in range(5):
        result = np.zeros(5, dtype=np.float32)
        result[rhs] = np.float32(1)
        for col in range(4):
            diagonal = work[col, col]
            work[col, col] = aux[col]
            dot = np.float32(0)
            for row in range(col, 5):
                dot = np.float32(dot + np.float32(work[row, col] * result[row]))
            scale = np.float32(-dot / work[col, col])
            for row in range(col, 5):
                result[row] = np.float32(result[row] + np.float32(scale * work[row, col]))
            work[col, col] = diagonal
        for col in range(4, -1, -1):
            result[col] = np.float32(result[col] / work[col, col])
            scale = np.float32(-result[col])
            for row in range(col):
                result[row] = np.float32(result[row] + np.float32(scale * work[row, col]))
        inverse[:, rhs] = result
    return inverse


def native_inverse(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw = path.read_bytes()
    count = int(np.frombuffer(raw, dtype="<i4", count=1)[0])
    values = np.frombuffer(raw, dtype="<f4", offset=4)
    design = values[:count*5].reshape(count, 5)
    height = values[count*5:count*6]
    cursor = count*6
    gram = values[cursor:cursor + 25].reshape(5, 5)
    inverse = values[cursor + 25:cursor + 50].reshape(5, 5)
    return gram, inverse, design, height


def scalar_from_inverse(inverse: np.ndarray, design: np.ndarray, height: np.ndarray) -> np.float32:
    result = np.float32(0)
    for neighbor in range(len(height)):
        pseudo = np.float32(0)
        for item in range(5):
            pseudo = np.float32(pseudo + np.float32(inverse[4, item] * design[neighbor, item]))
        result = np.float32(result + np.float32(pseudo * height[neighbor]))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--vertices", type=int, nargs="+", required=True)
    args = parser.parse_args()
    report = []
    for vertex in args.vertices:
        gram, truth, design, height = native_inverse(args.probe / f"lh.curvature.matrix.{vertex}")
        candidate = inverse5(gram)
        candidate_scalar = scalar_from_inverse(candidate, design, height)
        native_scalar = scalar_from_inverse(truth, design, height)
        difference = np.abs(candidate.astype(np.float64) - truth.astype(np.float64))
        report.append({
            "vertex": vertex,
            "exact_inverse_elements": int(np.count_nonzero(candidate == truth)),
            "max_inverse_error": float(difference.max()),
            "native_scalar": float(native_scalar),
            "candidate_scalar": float(candidate_scalar),
            "scalar_error": float(abs(float(candidate_scalar) - float(native_scalar))),
            "native_row_5": truth[4].tolist(),
            "candidate_row_5": candidate[4].tolist(),
        })
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
