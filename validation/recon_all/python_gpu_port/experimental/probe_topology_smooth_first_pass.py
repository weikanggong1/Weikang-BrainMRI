"""Find first installed-vs-Python difference within topology sphere smoothing."""

from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path

import nibabel.freesurfer.io as fsio
import numpy as np

from fnit.recon_all.inflate_center_python import center_vertices
from fnit.recon_all import topology_preflight_python as preflight


def _bits(values: np.ndarray) -> list[str]:
    return [f"0x{int(v):08x}" for v in np.asarray(values, np.float32).ravel().view(np.uint32)]


def _native(path: Path) -> dict[int, dict]:
    rows = {}
    for line in path.read_text().splitlines():
        f = line.split()
        if f[:1] != ["VERTEX"]:
            continue
        if f[2] != "VNUM" or f[4] != "VNEIGH" or f[11] != "AVG" or f[15] != "XYZ":
            raise ValueError(f"unexpected native first-pass line: {line}")
        rows[int(f[1])] = {
            "degree": int(f[3]), "neighbors_prefix": [int(x) for x in f[5:11]],
            "average": [f"0x{int(x, 16):08x}" for x in f[12:15]],
            "xyz": [f"0x{int(x, 16):08x}" for x in f[16:19]],
        }
    if len(rows) != 7:
        raise ValueError(f"expected seven diagnostic vertices, got {len(rows)}")
    return rows


def inspect(sphere_path: Path, native_path: Path) -> dict:
    raw, faces = fsio.read_geometry(str(sphere_path))
    faces = np.ascontiguousarray(faces, np.int32)
    centered = center_vertices(np.asarray(raw, np.float32))
    dist = np.sqrt(np.sum(centered.astype(np.float64) ** 2, axis=1))
    projected = (centered.astype(np.float64) * (100.0 / dist)[:, None]).astype(np.float32)
    neighbors = preflight.ordered_neighbors(faces, len(projected))
    degrees = np.fromiter((len(row) for row in neighbors), np.int32, count=len(projected))
    indices = np.zeros((len(projected), int(degrees.max())), np.int32)
    for vertex, row in enumerate(neighbors):
        indices[vertex, :len(row)] = row
    one_pass = preflight._smooth_on_sphere(projected, indices, degrees, 1)
    native = _native(native_path)
    rows = {}
    for vertex, reference in native.items():
        ordered = neighbors[vertex]
        accum = np.zeros(3, np.float32)
        for neighbor in ordered:
            for axis in range(3):
                accum[axis] = np.float32(accum[axis] + projected[neighbor, axis])
        average = np.float32(accum / np.float32(len(ordered)))
        avg_bits = _bits(average)
        xyz_bits = _bits(one_pass[vertex])
        rows[str(vertex)] = {
            "degree_equal": len(ordered) == reference["degree"],
            "ordered_neighbor_prefix_equal": ordered[:6] == reference["neighbors_prefix"][:min(6, len(ordered))],
            "ordered_neighbors": ordered,
            "native_neighbor_prefix": reference["neighbors_prefix"],
            "python_accumulator_hex": _bits(accum),
            "python_average_hex": avg_bits,
            "native_average_hex": reference["average"],
            "average_bitwise_equal": sum(a == b for a, b in zip(avg_bits, reference["average"])),
            "python_projected_hex": xyz_bits,
            "native_projected_hex": reference["xyz"],
            "projected_bitwise_equal": sum(a == b for a, b in zip(xyz_bits, reference["xyz"])),
        }
    return {
        "sphere_sha256": sha256(sphere_path.read_bytes()).hexdigest(),
        "native_log_sha256": sha256(native_path.read_bytes()).hexdigest(),
        "vertices": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sphere", type=Path, required=True)
    parser.add_argument("--native-log", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = inspect(args.sphere, args.native_log)
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: {field: value for field, value in row.items()
                            if field in ("degree_equal", "ordered_neighbor_prefix_equal",
                                         "average_bitwise_equal", "projected_bitwise_equal")}
                      for key, row in result["vertices"].items()}, indent=2))


if __name__ == "__main__":
    main()
