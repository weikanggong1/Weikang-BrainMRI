"""Inspect selected first-pass curvature vertices from frozen mesh inputs.

Native curvature text is used only as a final comparison, after the independent
Python fit and all intermediate values have been computed.
"""

from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path

import nibabel.freesurfer.io as fsio
import numpy as np

from fnit.recon_all import topology_principal_curvature as curvature
from fnit.recon_all.topology_vnl_svd import add, mul, svd_inverse_3


def _values(array: np.ndarray) -> dict[str, object]:
    values = np.asarray(array, np.float32)
    return {"values": values.tolist(), "float32_hex":
            [f"0x{int(bit):08x}" for bit in values.ravel().view(np.uint32)]}


def inspect(orig: Path, sphere: Path, vertices: list[int], native: Path | None) -> dict:
    xyz, faces = fsio.read_geometry(str(orig))
    spherical, sphere_faces = fsio.read_geometry(str(sphere))
    if not np.array_equal(faces, sphere_faces):
        raise ValueError("orig and qsphere face order differs")
    xyz = np.ascontiguousarray(xyz, np.float32)
    spherical = np.ascontiguousarray(spherical, np.float32)
    faces = np.ascontiguousarray(faces, np.int32)
    offsets, face_ids, corners = curvature._face_index(faces, len(xyz))
    canonical = curvature.center_sphere(
        curvature.project_and_smooth_sphere(spherical, faces))[0]
    labels = curvature.defect_component_labels(canonical, faces)
    ripped = labels != 0
    normals = curvature._vertex_normals(xyz, faces, offsets, face_ids, corners, ripped)
    sphere_normals = curvature._vertex_normals(
        canonical, faces, offsets, face_ids, corners, np.zeros(len(xyz), np.bool_))
    normals[ripped] = sphere_normals[ripped]
    neighbors = curvature.ordered_neighbors(faces, len(xyz))
    two_rings, degree = curvature._two_rings(neighbors)
    gram, rhs, extreme, valid = curvature._fit_normal_equations(
        xyz, normals, two_rings, degree)
    native_values = np.loadtxt(native, dtype=np.float32) if native else None
    report = {
        "input_sha256": {"orig": sha256(orig.read_bytes()).hexdigest(),
                         "qsphere": sha256(sphere.read_bytes()).hexdigest()},
        "native_curvature_sha256": sha256(native.read_bytes()).hexdigest() if native else None,
        "vertices": {},
    }
    for vertex in vertices:
        if vertex < 0 or vertex >= len(xyz):
            raise ValueError(f"vertex {vertex} outside mesh")
        inverse = svd_inverse_3(gram[vertex])
        coeff = np.zeros(3, np.float32)
        for row in range(3):
            value = np.float32(0)
            for col in range(3):
                value = add(value, mul(inverse[row, col], rhs[vertex, col]))
            coeff[row] = value
        hessian = np.array([[np.float32(2 * coeff[0]), np.float32(2 * coeff[1])],
                            [np.float32(2 * coeff[1]), np.float32(2 * coeff[2])]],
                           np.float32)
        a, b, c = float(hessian[0, 0]), float(hessian[0, 1]), float(hessian[1, 1])
        center = (a + c) / 2.0
        spread = np.sqrt(((a - c) / 2.0) ** 2 + b ** 2)
        eigen = np.array([center - spread, center + spread], np.float32)
        k1, k2 = (eigen if abs(eigen[0]) >= abs(eigen[1]) else eigen[::-1])
        if native_values is not None and int(native_values[vertex, 0]) != vertex:
            raise ValueError("native curvature vertex order differs")
        report["vertices"][str(vertex)] = {
            "defect_label": int(labels[vertex]),
            "original_xyz": _values(xyz[vertex]),
            "canonical_xyz": _values(canonical[vertex]),
            "normal": _values(normals[vertex]),
            "ordered_one_ring": [int(v) for v in neighbors[vertex]],
            "canonical_one_ring_xyz": _values(canonical[neighbors[vertex]]),
            "ordered_two_ring": [int(v) for v in two_rings[vertex, :degree[vertex]]],
            "valid_fit_neighbors": int(valid[vertex]),
            "gram": _values(gram[vertex]), "rhs": _values(rhs[vertex]),
            "inverse": _values(inverse), "coeff": _values(coeff),
            "hessian": _values(hessian), "eigen_low_high": _values(eigen),
            "predicted_k1_k2": _values(np.array([k1, k2], np.float32)),
            "native_printed_k1_k2": _values(native_values[vertex, 1:3])
            if native_values is not None else None,
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orig", type=Path, required=True)
    parser.add_argument("--sphere", type=Path, required=True)
    parser.add_argument("--vertices", type=int, nargs="+", required=True)
    parser.add_argument("--native-curvature", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = inspect(args.orig, args.sphere, args.vertices, args.native_curvature)
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"input_sha256": result["input_sha256"],
                      "vertices": args.vertices}, indent=2))


if __name__ == "__main__":
    main()
