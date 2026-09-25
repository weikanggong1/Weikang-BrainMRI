"""Compare independent topology sphere stages with installed FreeSurfer GDB snapshots.

The native log selects diagnostic vertices only; the Python coordinates are
computed from the frozen qsphere input and never from native stage values.
"""

from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path

import nibabel.freesurfer.io as fsio
import numpy as np

from fnit.recon_all.inflate_center_python import center_vertices
from fnit.recon_all import topology_preflight_python as preflight
from fnit.recon_all import topology_principal_curvature as curvature


def _bits(values: np.ndarray) -> list[str]:
    array = np.asarray(values, np.float32)
    return [f"0x{int(bit):08x}" for bit in array.ravel().view(np.uint32)]


def _native_stages(path: Path) -> dict[str, dict[int, dict[str, list[str]]]]:
    stages: dict[str, dict[int, dict[str, list[str]]]] = {}
    stage = None
    for line in path.read_text().splitlines():
        fields = line.split()
        if fields[:1] == ["STAGE"]:
            stage = fields[1]
            if stage in stages:
                raise ValueError(f"repeated native stage: {stage}")
            stages[stage] = {}
        elif fields[:1] == ["VERTEX"] and stage is not None:
            if fields[2] != "XYZ" or fields[6] != "HEX" or fields[10] != "NORMAL" or fields[14] != "NHEX":
                raise ValueError(f"unexpected native vertex line: {line}")
            stages[stage][int(fields[1])] = {
                "xyz": [f"0x{int(value, 16):08x}" for value in fields[7:10]],
                "normal": [f"0x{int(value, 16):08x}" for value in fields[15:18]],
            }
    if list(stages) != ["projected", "smoothed", "centered"]:
        raise ValueError(f"unexpected native stage order: {list(stages)}")
    if any(list(stage) != list(stages["projected"]) for stage in stages.values()):
        raise ValueError("native diagnostic vertex order differs between stages")
    return stages


def validate(surface_dir: Path, hemisphere: str, native_log: Path) -> dict:
    orig = surface_dir / f"{hemisphere}.orig.nofix"
    sphere = surface_dir / f"{hemisphere}.qsphere.nofix"
    original, faces = fsio.read_geometry(str(orig))
    spherical, sphere_faces = fsio.read_geometry(str(sphere))
    if not np.array_equal(faces, sphere_faces):
        raise ValueError("orig and qsphere faces differ")
    faces = np.ascontiguousarray(faces, np.int32)
    native = _native_stages(native_log)
    vertices = list(native["projected"])
    centered_input = center_vertices(np.asarray(spherical, np.float32))
    distance = np.sqrt(np.sum(centered_input.astype(np.float64) ** 2, axis=1))
    projected = (centered_input.astype(np.float64) * (100.0 / distance)[:, None]).astype(np.float32)
    smoothed = preflight.project_and_smooth_sphere(spherical, faces)
    centered, iterations = preflight.center_sphere(smoothed)
    offsets, face_ids, corners = curvature._face_index(faces, len(original))
    normals = curvature._vertex_normals(centered, faces, offsets, face_ids, corners,
                                       np.zeros(len(original), np.bool_))
    result = {
        "hemisphere": hemisphere,
        "input_sha256": {"orig": sha256(orig.read_bytes()).hexdigest(),
                         "qsphere": sha256(sphere.read_bytes()).hexdigest()},
        "native_log_sha256": sha256(native_log.read_bytes()).hexdigest(),
        "diagnostic_vertices": vertices,
        "center_sphere_iterations": iterations,
        "stages": {},
    }
    for name, points in (("projected", projected), ("smoothed", smoothed),
                         ("centered", centered)):
        predicted = _bits(points[vertices])
        reference = sum((native[name][vertex]["xyz"] for vertex in vertices), [])
        mismatches = [{"vertex": vertices[index // 3], "axis": "xyz"[index % 3],
                       "python": left, "native": right}
                      for index, (left, right) in enumerate(zip(predicted, reference))
                      if left != right]
        section = {
            "xyz_bitwise_equal": len(predicted) - len(mismatches),
            "xyz_components": len(predicted),
            "first_xyz_mismatch": mismatches[0] if mismatches else None,
            "native_xyz_hex": {str(vertex): native[name][vertex]["xyz"] for vertex in vertices},
            "python_xyz_hex": {str(vertex): _bits(points[vertex]) for vertex in vertices},
            "native_normal_hex": {str(vertex): native[name][vertex]["normal"] for vertex in vertices},
        }
        if name == "centered":
            predicted_normals = _bits(normals[vertices])
            reference_normals = sum((native[name][vertex]["normal"] for vertex in vertices), [])
            section["normal_bitwise_equal"] = sum(left == right for left, right in zip(
                predicted_normals, reference_normals))
            section["normal_components"] = len(predicted_normals)
            section["python_normal_hex"] = {str(vertex): _bits(normals[vertex])
                                            for vertex in vertices}
        result["stages"][name] = section
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surface-dir", type=Path, required=True)
    parser.add_argument("--hemisphere", choices=("lh", "rh"), required=True)
    parser.add_argument("--native-log", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = validate(args.surface_dir, args.hemisphere, args.native_log)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"hemisphere": args.hemisphere,
                      "stages": {name: {"xyz_bitwise_equal": part["xyz_bitwise_equal"],
                                        "xyz_components": part["xyz_components"],
                                        "first_xyz_mismatch": part["first_xyz_mismatch"],
                                        "normal_bitwise_equal": part.get("normal_bitwise_equal")}
                                 for name, part in report["stages"].items()}}, indent=2))


if __name__ == "__main__":
    main()
