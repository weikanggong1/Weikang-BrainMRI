"""Replay the connected Python pretess-to-quick-sphere path on a frozen subject."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def surface_parts(path: Path) -> tuple[int, int, bytes, bytes, bytes]:
    raw = path.read_bytes()
    if raw[:3] != b"\xff\xff\xfe":
        raise ValueError(f"expected triangle surface: {path}")
    offset = raw.index(b"\n\n", 3) + 2
    vertices = int.from_bytes(raw[offset:offset + 4], "big")
    faces = int.from_bytes(raw[offset + 4:offset + 8], "big")
    xyz_start = offset + 8
    face_start = xyz_start + vertices * 12
    face_end = face_start + faces * 12
    geometry_start = raw.index(b"valid = ", face_end)
    geometry_end = raw.index(b"\n", raw.index(b"cras   = ", geometry_start)) + 1
    return (vertices, faces, raw[xyz_start:face_start],
            raw[face_start:face_end], raw[geometry_start:geometry_end])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--native-pretess", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--hemisphere", choices=("lh", "rh"), default="lh")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    mri = args.scratch / "mri"
    surf = args.scratch / "surf"
    mri.mkdir(parents=True, exist_ok=True)
    surf.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(args.source / "src")
    stage_rows = []

    def run(stage: str, module: str, arguments: list[str], cwd: Path) -> None:
        command = [sys.executable, "-m", "fnit.recon_all." + module, *arguments]
        start = time.perf_counter()
        subprocess.run(command, cwd=cwd, env=env, check=True, capture_output=True, text=True)
        stage_rows.append({"stage": stage, "seconds": time.perf_counter() - start,
                           "arguments": arguments})

    hemi = args.hemisphere
    label = "255" if hemi == "lh" else "127"
    pretess = mri / ("filled-pretess" + label + ".mgz")
    run("mri_pretess", "pretess_python",
        [str(args.subject / "mri/filled.mgz"), label,
         str(args.subject / "mri/norm.mgz"), str(pretess)], args.scratch)
    actual = nib.load(str(pretess))
    expected = nib.load(str(args.native_pretess))
    same_pretess = bool(np.array_equal(np.asanyarray(actual.dataobj),
                                      np.asanyarray(expected.dataobj))
                        and np.array_equal(actual.affine, expected.affine))
    if not same_pretess:
        raise AssertionError("pretess voxel or affine mismatch")

    raw = surf / (hemi + ".orig.raw.quad")
    outputs = {
        "orig": surf / (hemi + ".orig.nofix"),
        "smoothwm": surf / (hemi + ".smoothwm.nofix"),
        "inflated": surf / (hemi + ".inflated.nofix"),
        "qsphere": surf / (hemi + ".qsphere.nofix"),
    }
    run("mri_tessellate", "tessellate_gpu",
        ["../mri/filled-pretess" + label + ".mgz", label, str(raw), "--device", "cpu"], surf)
    run("mris_extract_main_component", "extract_main_component_python",
        [str(raw), str(outputs["orig"])], surf)
    run("mris_smooth", "smooth_surface_python",
        [str(outputs["orig"]), str(outputs["smoothwm"]), "--device", "cpu"], surf)
    run("mris_inflate", "inflate_python",
        [str(outputs["smoothwm"]), str(outputs["inflated"])], surf)
    run("mris_sphere -q", "sphere_quick_python",
        [str(outputs["inflated"]), str(outputs["qsphere"])], surf)

    comparisons = {}
    for name, output in outputs.items():
        official = args.subject / "surf" / (hemi + "." + name + ".nofix")
        vertices, faces, xyz, triangles, geometry = surface_parts(output)
        ref_vertices, ref_faces, ref_xyz, ref_triangles, ref_geometry = surface_parts(official)
        comparisons[name] = {
            "vertices": vertices,
            "faces": faces,
            "exact_vertices": vertices == ref_vertices and xyz == ref_xyz,
            "exact_faces": faces == ref_faces and triangles == ref_triangles,
            "exact_volume_geometry": geometry == ref_geometry,
            "python_sha256": sha256(output),
            "native_sha256": sha256(official),
        }
    result = {
        "subject": args.subject.name,
        "hemisphere": hemi,
        "input_filled_sha256": sha256(args.subject / "mri/filled.mgz"),
        "input_norm_sha256": sha256(args.subject / "mri/norm.mgz"),
        "python_pretess_sha256": sha256(pretess),
        "native_pretess_sha256": sha256(args.native_pretess),
        "pretess_voxels_and_affine_exact": same_pretess,
        "source_sha256": {
            name: sha256(args.source / "src/fnit/recon_all" / (name + ".py"))
            for name in ("pretess_python", "tessellate_gpu",
                         "extract_main_component_python", "smooth_surface_python",
                         "inflate_python", "sphere_quick_python")
        },
        "stages": stage_rows,
        "surfaces": comparisons,
    }
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not all(all(row[key] for key in ("exact_vertices", "exact_faces",
                                        "exact_volume_geometry"))
               for row in comparisons.values()):
        raise AssertionError("one or more connected surface outputs differ")


if __name__ == "__main__":
    main()
