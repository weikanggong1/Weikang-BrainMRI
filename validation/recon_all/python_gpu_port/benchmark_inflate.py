"""Paired FreeSurfer/Python ``mris_inflate -no-save-sulc`` comparison."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import time

import numpy as np

from fnit.recon_all.inflate_python import inflate_surface
from probe_inflate import read_surface


def compare(native: Path, candidate: Path):
    n_xyz, n_faces, _ = read_surface(native)
    p_xyz, p_faces, _ = read_surface(candidate)
    faces_identical = bool(np.array_equal(n_faces, p_faces))
    n_xyz = n_xyz.astype(np.float64)
    p_xyz = p_xyz.astype(np.float64)
    faces = n_faces.astype(np.int32)
    distance = np.linalg.norm(n_xyz - p_xyz, axis=1)

    def metrics(xyz):
        v0, v1, v2 = (xyz[faces[:, i]] for i in range(3))
        cross = np.cross(v1 - v0, v2 - v0)
        face_area = 0.5 * np.linalg.norm(cross, axis=1)
        vertex_area = np.zeros(len(xyz), np.float64)
        np.add.at(vertex_area, faces.reshape(-1), np.repeat(face_area / 3, 3))
        volume = np.sum(np.einsum("ij,ij->i", v0, np.cross(v1, v2))) / 6
        return float(face_area.sum()), vertex_area, float(volume)

    n_area, n_vertex_area, n_volume = metrics(n_xyz)
    p_area, p_vertex_area, p_volume = metrics(p_xyz)
    return {"vertices": len(n_xyz), "faces": len(faces),
            "faces_identical": faces_identical,
            "vertex_distance_mm": {"median": float(np.median(distance)),
                                   "rms": float(np.sqrt(np.mean(distance ** 2))),
                                   "max": float(distance.max())},
            "surface_area_mm2": {"native": n_area, "python": p_area,
                                 "relative_error": abs(p_area - n_area) / n_area},
            "barycentric_vertex_area_mm2": {"max_absolute_error": float(np.max(np.abs(p_vertex_area - n_vertex_area))),
                                            "median_relative_error": float(np.median(
                                    np.abs(p_vertex_area - n_vertex_area) /
                                    np.maximum(n_vertex_area, 1e-12)))},
            "signed_enclosed_volume_mm3": {"native": n_volume, "python": p_volume,
                                           "relative_error": abs(p_volume - n_volume) /
                                           max(abs(n_volume), 1e-12)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-surface", type=Path, required=True)
    parser.add_argument("--right-surface", type=Path, required=True)
    parser.add_argument("--native-binary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for hemi, surface in (("lh", args.left_surface), ("rh", args.right_surface)):
        elapsed = {"native": [], "python": []}
        comparison = None
        for repeat in range(args.repeats):
            outputs = {side: args.output_dir / f"{hemi}.{side}.{repeat}.surf"
                       for side in elapsed}
            for side in (("native", "python") if repeat % 2 == 0
                         else ("python", "native")):
                begin = time.perf_counter()
                if side == "native":
                    subprocess.run([str(args.native_binary), "-no-save-sulc",
                                    str(surface), str(outputs[side])],
                                   capture_output=True, check=True)
                else:
                    inflate_surface(surface, outputs[side])
                elapsed[side].append(time.perf_counter() - begin)
            comparison = compare(outputs["native"], outputs["python"])
            if not comparison["faces_identical"] or comparison["vertex_distance_mm"]["max"] > 0.001:
                raise ValueError(f"{hemi} repeat {repeat} outside parity gate: {comparison}")
        rows.append({"hemi": hemi, "comparison": comparison,
                     "wall_seconds": elapsed,
                     "median_seconds": {side: statistics.median(values)
                                        for side, values in elapsed.items()}})
    report = {"host": platform.node(), "native_binary": str(args.native_binary),
              "input_sha256": {hemi: hashlib.sha256(path.read_bytes()).hexdigest()
                               for hemi, path in (("lh", args.left_surface),
                                                  ("rh", args.right_surface))},
              "repeats": args.repeats,
              "timing_scope": "same-host file I/O; native process startup included; Python imports excluded",
              "vertex_parity_gate_mm": 0.001,
              "rows": rows}
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
