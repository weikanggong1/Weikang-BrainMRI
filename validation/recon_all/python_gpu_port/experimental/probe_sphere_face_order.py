"""Isolated first q-sphere gradient arithmetic probe using passive GDB arrays."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import nibabel.freesurfer as fs
import numpy as np

from fnit.recon_all.sphere_quick_python import reference_face_areas
from fnit.recon_all.sphere_python import initial_scale


def stats(candidate: np.ndarray, native: np.ndarray) -> dict:
    delta = candidate.astype(np.float64) - native.astype(np.float64)
    return {
        "exact": int(np.count_nonzero(candidate == native)),
        "total": int(native.size),
        "max_abs": float(np.max(np.abs(delta))),
        "mean_abs": float(np.mean(np.abs(delta))),
        "first": candidate[0].tolist(),
    }


def main() -> None:
    surface, directory = Path(sys.argv[1]), Path(sys.argv[2])
    source, faces = fs.read_geometry(surface)
    coords = np.fromfile(directory / "positions_before_line.bin", "<f4").reshape(-1, 3)
    native_gradient = np.fromfile(directory / "gradient_before_average.bin", "<f4").reshape(-1, 3)
    face_data = np.fromfile(directory / "face_normal_original_area.bin", "<f4").reshape(-1, 4)
    face_area = np.fromfile(directory / "face_current_area.bin", "<f4")
    native_normal, orig_area = face_data[:, :3], face_data[:, 3]
    ref_area = reference_face_areas(source, faces)
    ref_points = initial_scale(source)[faces]
    ref_a = ref_points[:, 1] - ref_points[:, 0]
    ref_b = ref_points[:, 2] - ref_points[:, 0]
    ref_cross = np.cross(ref_a, ref_b)
    ref_length = np.sqrt(np.sum(ref_cross * ref_cross, axis=1, dtype=np.float32))
    ref_normal = ref_cross / ref_length[:, None]
    points = coords[faces]
    a, b = points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]
    cross = np.cross(a, b)
    length = np.sqrt(np.sum(cross * cross, axis=1, dtype=np.float32))
    center = points[:, 0] + points[:, 1] + points[:, 2]
    dot = center[:, 0] * cross[:, 0]
    dot += center[:, 1] * cross[:, 1]
    dot += center[:, 2] * cross[:, 2]
    sign = np.where(dot < 0, -1, 1).astype(np.float32)
    python_normal = cross * sign[:, None] / length[:, None]
    python_area = length * np.float32(.5) * sign
    ratio = float(np.float32(np.float32(6759.59326171875) / np.float32(125663.703125)))
    v20 = points[:, 2] - points[:, 0]
    v12 = points[:, 1] - points[:, 2]
    alt_cross = np.stack((-v12[:, 1]*v20[:, 2] + v20[:, 1]*v12[:, 2],
                           v12[:, 0]*v20[:, 2] - v20[:, 0]*v12[:, 2],
                          -v12[:, 0]*v20[:, 1] + v20[:, 0]*v12[:, 1]), axis=1)
    alt_len = np.sqrt(np.sum(alt_cross * alt_cross, axis=1, dtype=np.float32))
    normal_variants = {
        "current_div_f32": python_normal,
        "current_mul_recip_f32": cross * sign[:, None] * (np.float32(1) / length)[:, None],
        "current_div_f64length": (cross.astype(np.float64) * sign[:, None] /
                                  np.sqrt(np.sum(cross.astype(np.float64)**2, axis=1))[:, None]).astype(np.float32),
        "alt_edge_div_f32": alt_cross * sign[:, None] / alt_len[:, None],
        "alt_edge_div_current_area": alt_cross * sign[:, None] / (face_area * np.float32(2))[:, None],
        "current_div_native_area": cross * sign[:, None] / (face_area * np.float32(2))[:, None],
    }
    report = {
        "native_normal_variants": {key: stats(value, native_normal) for key,value in normal_variants.items()},
        "native_normal_vs_python": stats(python_normal, native_normal),
        "native_normal_vs_reference": stats(ref_normal, native_normal),
        "native_area_vs_python": stats(python_area, face_area),
        "native_orig_area_vs_python": stats(ref_area, orig_area),
        "ratio": ratio,
    }
    for name, normal, area in (("python", python_normal, python_area),
                               ("reference_norm", ref_normal, python_area),
                               ("native_norm", native_normal, python_area),
                               ("native_norm_area", native_normal, face_area)):
        adjusted_area = ratio * area.astype(np.float64)
        delta = (adjusted_area - orig_area.astype(np.float64)) / (1 + np.exp(10 * adjusted_area))
        a_cross_n = np.cross(a, normal)
        b_cross_n = np.cross(b, normal)
        terms = np.stack(((b_cross_n - a_cross_n) * delta[:, None],
                          b_cross_n * (-delta[:, None]),
                          a_cross_n * delta[:, None]), axis=1).astype(np.float32)
        result_interleaved = np.zeros_like(coords)
        np.add.at(result_interleaved, faces.ravel(), terms.reshape(-1, 3))
        result_corners = np.zeros_like(coords)
        for corner in range(3):
            np.add.at(result_corners, faces[:, corner], terms[:, corner])
        report[name] = {
            "interleaved": stats(result_interleaved, native_gradient),
            "corners": stats(result_corners, native_gradient),
        }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
