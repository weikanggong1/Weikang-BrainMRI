"""Compare the RH installed smoothwm checkpoint with independent source inputs."""

import hashlib
import json
from pathlib import Path

import nibabel.freesurfer.io as fsio
import numpy as np
import torch

from fnit.recon_all.mris_register_nonlinear import sphere_vertex_normals
from fnit.recon_all.mris_register_smoothwm import smoothwm_mean_curvature


def compare(candidate, reference):
    error = np.abs(candidate - reference)
    different = np.flatnonzero(candidate != reference)
    return {"exact_elements": int(candidate.size - len(different)),
            "total_elements": int(candidate.size),
            "first_different_element": int(different[0]) if len(different) else None,
            "max_abs_error": float(error.max())}


def main():
    directory = Path(__file__).resolve().parent
    smoothwm = directory / "rh.smoothwm"
    positions, faces = fsio.read_geometry(str(smoothwm))
    positions = positions.astype(np.float32)
    vertices = torch.from_numpy(positions)
    triangles = torch.from_numpy(faces.astype(np.int64))
    torch.set_num_threads(4)
    normals = sphere_vertex_normals(vertices, triangles).numpy()
    curvature = smoothwm_mean_curvature(vertices, triangles).numpy()
    native_positions = np.fromfile(directory / "native_smoothwm_positions.bin", dtype="<f4").reshape(-1, 3)
    native_normals = np.fromfile(directory / "native_smoothwm_normals.bin", dtype="<f4").reshape(-1, 3)
    native_curvature = np.fromfile(directory / "native_smoothwm_curvature.bin", dtype="<f4")
    assert native_positions.shape == positions.shape == native_normals.shape
    assert native_curvature.shape == curvature.shape
    report = {
        "smoothwm_sha256": hashlib.sha256(smoothwm.read_bytes()).hexdigest(),
        "native_capture_sha256": hashlib.sha256(
            (directory / "native_smoothwm_capture.json").read_bytes()).hexdigest(),
        "candidate_module_sha256": {name: hashlib.sha256(
            (directory / "src/fnit/recon_all" / name).read_bytes()).hexdigest()
            for name in ("mris_register_nonlinear.py", "mris_register_smoothwm.py")},
        "positions": compare(positions, native_positions),
        "normals": compare(normals, native_normals),
        "raw_curvature": compare(curvature, native_curvature),
        "raw_curvature_sha256": hashlib.sha256(curvature.tobytes()).hexdigest(),
        "selected": {str(index): {"position": positions[index].tolist(),
                                  "normal_candidate": normals[index].tolist(),
                                  "normal_native": native_normals[index].tolist(),
                                  "H_candidate": float(curvature[index]),
                                  "H_native": float(native_curvature[index])}
                     for index in (0, 57378)},
    }
    (directory / "rh_smoothwm_first_operator.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
