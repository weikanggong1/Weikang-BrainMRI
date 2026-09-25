"""Paired first angle/area gradient from frozen surface geometry."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import nibabel.freesurfer.io as fsio
import numpy as np
import torch

from fnit.recon_all.mris_register_kernels import project_sphere
from fnit.recon_all.mris_register_nonlinear import (
    face_area_normals, first_area_gradient, first_distance_gradient,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("input_sphere", "smoothwm", "rigid_sphere", "native_after_distance",
                 "native_after_area", "native_current_areas", "native_original_areas",
                 "native_face_normals"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    sphere, faces = fsio.read_geometry(str(args.input_sphere))
    original, original_faces = fsio.read_geometry(str(args.smoothwm))
    rigid, rigid_faces = fsio.read_geometry(str(args.rigid_sphere))
    assert np.array_equal(faces, original_faces) and np.array_equal(faces, rigid_faces)
    read_xyz = lambda path: np.fromfile(path, dtype="<f4").reshape(-1, 3)
    before = read_xyz(args.native_after_distance)
    native = read_xyz(args.native_after_area)
    native_areas = np.fromfile(args.native_current_areas, dtype="<f4")
    native_original_areas = np.fromfile(args.native_original_areas, dtype="<f4")
    native_normals = read_xyz(args.native_face_normals)
    vertices = torch.from_numpy(sphere).to(args.device)
    original = torch.from_numpy(original).to(args.device)
    rigid = torch.from_numpy(rigid).to(args.device)
    triangles = torch.from_numpy(faces.astype(np.int64)).to(args.device)
    start = perf_counter()
    after = first_area_gradient(vertices, original, rigid, triangles,
                                torch.from_numpy(before).to(args.device))
    if after.is_cuda:
        torch.cuda.synchronize(after.device)
    area_seconds = perf_counter() - start
    after = after.cpu().numpy()
    start = perf_counter()
    independent_distance = first_distance_gradient(vertices, original, rigid, triangles)
    independent_after = first_area_gradient(vertices, original, rigid, triangles,
                                            independent_distance)
    if independent_after.is_cuda:
        torch.cuda.synchronize(independent_after.device)
    combined_seconds = perf_counter() - start
    independent_after = independent_after.cpu().numpy()
    current_areas, normals = face_area_normals(project_sphere(rigid.float()), triangles)
    original_areas, _ = face_area_normals(original.float(), triangles)
    current_areas = current_areas.cpu().numpy()
    normals = normals.cpu().numpy()
    original_areas = original_areas.cpu().numpy()
    print(json.dumps({
        "device": args.device,
        "vertices": len(sphere),
        "faces": len(faces),
        "exact_current_face_areas": int(np.count_nonzero(current_areas == native_areas)),
        "exact_original_face_areas": int(np.count_nonzero(original_areas == native_original_areas)),
        "exact_current_face_normals": int(np.count_nonzero(np.all(normals == native_normals, axis=1))),
        "exact_after_area_with_native_distance_gradient": int(np.count_nonzero(np.all(after == native, axis=1))),
        "exact_after_area_from_input_surfaces": int(np.count_nonzero(np.all(independent_after == native, axis=1))),
        "max_abs_from_input_surfaces": float(np.max(np.abs(independent_after - native))),
        "seconds_excluding_io": {"area_only": area_seconds,
                                 "distance_plus_area": combined_seconds},
        "sha256": {key: hashlib.sha256(path.read_bytes()).hexdigest()
                   for key, path in (("input_sphere", args.input_sphere), ("smoothwm", args.smoothwm),
                                     ("rigid_sphere", args.rigid_sphere),
                                     ("native_after_distance", args.native_after_distance),
                                     ("native_after_area", args.native_after_area),
                                     ("native_current_areas", args.native_current_areas),
                                     ("native_original_areas", args.native_original_areas),
                                     ("native_face_normals", args.native_face_normals))},
        "predicted_from_surfaces_sha256": hashlib.sha256(independent_after.tobytes()).hexdigest(),
    }, indent=2))


if __name__ == "__main__":
    main()
