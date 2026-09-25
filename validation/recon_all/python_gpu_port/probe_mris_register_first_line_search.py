"""Frozen first mris_register line-search check; no native trial SSE is used."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import nibabel as nib
import nibabel.freesurfer.io as fsio
import numpy as np
import torch

from fnit.recon_all.mris_register_line_search import (
    first_registration_line_search, first_registration_sse,
)
from fnit.recon_all.mris_register_nonlinear import (
    apply_spherical_gradient, face_area_normals, ordered_neighbors_from_faces,
    original_chord_distances, registration_orig_area, registration_total_area,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sphere", type=Path)
    parser.add_argument("smoothwm", type=Path)
    parser.add_argument("curvature", type=Path)
    parser.add_argument("target_grid", type=Path)
    parser.add_argument("capture_dir", type=Path)
    parser.add_argument("native_first_step", type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    start = perf_counter()
    sphere, faces = fsio.read_geometry(str(args.sphere))
    smoothwm, original_faces = fsio.read_geometry(str(args.smoothwm))
    native, native_faces = fsio.read_geometry(str(args.native_first_step))
    assert np.array_equal(faces, original_faces) and np.array_equal(faces, native_faces)
    captured_positions = np.fromfile(args.capture_dir / "positions_before_line.bin", "<f4").reshape(-1, 3)
    captured_gradient = np.fromfile(args.capture_dir / "gradient_after_average.bin", "<f4").reshape(-1, 3)
    curve = fsio.read_morph_data(str(args.curvature)).astype(np.float32)
    grid = np.asarray(nib.load(str(args.target_grid)).dataobj, dtype=np.float32)
    device = args.device
    x = torch.from_numpy(captured_positions).to(device)
    gradient = torch.from_numpy(captured_gradient).to(device)
    triangles = torch.from_numpy(faces.astype(np.int64)).to(device)
    original = torch.from_numpy(smoothwm.astype(np.float32)).to(device)
    neighbors, degrees = ordered_neighbors_from_faces(triangles, len(sphere))
    original_distances = original_chord_distances(original, neighbors, degrees)
    original_face_areas, _ = face_area_normals(original, triangles)
    expected_dists = np.fromfile(args.capture_dir / "distance_original_distances.bin", "<f4")
    expected_areas = np.fromfile(args.capture_dir / "area_original_areas.bin", "<f4")
    mask = torch.arange(neighbors.shape[1], device=device)[None, :] < degrees[:, None]
    assert np.array_equal(original_distances[mask].cpu().numpy(), expected_dists)
    assert np.array_equal(original_face_areas.cpu().numpy(), expected_areas)
    mean = torch.from_numpy(grid[:, :, 3].T.copy()).to(device)
    variance = torch.from_numpy(grid[:, :, 4].T.copy()).to(device)
    curvature = torch.from_numpy(curve).to(device)
    original_area = registration_orig_area(torch.from_numpy(sphere).to(device), triangles)
    total_area = registration_total_area()
    setup_seconds = perf_counter() - start
    start = perf_counter()

    sample_terms = []

    def objective(positions):
        terms = first_registration_sse(positions, triangles, neighbors, degrees,
                                       original_distances, original_face_areas,
                                       curvature, mean, variance,
                                       original_area, total_area,
                                       return_terms=True)
        sample_terms.append(terms)
        return sum(terms.values())

    selected_dt, samples = first_registration_line_search(x, gradient, objective)
    predicted = apply_spherical_gradient(x, gradient, selected_dt).cpu().numpy()
    if x.is_cuda:
        torch.cuda.synchronize(x.device)
    objective_seconds = perf_counter() - start
    print(json.dumps({
        "device": device, "vertices": len(sphere),
        "selected_dt": selected_dt, "samples": samples,
        "sample_terms": sample_terms,
        "native_dt_reference": {"lh": 70.93703089944263,
                                "rh": 69.6503677368164}[args.sphere.name[:2]],
        "exact_ordered_vertices": int(np.count_nonzero(np.all(predicted == native, axis=1))),
        "max_abs_coordinate_error_mm": float(np.max(np.abs(predicted - native))),
        "seconds": {"setup": setup_seconds, "objective": objective_seconds},
        "sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest()
                    for name, path in (("sphere", args.sphere),
                                       ("smoothwm", args.smoothwm),
                                       ("curvature", args.curvature),
                                       ("target_grid", args.target_grid),
                                       ("positions", args.capture_dir / "positions_before_line.bin"),
                                       ("gradient", args.capture_dir / "gradient_after_average.bin"),
                                       ("native_step", args.native_first_step))},
    }, indent=2))


if __name__ == "__main__":
    main()
