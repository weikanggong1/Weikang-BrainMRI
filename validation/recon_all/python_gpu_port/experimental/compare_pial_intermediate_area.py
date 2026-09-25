"""Quantify vertex-area impact of same-input official/source pial meshes."""

import hashlib
import json
from pathlib import Path
import sys

from nibabel import freesurfer as fs
import numpy as np
import fnit.recon_all.surface_area_gpu as area_module
from fnit.recon_all.surface_area_gpu import vertex_area


def area(path):
    xyz, faces = fs.read_geometry(str(path))
    return faces, vertex_area(xyz, faces, device="cpu")


reference, candidate = map(Path, sys.argv[1:3])
faces_a, a = area(reference)
faces_b, b = area(candidate)
if not np.array_equal(faces_a, faces_b):
    raise ValueError("ordered faces differ")
delta = np.abs(a - b)
threshold = .001 + .001 * np.abs(a)
report = {
    "scope": "first LH pial normal CLI, not final pial or complete recon-all",
    "method": "validated fnit.recon_all.surface_area_gpu.vertex_area CPU kernel",
    "area_kernel_sha256": hashlib.sha256(Path(area_module.__file__).read_bytes()).hexdigest(),
    "reference_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
    "candidate_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
    "vertices": len(a), "face_count": len(faces_a),
    "area_total_reference_mm2": float(a.sum()),
    "area_total_candidate_mm2": float(b.sum()),
    "exact_vertex_areas": int(np.count_nonzero(a == b)),
    "outliers_at_existing_area_rule": int(np.count_nonzero(delta > threshold)),
    "max_abs_vertex_area_mm2": float(delta.max()),
    "p99_abs_vertex_area_mm2": float(np.quantile(delta, .99)),
    "first_outlier": int(np.flatnonzero(delta > threshold)[0])
    if np.any(delta > threshold) else None,
}
Path(sys.argv[3]).write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report))
