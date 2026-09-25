"""Compare a verified intermediate registration sphere with final FreeSurfer output."""

import argparse
import hashlib
import json
from pathlib import Path

import nibabel.freesurfer.io as fsio
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("fifth_surface", type=Path)
parser.add_argument("final_surface", type=Path)
parser.add_argument("report", type=Path)
args = parser.parse_args()
fifth, fifth_faces = fsio.read_geometry(str(args.fifth_surface))
final, final_faces = fsio.read_geometry(str(args.final_surface))
assert fifth.shape == final.shape
assert np.array_equal(fifth_faces, final_faces)
absolute = np.abs(fifth - final)
distance = np.linalg.norm(fifth - final, axis=1)
report = {
    "fifth_surface_sha256": hashlib.sha256(args.fifth_surface.read_bytes()).hexdigest(),
    "final_surface_sha256": hashlib.sha256(args.final_surface.read_bytes()).hexdigest(),
    "vertices": len(fifth), "faces_equal": True,
    "exact_vertices": int(np.count_nonzero(np.all(fifth == final, axis=1))),
    "max_abs_coordinate_error_mm": float(absolute.max()),
    "median_vertex_distance_mm": float(np.median(distance)),
    "p95_vertex_distance_mm": float(np.percentile(distance, 95)),
}
args.report.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report))
