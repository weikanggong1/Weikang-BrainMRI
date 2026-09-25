"""Compare first-epoch Python trial SSE with pinned native mris_sphere logs."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import nibabel.freesurfer.io as fsio
import numpy as np

from fnit.recon_all.sphere_python import project_radially
from fnit.recon_all.sphere_standard_line_search import first_epoch_line_search, first_epoch_sse
from fnit.recon_all.sphere_standard_metric import (
    average_standard_metric, sample_standard_metric_matrix,
)
from fnit.recon_all.sphere_standard_unfold import (
    _face_geometry, first_epoch_gradient,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("smoothwm", type=Path)
    parser.add_argument("before", type=Path)
    parser.add_argument("--project-start", action="store_true")
    parser.add_argument("--trial", type=float, action="append", default=[])
    parser.add_argument("--report", type=Path)
    parser.add_argument("--search", action="store_true")
    parser.add_argument("--after", type=Path)
    args = parser.parse_args()
    original, original_faces = fsio.read_geometry(str(args.smoothwm))
    before, faces = fsio.read_geometry(str(args.before))
    if not np.array_equal(faces, original_faces):
        raise ValueError("face order changed")
    faces = np.asarray(faces, np.int32)
    offsets, ids, raw, _ = sample_standard_metric_matrix(original, faces)
    original_dist, _, _ = average_standard_metric(offsets, ids, raw)
    start = (project_radially(np.asarray(before, np.float32), already_sphere=True)
             if args.project_start else np.asarray(before, np.float32))
    _, _, gradient, geometry = first_epoch_gradient(
        start, faces, original, offsets, ids, original_dist)
    original_area, _ = _face_geometry(np.asarray(original, np.float32), faces)
    original_area = np.abs(original_area)
    original_total = np.float32(np.sum(original_area, dtype=np.float64))
    trials = []
    for dt in [0.0, *args.trial]:
        shifted = (start.astype(np.float64)
                   + dt * gradient.astype(np.float64)).astype(np.float32)
        xyz = project_radially(shifted, already_sphere=True) if dt else start
        t0 = time.perf_counter()
        sse = first_epoch_sse(xyz, faces, offsets, ids, original_dist,
                              original_area, original_total)
        trials.append({"dt": dt, "sse": sse, "seconds": time.perf_counter() - t0})
    result = {"geometry": geometry, "trials": trials}
    if args.search:
        t0 = time.perf_counter()
        search = first_epoch_line_search(start, gradient, faces, offsets, ids,
                                         original_dist, original_area, original_total)
        search["seconds"] = time.perf_counter() - t0
        if args.after:
            after, after_faces = fsio.read_geometry(str(args.after))
            if not np.array_equal(faces, after_faces):
                raise ValueError("after face order changed")
            shifted = (start.astype(np.float64) + search["selected_dt"]
                       * gradient.astype(np.float64)).astype(np.float32)
            predicted = project_radially(shifted, already_sphere=True)
            delta = np.linalg.norm(predicted.astype(np.float64)
                                   - after.astype(np.float64), axis=1)
            search["coordinate_check"] = {
                "exact_components": int(np.count_nonzero(predicted == after)),
                "total_components": int(after.size),
                "vertices_le_1e-5_mm": int(np.count_nonzero(delta <= 1e-5)),
                "vertices": len(delta),
                "max_error_mm": float(delta.max()),
                "rms_error_mm": float(np.sqrt(np.mean(delta * delta)))}
        result["search"] = search
    output = json.dumps(result, indent=2)
    if args.report:
        args.report.write_text(output + "\n")
    print(output)


if __name__ == "__main__":
    main()
