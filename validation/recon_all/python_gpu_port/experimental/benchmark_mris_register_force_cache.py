"""Compare cached registration forces with the existing path on a real mesh."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import nibabel.freesurfer.io as fsio
import numpy as np
import torch

from fnit.recon_all.mris_register_nonlinear import (
    first_area_gradient, first_distance_gradient,
    prepare_registration_force_cache, sphere_vertex_normals,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("sphere", "smoothwm", "registered", "report"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    sphere_xyz, faces = fsio.read_geometry(str(args.sphere))
    smooth_xyz, smooth_faces = fsio.read_geometry(str(args.smoothwm))
    registered_xyz, registered_faces = fsio.read_geometry(str(args.registered))
    assert np.array_equal(faces, smooth_faces) and np.array_equal(faces, registered_faces)
    sphere = torch.from_numpy(sphere_xyz.astype(np.float32))
    smoothwm = torch.from_numpy(smooth_xyz.astype(np.float32))
    registered = torch.from_numpy(registered_xyz.astype(np.float32))
    triangles = torch.from_numpy(faces.astype(np.int64))

    started = perf_counter()
    cache = prepare_registration_force_cache(sphere, smoothwm, triangles)
    cache_setup_seconds = perf_counter() - started
    results = []
    for project in (False, True):
        started = perf_counter()
        reference_normals = sphere_vertex_normals(registered, triangles)
        reference_distance = first_distance_gradient(
            sphere, smoothwm, registered, triangles, project=project)
        reference_area = first_area_gradient(
            sphere, smoothwm, registered, triangles, reference_distance,
            project=project)
        reference_seconds = perf_counter() - started
        started = perf_counter()
        cached_normals = sphere_vertex_normals(
            registered, triangles, incidence=cache.incidence)
        cached_distance = first_distance_gradient(
            sphere, smoothwm, registered, triangles,
            project=project, cache=cache)
        cached_area = first_area_gradient(
            sphere, smoothwm, registered, triangles, cached_distance,
            project=project, cache=cache)
        cached_seconds = perf_counter() - started
        comparisons = {}
        for name, reference, actual in (
            ("normals", reference_normals, cached_normals),
            ("distance", reference_distance, cached_distance),
            ("area", reference_area, cached_area),
        ):
            ref, got = reference.numpy(), actual.numpy()
            comparisons[name] = {
                "exact_vertices": int(np.all(ref == got, axis=1).sum()),
                "max_abs_error": float(np.abs(ref - got).max()),
            }
        results.append({"project": project,
                        "reference_seconds": reference_seconds,
                        "cached_seconds": cached_seconds,
                        "comparisons": comparisons})
    report = {"vertices": len(sphere), "faces": len(triangles),
              "cache_setup_seconds": cache_setup_seconds, "cases": results}
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    if any(item["exact_vertices"] != len(sphere)
           for case in results for item in case["comparisons"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
