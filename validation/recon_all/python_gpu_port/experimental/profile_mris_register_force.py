"""Profile the current source-order spherical-registration force on a real mesh."""

import argparse
import cProfile
import io
import json
import pstats
from pathlib import Path
from time import perf_counter

import nibabel.freesurfer.io as fsio
import numpy as np
import torch

from fnit.recon_all.mris_register_nonlinear import (
    first_area_gradient, first_distance_gradient, ordered_neighbors_from_faces,
    sphere_vertex_normals, three_hop_avg_nbrs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("sphere", "smoothwm", "registered", "report"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    sphere, faces = fsio.read_geometry(str(args.sphere))
    smoothwm, _ = fsio.read_geometry(str(args.smoothwm))
    registered, _ = fsio.read_geometry(str(args.registered))
    sphere = torch.from_numpy(sphere.astype(np.float32))
    smoothwm = torch.from_numpy(smoothwm.astype(np.float32))
    registered = torch.from_numpy(registered.astype(np.float32))
    faces = torch.from_numpy(faces.astype(np.int64))
    timings = {}

    def measure(name, fn):
        started = perf_counter()
        result = fn()
        timings[name] = perf_counter() - started
        return result

    neighbors, degrees = measure("ordered_neighbors", lambda:
        ordered_neighbors_from_faces(faces, len(sphere)))
    measure("three_hop_count", lambda: three_hop_avg_nbrs(neighbors, degrees))
    measure("vertex_normals", lambda: sphere_vertex_normals(registered, faces))
    gradient = measure("distance_force", lambda: first_distance_gradient(
        sphere, smoothwm, registered, faces, project=False))
    measure("area_force", lambda: first_area_gradient(
        sphere, smoothwm, registered, faces, gradient, project=False))
    profiler = cProfile.Profile()
    profiler.enable()
    first_distance_gradient(sphere, smoothwm, registered, faces, project=False)
    profiler.disable()
    stream = io.StringIO()
    pstats.Stats(profiler, stream=stream).sort_stats("cumulative").print_stats(18)
    report = {"vertices": len(sphere), "faces": len(faces),
              "seconds": timings, "distance_profile": stream.getvalue()}
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"vertices": len(sphere), "seconds": timings}))


if __name__ == "__main__":
    main()
