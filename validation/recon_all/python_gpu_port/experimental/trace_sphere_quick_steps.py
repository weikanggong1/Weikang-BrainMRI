"""Compare native -W 1 snapshots to each experimental Python q-sphere update."""
from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

import nibabel.freesurfer as fs
import numpy as np

from fnit.recon_all.sphere_quick_python import (
    quick_sphere_from_projected, reference_face_areas,
)


class Finished(Exception):
    pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("original", type=Path)
    parser.add_argument("projected", type=Path)
    parser.add_argument("native_snapshot_prefix", type=Path)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--snapshot-offset", type=int, default=0)
    parser.add_argument("--stop-on-mismatch", action="store_true")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--initial-momentum", type=Path)
    args = parser.parse_args()
    original, faces = fs.read_geometry(args.original)
    projected, projected_faces = fs.read_geometry(args.projected)
    assert np.array_equal(faces, projected_faces)
    reference = reference_face_areas(original, faces)
    momentum = None
    if args.initial_momentum is not None:
        momentum = np.fromfile(args.initial_momentum, dtype="<f4").reshape(-1, 3)
    source, start = inspect.getsourcelines(quick_sphere_from_projected)
    append_line = next(start + offset for offset, line in enumerate(source)
                       if "trace.append((k, averages, mode, dt))" in line)
    target = quick_sphere_from_projected.__code__

    def local_trace(frame, event, arg):
        if event != "line" or frame.f_lineno != append_line:
            return local_trace
        i = len(frame.f_locals["trace"]) + 1
        native, native_faces = fs.read_geometry(Path(f"{args.native_snapshot_prefix}{i + args.snapshot_offset:04d}"))
        candidate = frame.f_locals["xyz"]
        assert np.array_equal(faces, native_faces)
        error = np.linalg.norm(candidate.astype(np.float64) - native.astype(np.float64), axis=1)
        exact = int(np.count_nonzero(np.all(candidate == native, axis=1)))
        print(json.dumps({"step": i, "k": frame.f_locals["k"],
                          "averages": frame.f_locals["averages"],
                          "mode": frame.f_locals["mode"], "dt": frame.f_locals["dt"],
                          "exact_vertices": exact,
                          "median_mm": float(np.median(error)), "max_mm": float(np.max(error))}),
              flush=True)
        if i >= args.max_steps or (args.stop_on_mismatch and exact != len(candidate)):
            raise Finished
        return local_trace

    def global_trace(frame, event, arg):
        return local_trace if event == "call" and frame.f_code is target else None

    sys.settrace(global_trace)
    try:
        quick_sphere_from_projected(projected, faces, reference,
                                    float(np.sum(reference, dtype=np.float64)),
                                    niterations=args.iterations,
                                    initial_momentum=momentum)
    except Finished:
        pass
    finally:
        sys.settrace(None)


if __name__ == "__main__":
    main()
