"""Paired zero-iteration mris_inflate centering probe on frozen surfaces."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import time

from fnit.recon_all.inflate_center_python import center_surface
from probe_inflate import compare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-surface", type=Path, required=True)
    parser.add_argument("--right-surface", type=Path, required=True)
    parser.add_argument("--native-binary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for hemi, surface in (("lh", args.left_surface), ("rh", args.right_surface)):
        timings = {"native": [], "python": []}
        comparison = None
        for repeat in range(args.repeats):
            outputs = {side: args.output_dir / f"{hemi}.{side}.{repeat}.surf"
                       for side in timings}
            for side in (("native", "python") if repeat % 2 == 0
                         else ("python", "native")):
                start = time.perf_counter()
                if side == "native":
                    subprocess.run([str(args.native_binary), "-no-save-sulc", "-N", "0",
                                    "-A", "0", str(surface), str(outputs[side])],
                                   capture_output=True, check=True)
                else:
                    center_surface(surface, outputs[side])
                timings[side].append(time.perf_counter() - start)
            comparison = compare(outputs["native"], outputs["python"])
            if (comparison["identical_vertex_count"] != comparison["vertices"] or
                    comparison["identical_face_count"] != comparison["faces"]):
                raise ValueError(f"{hemi} repeat {repeat} differs: {comparison}")
        rows.append({"hemi": hemi, "comparison": comparison,
                     "wall_seconds": timings,
                     "median_seconds": {side: statistics.median(values)
                                        for side, values in timings.items()}})
    report = {"host": platform.node(), "native_binary": str(args.native_binary),
              "input_sha256": {hemi: hashlib.sha256(path.read_bytes()).hexdigest()
                               for hemi, path in (("lh", args.left_surface),
                                                  ("rh", args.right_surface))},
              "repeats": args.repeats,
              "timing_scope": "same-host file I/O; native process startup included; Python imports excluded",
              "scope": "zero-iteration diagnostic, not full inflation",
              "rows": rows}
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
