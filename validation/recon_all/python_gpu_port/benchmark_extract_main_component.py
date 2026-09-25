"""Paired raw quad → largest triangular surface replay against FreeSurfer 8.2."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import time

from fnit.recon_all.extract_main_component_python import extract_main_component


def compare(native: Path, candidate: Path) -> dict:
    original = native.read_bytes()
    actual = candidate.read_bytes()
    if original[:3] != actual[:3] or original[:3] != b"\xff\xff\xfe":
        raise ValueError("unexpected triangle surface magic")
    original_body = original[original.index(b"\n\n", 3) + 2:]
    candidate_body = actual[actual.index(b"\n\n", 3) + 2:]
    vertices = int.from_bytes(original_body[:4], "big")
    triangles = int.from_bytes(original_body[4:8], "big")
    return {"vertices": vertices, "triangles": triangles,
            "bytes_after_variable_stamp_identical": original_body == candidate_body}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-quad", type=Path, required=True)
    parser.add_argument("--right-quad", type=Path, required=True)
    parser.add_argument("--native-binary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inputs = {"lh": args.left_quad.resolve(), "rh": args.right_quad.resolve()}
    rows = []
    for hemi, surface in inputs.items():
        timings = {"native": [], "python": []}
        comparison = None
        components = None
        for repeat in range(args.repeats):
            outputs = {side: args.output_dir / f"{hemi}.{side}.{repeat}.surf"
                       for side in timings}
            for side in (("native", "python") if repeat % 2 == 0
                         else ("python", "native")):
                start = time.perf_counter()
                if side == "native":
                    subprocess.run([str(args.native_binary), str(surface),
                                    str(outputs[side])], capture_output=True, check=True)
                else:
                    components = extract_main_component(surface, outputs[side])
                timings[side].append(time.perf_counter() - start)
            comparison = compare(outputs["native"], outputs["python"])
            if not comparison["bytes_after_variable_stamp_identical"]:
                raise ValueError(f"{hemi} repeat {repeat} differs: {comparison}")
        rows.append({"hemi": hemi, "components": components,
                     "comparison": comparison, "wall_seconds": timings,
                     "median_seconds": {side: statistics.median(values)
                                        for side, values in timings.items()}})
    report = {"host": platform.node(),
              "native_binary": str(args.native_binary),
              "input_sha256": {hemi: hashlib.sha256(path.read_bytes()).hexdigest()
                               for hemi, path in inputs.items()},
              "repeats": args.repeats,
              "timing_scope": "same-host file I/O; native process startup included; Python imports excluded",
              "rows": rows}
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
