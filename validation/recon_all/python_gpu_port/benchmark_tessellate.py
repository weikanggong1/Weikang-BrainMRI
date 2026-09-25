"""Replay the raw FreeSurfer quad tessellation on identical pretess volumes."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import time

import nibabel as nib
import numpy as np

from fnit.recon_all.tessellate_gpu import tessellate_quads, write_quad_surface


def geometry_comparison(native: Path, candidate: Path) -> dict:
    original = native.read_bytes()
    actual = candidate.read_bytes()
    if original[:3] != actual[:3] or original[:3] != b"\xff\xff\xfd":
        raise ValueError("unexpected FreeSurfer quad surface magic")
    vertex_count = int.from_bytes(original[3:6], "big")
    quad_count = int.from_bytes(original[6:9], "big")
    if original[3:9] != actual[3:9]:
        raise ValueError("vertex or quad count differs")
    core_end = 9 + 12 * (vertex_count + quad_count)
    geometry_end = original.index(b"\n", original.index(b"cras   = ", core_end)) + 1
    return {
        "vertices": vertex_count,
        "quads": quad_count,
        "vertex_and_quad_bytes_identical": original[:core_end] == actual[:core_end],
        "geometry_tags_identical": original[core_end:geometry_end] == actual[core_end:geometry_end],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-pretess", type=Path, required=True)
    parser.add_argument("--right-pretess", type=Path, required=True)
    parser.add_argument("--native-binary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for hemi, value, input_path in (("lh", 255, args.left_pretess),
                                    ("rh", 127, args.right_pretess)):
        input_path = input_path.resolve()
        timings = {"native": [], "python": []}
        comparison = None
        for repeat in range(args.repeats):
            outputs = {side: args.output_dir / f"{hemi}.{side}.{repeat}.quad"
                       for side in timings}
            for side in (("native", "python") if repeat % 2 == 0
                         else ("python", "native")):
                start = time.perf_counter()
                if side == "native":
                    subprocess.run([str(args.native_binary), str(input_path),
                                    str(value), str(outputs[side])],
                                   capture_output=True, check=True)
                else:
                    image = nib.load(str(input_path))
                    volume = np.asanyarray(image.dataobj)
                    vertices, quads = tessellate_quads(
                        volume, value, image.header.get_vox2ras_tkr(), args.device)
                    write_quad_surface(outputs[side], vertices, quads, image, input_path)
                timings[side].append(time.perf_counter() - start)
            comparison = geometry_comparison(outputs["native"], outputs["python"])
            if not all((comparison["vertex_and_quad_bytes_identical"],
                        comparison["geometry_tags_identical"])):
                raise ValueError(f"{hemi} repeat {repeat} differs: {comparison}")
        rows.append({"hemi": hemi, "comparison": comparison,
                     "wall_seconds": timings,
                     "median_seconds": {side: statistics.median(values)
                                        for side, values in timings.items()}})
    inputs = {"lh": args.left_pretess, "rh": args.right_pretess}
    report = {"host": platform.node(), "device": args.device,
              "native_binary": str(args.native_binary),
              "input_sha256": {hemi: hashlib.sha256(path.read_bytes()).hexdigest()
                               for hemi, path in inputs.items()},
              "repeats": args.repeats,
              "timing_scope": "same-host input and output I/O; native process startup included; Python imports excluded",
              "rows": rows}
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
