"""Pair the Python and native FreeSurfer N4 postprocessing on one fixed nu0."""

import argparse
from decimal import Decimal
import gzip
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

from fnit.recon_all.n4_wrapper import make_nu


def run_native(bundle, original, nu0, tal, work, env):
    work.mkdir(parents=True, exist_ok=True)
    timings = {}

    def run(stage, program, *args):
        began = time.perf_counter()
        subprocess.run([str(bundle / "bin" / program), *map(str, args)], env=env,
                       capture_output=True, check=True)
        timings[stage] = time.perf_counter() - began

    ones = work / "ones.mgz"
    run("binarize", "mri_binarize", "--i", nu0, "--min", -1, "--o", ones)
    means = []
    for name, volume in (("input", original), ("output", nu0)):
        path = work / f"{name}.mean.dat"
        run(f"segstats_{name}", "mri_segstats", "--id", 1, "--seg", ones,
            "--i", volume, "--sum", work / "sum.junk", "--avgwf", path)
        means.append(Decimal(path.read_text().strip()))
    scale = str(means[0] / means[1])
    scaled, converted, result = (work / name for name in ("scaled.mgz", "converted.mgz", "nu.mgz"))
    run("scale", "mris_calc", "-o", scaled, nu0, "mul", scale)
    run("convert_like", "mri_convert", scaled, converted, "--like", original)
    run("make_uchar", "mri_make_uchar", converted, tal, result)
    run("add_xform", "mri_add_xform_to_header", "-c", tal, result, result)
    return result, timings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("orig", "nu0", "tal", "native-bundle", "license", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "FREESURFER_HOME": str(args.native_bundle),
           "FS_LICENSE": str(args.license), "LD_LIBRARY_PATH": str(args.native_bundle / "lib"),
           "PATH": f"{args.native_bundle / 'bin'}:{os.environ['PATH']}"}
    rows = []
    for trial in range(args.repeats):
        outputs, timings = {}, {}
        for side in (("native", "python") if trial % 2 == 0 else ("python", "native")):
            began = time.perf_counter()
            if side == "native":
                outputs[side], stages = run_native(args.native_bundle, args.orig, args.nu0,
                                                   args.tal, args.output_dir / f"native_{trial}", env)
                timings["native_stages"] = stages
            else:
                outputs[side] = args.output_dir / f"python_{trial}.mgz"
                make_nu(args.orig, args.nu0, args.tal, outputs[side])
            timings[side] = time.perf_counter() - began
        native_raw = gzip.decompress(outputs["native"].read_bytes())
        python_raw = gzip.decompress(outputs["python"].read_bytes())
        rows.append({"trial": trial, "decompressed_mgh_identical": native_raw == python_raw,
                     "native_sha256": hashlib.sha256(native_raw).hexdigest(),
                     "python_sha256": hashlib.sha256(python_raw).hexdigest(),
                     "wall_seconds": timings})
    report = {"repeats": args.repeats,
              "timing_scope": "Fixed nu0 input through final nu; native subprocess launch and IO included; Python resident function import excluded; N4 bias correction excluded",
              "all_decompressed_mgh_identical": all(row["decompressed_mgh_identical"] for row in rows),
              "median_seconds": {side: statistics.median(row["wall_seconds"][side] for row in rows)
                                 for side in ("native", "python")}, "rows": rows}
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
