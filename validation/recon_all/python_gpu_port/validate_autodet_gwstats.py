"""Compare Python gray/white statistics with the frozen official surface files."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from fnit.recon_all.autodet_gwstats_python import write_autodet_stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--require-exact", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"reference": "FreeSurfer 8.2.0 fixed fs_sub01 run", "cases": {}}
    for hemisphere in ("lh", "rh"):
        reference = args.subject / "surf" / f"autodet.gw.stats.{hemisphere}.dat"
        output = args.output_dir / f"autodet.gw.stats.{hemisphere}.python.dat"
        if output.resolve() == reference.resolve():
            raise ValueError("Python output cannot overwrite the native reference")
        start = time.perf_counter()
        stats = write_autodet_stats(
            args.subject / "mri/brain.finalsurfs.mgz",
            args.subject / "mri/wm.mgz",
            args.subject / "surf" / f"{hemisphere}.orig.premesh",
            output,
            hemisphere,
        )
        native_bytes = reference.read_bytes()
        python_bytes = output.read_bytes()
        native_fields = dict(line.split() for line in native_bytes.decode().splitlines())
        python_fields = dict(line.split() for line in python_bytes.decode().splitlines())
        report["cases"][hemisphere] = {
            "fields": len(stats),
            "exact_fields": sum(native_fields.get(key) == value for key, value in python_fields.items()),
            "byte_exact": python_bytes == native_bytes,
            "native_sha256": hashlib.sha256(native_bytes).hexdigest(),
            "python_sha256": hashlib.sha256(python_bytes).hexdigest(),
            "seconds_including_io_and_jit": time.perf_counter() - start,
            "differences": {key: [native_fields.get(key), value] for key, value in python_fields.items() if native_fields.get(key) != value},
        }
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if args.require_exact and any(not case["byte_exact"] for case in report["cases"].values()):
        raise SystemExit("gray/white statistics differ from native output")


if __name__ == "__main__":
    main()
