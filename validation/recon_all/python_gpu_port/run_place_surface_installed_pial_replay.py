"""Continuously replay four installed pial outer passes with fixed native controls."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--hemisphere", choices=("lh", "rh"), required=True)
    parser.add_argument("--installed-binary", type=Path, required=True)
    parser.add_argument("--native-log", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--checkpoint", action="append", default=[], metavar="STEP:PATH")
    args = parser.parse_args()
    log = args.native_log.read_text()
    endpoints = []
    for outer in range(4):
        chunk = log.split(f"Iteration {outer} =========================================", 1)[1]
        chunk = chunk.split("maximum number of reductions reached", 1)[0]
        endpoints.append(max(int(index) for index in re.findall(r"(?m)^(\d{3}): dt:", chunk)))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    passes = []
    for outer, endpoint in enumerate(endpoints):
        name = args.output_dir / f"{args.hemisphere}.outer{outer}.report.json"
        command = [
            sys.executable, str(Path(__file__).with_name("validate_place_surface_installed_second_step.py")),
            "--subject", str(args.subject), "--root", str(args.root),
            "--module", str(args.module), "--hemisphere", args.hemisphere,
            "--installed-binary", str(args.installed_binary),
            "--native-log", str(args.native_log), "--outer-pass", str(outer),
            "--iterations", str(endpoint), "--output-dir", str(args.output_dir),
            "--report", str(name), "--require-exact",
        ]
        if outer:
            command.extend(("--resume-step", str(endpoints[outer - 1])))
        for spec in args.checkpoint:
            command.extend(("--checkpoint", spec))
        began = time.perf_counter()
        with (args.output_dir / f"{args.hemisphere}.outer{outer}.stdout.log").open("w") as output:
            subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, check=True)
        elapsed = time.perf_counter() - began
        detail = json.loads(name.read_text())
        passes.append({
            "outer_pass": outer,
            "accepted_steps": [endpoints[outer - 1] + 1 if outer else 1, endpoint],
            "report": str(name),
            "report_sha256": digest(name),
            "wall_seconds": elapsed,
            "reported_timing_seconds": detail["timing_seconds"],
            "sum_step_timing_seconds": {
                key: sum(step["timing_seconds"][key] for step in detail["steps"])
                for key in ("gradient", "rejected_trial", "accepted_update",
                            "comparison_and_checkpoint", "total")
            },
            "checkpoint_comparisons": [
                {"step": step["step"], "result": step["accepted_vs_installed"]}
                for step in detail["steps"] if step["accepted_vs_installed"] is not None
            ],
        })
        print(f"outer {outer}: step {endpoint}, {elapsed:.3f} s", flush=True)
    report = {
        "scope": f"{args.hemisphere} pial optimizer; four continuous outer passes",
        "native_control_input": "installed dt and reject/retry schedule; diagnostic replay, not independent scheduler",
        "sha256": {
            "installed_binary": digest(args.installed_binary),
            "native_control_log": digest(args.native_log),
            "white": digest(args.subject / "surf" / f"{args.hemisphere}.white"),
            "brain": digest(args.subject / "mri/brain.finalsurfs.mgz"),
            "wm": digest(args.subject / "mri/wm.mgz"),
            "aseg": digest(args.subject / "mri/aseg.presurf.mgz"),
            "cortex_label": digest(args.subject / "label" / f"{args.hemisphere}.cortex+hipamyg.label"),
            "autodet": digest(args.subject / "surf" / f"autodet.gw.stats.{args.hemisphere}.dat"),
            "optimizer_state": digest(args.output_dir / f"{args.hemisphere}.step{endpoints[-1]:02d}.npz"),
        },
        "accepted_pass_endpoints": endpoints,
        "passes": passes,
        "total_wall_seconds": time.perf_counter() - started,
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"accepted_pass_endpoints": endpoints,
                      "total_wall_seconds": report["total_wall_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
