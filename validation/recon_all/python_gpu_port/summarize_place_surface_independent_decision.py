"""Audit fixed LH pial decisions and displayed objectives after independent replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
from pathlib import Path

import nibabel as nib
import numpy as np


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def printed(value: float, decimals: int) -> str:
    native_float = struct.unpack("f", struct.pack("f", value))[0]
    return f"{native_float:.{decimals}f}"


def mesh_match(candidate: np.ndarray, path: Path) -> dict:
    reference, _ = nib.freesurfer.read_geometry(path)
    exact = candidate.view(np.uint32) == reference.astype(np.float32).view(np.uint32)
    return {
        "exact_vertices": int(np.count_nonzero(np.all(exact, axis=1))),
        "total_vertices": len(candidate),
        "exact_components": int(exact.sum()),
        "total_components": int(exact.size),
        "sha256": sha(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--middle", type=Path, required=True)
    parser.add_argument("--last", type=Path)
    parser.add_argument("--native-log", type=Path, required=True)
    parser.add_argument("--step37-candidate", type=Path, required=True)
    parser.add_argument("--step37-reference", type=Path, required=True)
    parser.add_argument("--step37-incompatible-reference", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report_paths = (args.first, args.middle) + ((args.last,) if args.last else ())
    parts = [json.loads(path.read_text()) for path in report_paths]
    steps = [step for part in parts for step in part["steps"]]
    if [step["step"] for step in steps] != list(range(1, 42)):
        raise ValueError("independent reports do not cover contiguous steps 1..41")
    log = args.native_log.read_text()
    accepted_mismatch = []
    initial_mismatch = []
    reduction_mismatch = []
    pass_ends = []
    starts = {0: parts[0]["initial_objective"]}
    starts.update({item["outer_pass"]: item["initial_objective"]
                   for item in parts[1].get("outer_pass_starts", [])})
    for outer in range(4):
        chunk = log.split(f"Iteration {outer} =========================================", 1)[1].split(
            "maximum number of reductions reached", 1)[0]
        native = {int(index): (sse, rms) for index, sse, rms in re.findall(
            r"(?m)^(\d{3}): dt: [0-9.]+, sse=([0-9.]+), rms=([0-9.]+)", chunk)}
        own = [step for step in steps if step.get("outer_pass", 0) == outer]
        pass_ends.append(own[-1]["step"])
        initial = starts[outer]
        if (printed(initial["sse"], 1), printed(initial["rms"], 3)) != native[0]:
            initial_mismatch.append(outer)
        for step in own:
            objective = step["trials"][-1]["objective"]
            if (printed(objective["sse"], 1), printed(objective["rms"], 3)) != native[step["step"]]:
                accepted_mismatch.append(step["step"])
        native_reductions = re.findall(
            r"rms = ([0-9.]+)/[0-9.]+, sse=([0-9.]+)/[0-9.]+, time step reduction", chunk)
        own_reductions = [trial for step in own for trial in step["trials"] if trial["reduced"]]
        if len(native_reductions) != len(own_reductions):
            reduction_mismatch.append({"outer_pass": outer, "reason": "count"})
        else:
            for index, ((rms, sse), trial) in enumerate(zip(native_reductions, own_reductions)):
                objective = trial["objective"]
                python_print = (printed(objective["rms"], 4), printed(objective["sse"], 1))
                if python_print != (rms, sse):
                    reduction_mismatch.append({
                        "outer_pass": outer, "reduction": index + 1,
                        "python_print_rms_sse": python_print,
                        "native_print_rms_sse": [rms, sse],
                    })
    candidate = np.load(args.step37_candidate)["xyz"].astype(np.float32)
    corrected = mesh_match(candidate, args.step37_reference)
    incompatible = mesh_match(candidate, args.step37_incompatible_reference)
    checkpoints = {step["step"]: step["coordinate_comparison"] for step in steps
                   if step["coordinate_comparison"] is not None}
    checkpoints[37] = corrected
    exact_checkpoints = [index for index, result in checkpoints.items()
                         if result["exact_components"] == result["total_components"]]
    report = {
        "scope": "Fixed LH pial optimizer; independent SSE/RMS decisions 1..41; installed log only post-decision reference",
        "sha256": {str(path): sha(path) for path in (
            *report_paths, args.native_log, args.step37_candidate)},
        "pass_endpoints": pass_ends,
        "decision_matches": sum(step["schedule_match"] for step in steps),
        "decision_steps": len(steps),
        "accepted_objective_print_matches": 41 - len(accepted_mismatch),
        "accepted_objective_print_mismatch_steps": accepted_mismatch,
        "pass_initial_objective_print_matches": 4 - len(initial_mismatch),
        "pass_initial_objective_print_mismatch_passes": initial_mismatch,
        "reduction_objective_print_matches": 12 - len(reduction_mismatch),
        "reduction_objective_print_mismatches": reduction_mismatch,
        "exact_ram_checkpoints": sorted(exact_checkpoints),
        "ram_checkpoints": sorted(checkpoints),
        "step37_correct_reference": corrected,
        "step37_incompatible_reference": incompatible,
        "decision_and_geometry_mismatch": None if (
            len(exact_checkpoints) == len(checkpoints)
            and all(step["schedule_match"] for step in steps)) else "see fields above",
        "objective_print_mismatch": None if (
            not accepted_mismatch and not initial_mismatch and not reduction_mismatch
        ) else "see fields above",
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
