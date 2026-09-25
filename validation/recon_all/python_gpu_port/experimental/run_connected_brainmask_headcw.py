"""Replay the one-call T1-to-brainmask API on the fixed public T1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from fnit.recon_all.input_brainmask_chain import run_input_brainmask_chain


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--t1", type=Path, required=True)
    parser.add_argument("--subject-dir", type=Path, required=True)
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--assets-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    report = run_input_brainmask_chain(
        args.t1, args.subject_dir, args.weights_dir, args.assets_dir,
        device="cpu", threads=4)
    report["end_to_end_seconds"] = time.perf_counter() - started
    args.report.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps({key: report[key] for key in (
        "nu", "T1", "brainmask", "end_to_end_seconds",
        "normalize_first_seconds", "brainmask_seconds")}, indent=2))


if __name__ == "__main__":
    main()
