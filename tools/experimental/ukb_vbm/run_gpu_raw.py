"""Run the experimental SynthSeg GM estimator on private T1 cases."""

import argparse
import json
import time
from pathlib import Path

from gpu_gm import SynthSegGM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--subjects-root", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--grid", choices=("input", "native-1mm"), default="input")
    parser.add_argument("--cases", nargs="+", required=True)
    args = parser.parse_args()
    cases = {item["case_id"]: item for item in json.load(open(args.manifest))["cases"]}
    started = time.perf_counter()
    model = SynthSegGM(weights=args.weights, device=args.device)
    load_seconds = time.perf_counter() - started
    print(f"model_load_sec={load_seconds:.3f}", flush=True)
    for case_id in args.cases:
        subject = Path(args.subjects_root) / case_id
        output = subject / "T1" / "T1_gpu_raw"
        output.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        model.run(cases[case_id]["path"], output / "GM_prob.nii.gz",
                  brain_output=output / "brain_mask.nii.gz", grid=args.grid)
        seconds = time.perf_counter() - started
        (subject / "gpu_gm_timing.private.json").write_text(json.dumps({
            "case_id": case_id, "gpu_gm_raw_seconds": seconds,
            "model_load_seconds": load_seconds, "grid": args.grid,
        }, indent=2))
        print(f"{case_id} gpu_gm_raw_sec={seconds:.3f}", flush=True)


if __name__ == "__main__":
    main()
