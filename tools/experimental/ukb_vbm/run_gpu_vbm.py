#!/usr/bin/env python3
"""Run the experimental raw-T1-to-VBM PyTorch workflow for one image.

This combines SynthSeg-derived grey-matter estimation with gpu_register.py.
It writes FSL bb_vbm-compatible final image names, but it is not an FNIRT
implementation and should not be treated as interchangeable with UKB v1.5.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import torch

from gpu_gm import SynthSegGM


FINAL_FILES = {
    "warped_gm": "T1_GM_to_template_GM.nii.gz",
    "jacobian": "T1_GM_JAC_nl.nii.gz",
    "modulated_gm": "T1_GM_to_template_GM_mod.nii.gz",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="single 3D T1 NIfTI")
    parser.add_argument("--template", type=Path, required=True,
                        help="GM template defining the output grid")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path,
                        help="WMH-SynthSeg checkpoint; defaults to package weight lookup")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int)
    parser.add_argument("--crop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--affine-steps", type=int, default=50)
    parser.add_argument("--deform-steps", type=int, default=40)
    parser.add_argument("--smoothness", type=float, default=10.0,
                        help="validated experiment setting; tuned on one case")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for label, path in (("input", args.input), ("template", args.template)):
        if not path.is_file():
            parser.error(f"{label} does not exist: {path}")
    if args.weights is not None and not args.weights.is_file():
        parser.error(f"weights do not exist: {args.weights}")
    if args.threads is not None and args.threads < 1:
        parser.error("--threads must be positive")
    if args.affine_steps < 0 or args.deform_steps < 0 or args.smoothness < 0:
        parser.error("steps and smoothness must be non-negative")

    outputs = [args.output_dir / name for name in FINAL_FILES.values()]
    outputs += [args.output_dir / "GM_prob.nii.gz",
                args.output_dir / "brain_mask.nii.gz",
                args.output_dir / "report.json"]
    existing = [path for path in outputs if path.exists()]
    if existing and not args.overwrite:
        parser.error(f"output exists: {existing[0]}; use --overwrite to replace it")
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    prefix_name = f".{args.output_dir.name or 'vbm'}.tmp-"
    with tempfile.TemporaryDirectory(prefix=prefix_name,
                                     dir=args.output_dir.parent) as directory:
        stage = Path(directory)
        total_started = time.perf_counter()
        load_started = time.perf_counter()
        estimator = SynthSegGM(weights=args.weights, device=args.device,
                               threads=args.threads, crop=args.crop)
        model_load_sec = time.perf_counter() - load_started
        gm_started = time.perf_counter()
        gm_path = stage / "GM_prob.nii.gz"
        brain_path = stage / "brain_mask.nii.gz"
        estimator.run(args.input, gm_path, brain_output=brain_path, grid="input")
        gm_sec = time.perf_counter() - gm_started
        del estimator
        if torch.device(args.device).type == "cuda":
            torch.cuda.empty_cache()

        prefix = stage / "_gpu_register"
        command = [
            sys.executable,
            str(Path(__file__).with_name("gpu_register.py")),
            "--moving", str(gm_path),
            "--fixed", str(args.template),
            "--output-prefix", str(prefix),
            "--device", args.device,
            "--affine-steps", str(args.affine_steps),
            "--deform-steps", str(args.deform_steps),
            "--smoothness", str(args.smoothness),
        ]
        if args.threads is not None:
            command.extend(("--threads", str(args.threads)))
        registration_started = time.perf_counter()
        subprocess.run(command, check=True)
        registration_sec = time.perf_counter() - registration_started

        internal_report_path = Path(str(prefix) + "_report.json")
        registration_report = json.loads(internal_report_path.read_text())
        final_paths = {}
        for field, filename in FINAL_FILES.items():
            source = Path(str(prefix) + f"_{field}.nii.gz")
            destination = stage / filename
            os.replace(source, destination)
            final_paths[field] = str(args.output_dir / filename)
        internal_report_path.unlink()

        report = {
            "status": "experimental",
            "fnirt_equivalent": False,
            "input": str(args.input),
            "template": str(args.template),
            "device": args.device,
            "settings": {
                "crop": args.crop,
                "affine_steps_per_scale": args.affine_steps,
                "deform_steps_per_scale": args.deform_steps,
                "smoothness": args.smoothness,
                "torch_threads": torch.get_num_threads(),
            },
            "timing_sec": {
                "model_load": model_load_sec,
                "gm_inference_and_io": gm_sec,
                "registration_jacobian_modulation_and_io": registration_sec,
                "cold_total": time.perf_counter() - total_started,
            },
            "outputs": {
                "gm_probability": str(args.output_dir / "GM_prob.nii.gz"),
                "brain_mask": str(args.output_dir / "brain_mask.nii.gz"),
                **final_paths,
            },
            "registration": {
                key: value for key, value in registration_report.items()
                if key not in {"moving", "fixed", "outputs"}
            },
        }
        (stage / "report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n")

        args.output_dir.mkdir(parents=True, exist_ok=True)
        staged_names = ["GM_prob.nii.gz", "brain_mask.nii.gz",
                        *FINAL_FILES.values(), "report.json"]
        for filename in staged_names:
            os.replace(stage / filename, args.output_dir / filename)
    print(json.dumps({"output_dir": str(args.output_dir),
                      "cold_total_sec": report["timing_sec"]["cold_total"]},
                     indent=2))


if __name__ == "__main__":
    main()
