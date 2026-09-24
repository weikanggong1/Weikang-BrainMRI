#!/usr/bin/env python3
"""Run the experimental raw-T1-to-VBM PyTorch workflow for one image.

Grey matter can come from the original SynthSeg-based estimator or TorchFAST.
The latter also performs log-domain bias-field correction. The registration
stage is not FNIRT and the full workflow is not interchangeable with UKB v1.5.
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
    parser.add_argument("--gm-method", choices=("synthseg", "torch-fast"),
                        default="synthseg")
    parser.add_argument("--synthstrip-weights", type=Path,
                        help="SynthStrip checkpoint used by torch-fast without --brain-mask")
    parser.add_argument("--brain-mask", type=Path,
                        help="existing input-grid brain mask; skips SynthStrip")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int)
    parser.add_argument("--crop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--affine-steps", type=int, default=50)
    parser.add_argument("--deform-steps", type=int, default=40)
    parser.add_argument("--smoothness", type=float, default=10.0,
                        help="validated experiment setting; tuned on one case")
    parser.add_argument("--fast-no-bias", action="store_true",
                        help="disable TorchFAST bias-field correction")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for label, path in (("input", args.input), ("template", args.template)):
        if not path.is_file():
            parser.error(f"{label} does not exist: {path}")
    if args.weights is not None and not args.weights.is_file():
        parser.error(f"weights do not exist: {args.weights}")
    if args.synthstrip_weights is not None and not args.synthstrip_weights.is_file():
        parser.error(f"SynthStrip weights do not exist: {args.synthstrip_weights}")
    if args.brain_mask is not None and not args.brain_mask.is_file():
        parser.error(f"brain mask does not exist: {args.brain_mask}")
    if args.gm_method == "synthseg" and args.brain_mask is not None:
        parser.error("--brain-mask is only used with --gm-method torch-fast")
    if args.threads is not None and args.threads < 1:
        parser.error("--threads must be positive")
    if args.affine_steps < 0 or args.deform_steps < 0 or args.smoothness < 0:
        parser.error("steps and smoothness must be non-negative")

    outputs = [args.output_dir / name for name in FINAL_FILES.values()]
    outputs += [args.output_dir / "GM_prob.nii.gz",
                args.output_dir / "brain_mask.nii.gz",
                args.output_dir / "report.private.json"]
    fast_filenames = (
        "T1_brain.nii.gz", "T1_brain_pve_0.nii.gz", "T1_brain_pve_1.nii.gz",
        "T1_brain_pve_2.nii.gz", "T1_brain_seg.nii.gz",
        "T1_brain_pveseg.nii.gz", "T1_brain_mixeltype.nii.gz",
        "T1_brain_bias.nii.gz", "T1_brain_restore.nii.gz",
    )
    if args.gm_method == "torch-fast":
        outputs += [args.output_dir / name for name in fast_filenames]
    existing = [path for path in outputs if path.exists()]
    if existing and not args.overwrite:
        parser.error(f"output exists: {existing[0]}; use --overwrite to replace it")
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    prefix_name = f".{args.output_dir.name or 'vbm'}.tmp-"
    with tempfile.TemporaryDirectory(prefix=prefix_name,
                                     dir=args.output_dir.parent) as directory:
        stage = Path(directory)
        total_started = time.perf_counter()
        gm_path = stage / "GM_prob.nii.gz"
        brain_path = stage / "brain_mask.nii.gz"
        gm_details = {"method": args.gm_method}
        brain_extraction_sec = 0.0
        load_started = time.perf_counter()
        if args.gm_method == "synthseg":
            estimator = SynthSegGM(weights=args.weights, device=args.device,
                                   threads=args.threads, crop=args.crop)
            model_load_sec = time.perf_counter() - load_started
            gm_started = time.perf_counter()
            estimator.run(args.input, gm_path, brain_output=brain_path, grid="input")
            gm_sec = time.perf_counter() - gm_started
            del estimator
            gm_details.update({"mask_source": "wmh-synthseg", "crop": args.crop})
        else:
            import numpy as np
            import surfa as sf
            from freesurfer_torch.fast import TorchFAST

            estimator = TorchFAST(
                device=args.device, threads=args.threads,
                bias_fwhm_mm=0.0 if args.fast_no_bias else 20.0,
            )

            if args.brain_mask is None:
                from freesurfer_torch.synthstrip import SynthStrip
                extractor = SynthStrip(weights=args.synthstrip_weights,
                                       device=args.device, threads=args.threads)
                model_load_sec = time.perf_counter() - load_started
                extraction_started = time.perf_counter()
                stripped = extractor(args.input)
                brain_extraction_sec = time.perf_counter() - extraction_started
                stripped.image.save(stage / "T1_brain.nii.gz")
                stripped.mask.save(brain_path)
                fast_image, fast_mask = stripped.image, stripped.mask
                del extractor, stripped
                gm_details["mask_source"] = "synthstrip"
            else:
                model_load_sec = time.perf_counter() - load_started
                fast_image = sf.load_volume(str(args.input))
                fast_mask = sf.load_volume(str(args.brain_mask))
                input_data = np.asarray(fast_image.data)
                mask_data = np.asarray(fast_mask.data)
                brain = fast_image.copy()
                brain[mask_data <= 0] = min(float(np.min(input_data)), 0.0)
                brain.save(stage / "T1_brain.nii.gz")
                fast_mask.save(brain_path)
                fast_image = brain
                gm_details["mask_source"] = "explicit"

            gm_started = time.perf_counter()
            fast_result = estimator(fast_image, mask=fast_mask)
            fast_outputs = {
                "T1_brain_pve_0.nii.gz": fast_result.pve_csf,
                "T1_brain_pve_1.nii.gz": fast_result.pve_gm,
                "T1_brain_pve_2.nii.gz": fast_result.pve_wm,
                "T1_brain_seg.nii.gz": fast_result.hard_segmentation,
                "T1_brain_pveseg.nii.gz": fast_result.pve_segmentation,
                "T1_brain_mixeltype.nii.gz": fast_result.mixel_type,
                "T1_brain_bias.nii.gz": fast_result.bias_field,
                "T1_brain_restore.nii.gz": fast_result.restored,
            }
            for filename, volume in fast_outputs.items():
                volume.save(stage / filename)
            fast_result.pve_gm.save(gm_path)
            gm_sec = time.perf_counter() - gm_started
            inside = np.asarray(fast_mask.data) > 0
            bias = np.asarray(fast_result.bias_field.data)
            gm_details.update({
                "bias_correction": not args.fast_no_bias,
                "fast_defaults": {
                    "classes": 3, "image_type": "T1", "init_iterations": 15,
                    "bias_iterations": 4,
                    "fixed_iterations": 4, "bias_fwhm_mm": 0.0 if args.fast_no_bias else 20.0,
                    "init_mrf": 0.02, "mrf": 0.1, "mixel_mrf": 0.3,
                    "pve_steps": 100,
                },
                "tissue_means": list(fast_result.tissue_means),
                "tissue_variances": list(fast_result.tissue_variances),
                "bias_range_inside_mask": [float(bias[inside].min()),
                                           float(bias[inside].max())],
            })
            del estimator, fast_result
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
                "gm_method": args.gm_method,
                "affine_steps_per_scale": args.affine_steps,
                "deform_steps_per_scale": args.deform_steps,
                "smoothness": args.smoothness,
                "torch_threads": torch.get_num_threads(),
            },
            "timing_sec": {
                "model_load": model_load_sec,
                "brain_extraction": brain_extraction_sec,
                "gm_inference_and_io": gm_sec,
                "registration_jacobian_modulation_and_io": registration_sec,
                "cold_total": time.perf_counter() - total_started,
            },
            "outputs": {
                "gm_probability": str(args.output_dir / "GM_prob.nii.gz"),
                "brain_mask": str(args.output_dir / "brain_mask.nii.gz"),
                **final_paths,
            },
            "gm_estimation": gm_details,
            "registration": {
                key: value for key, value in registration_report.items()
                if key not in {"moving", "fixed", "outputs"}
            },
        }
        (stage / "report.private.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n")

        args.output_dir.mkdir(parents=True, exist_ok=True)
        staged_names = ["GM_prob.nii.gz", "brain_mask.nii.gz",
                        *FINAL_FILES.values(), "report.private.json"]
        if args.gm_method == "torch-fast":
            staged_names += list(fast_filenames)
        for filename in staged_names:
            os.replace(stage / filename, args.output_dir / filename)
    print(json.dumps({"output_dir": str(args.output_dir),
                      "cold_total_sec": report["timing_sec"]["cold_total"]},
                     indent=2))


if __name__ == "__main__":
    main()
