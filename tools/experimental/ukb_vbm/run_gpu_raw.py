"""Run a persistent GPU GM estimator on selected private T1 cases."""

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

from gpu_gm import SynthSegGM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--subjects-root", type=Path, required=True)
    parser.add_argument("--gm-method", choices=("synthseg", "torch-fast"),
                        default="synthseg")
    parser.add_argument("--weights", help="WMH-SynthSeg checkpoint")
    parser.add_argument("--synthstrip-weights", help="SynthStrip checkpoint")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--grid", choices=("input", "native-1mm"), default="input")
    parser.add_argument("--cases", nargs="+", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.gm_method == "torch-fast" and args.grid != "input":
        parser.error("torch-fast outputs use the input grid; set --grid input")
    if args.threads < 1:
        parser.error("--threads must be positive")
    if len(set(args.cases)) != len(args.cases):
        parser.error("--cases contains duplicate IDs")
    cases = {item["case_id"]: item for item in json.loads(
        args.manifest.read_text())["cases"]}
    missing = sorted(set(args.cases) - set(cases))
    if missing:
        parser.error(f"case IDs absent from manifest: {', '.join(missing)}")

    synthseg_names = ("GM_prob.nii.gz", "brain_mask.nii.gz")
    fast_names = (
        "T1_brain.nii.gz", "brain_mask.nii.gz", "T1_brain_pve_0.nii.gz",
        "T1_brain_pve_1.nii.gz", "T1_brain_pve_2.nii.gz",
        "T1_brain_seg.nii.gz", "T1_brain_pveseg.nii.gz",
        "T1_brain_mixeltype.nii.gz", "T1_brain_bias.nii.gz",
        "T1_brain_restore.nii.gz", "GM_prob.nii.gz",
    )
    names = synthseg_names if args.gm_method == "synthseg" else fast_names
    plans = []
    for case_id in args.cases:
        subject = args.subjects_root / case_id
        output = subject / "T1" / (
            "T1_gpu_raw" if args.gm_method == "synthseg" else "T1_gpu_fast")
        timing_name = ("gpu_gm_timing.private.json" if args.gm_method == "synthseg"
                       else "gpu_fast_timing.private.json")
        targets = [output / name for name in names] + [subject / timing_name]
        existing = [path for path in targets if path.exists()]
        if existing and not args.overwrite:
            parser.error(f"{case_id}: output exists: {existing[0]}; use --overwrite")
        plans.append((case_id, subject, output, timing_name))
    started = time.perf_counter()
    if args.gm_method == "synthseg":
        model = SynthSegGM(weights=args.weights, device=args.device,
                           threads=args.threads)
        extractor = None
    else:
        from freesurfer_torch.fast import TorchFAST
        from freesurfer_torch.synthstrip import SynthStrip
        extractor = SynthStrip(weights=args.synthstrip_weights,
                               device=args.device, threads=args.threads)
        model = TorchFAST(device=args.device, threads=args.threads)
    load_seconds = time.perf_counter() - started
    print(f"model_load_sec={load_seconds:.3f}", flush=True)
    for case_id, subject, output, timing_name in plans:
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f".{output.name}.tmp-",
                                         dir=output.parent) as directory:
            stage = Path(directory)
            started = time.perf_counter()
            if args.gm_method == "synthseg":
                model.run(cases[case_id]["path"], stage / "GM_prob.nii.gz",
                          brain_output=stage / "brain_mask.nii.gz", grid=args.grid)
            else:
                stripped = extractor(cases[case_id]["path"])
                stripped.image.save(stage / "T1_brain.nii.gz")
                stripped.mask.save(stage / "brain_mask.nii.gz")
                result = model(stripped.image, mask=stripped.mask)
                files = {
                    "T1_brain_pve_0.nii.gz": result.pve_csf,
                    "T1_brain_pve_1.nii.gz": result.pve_gm,
                    "T1_brain_pve_2.nii.gz": result.pve_wm,
                    "T1_brain_seg.nii.gz": result.hard_segmentation,
                    "T1_brain_pveseg.nii.gz": result.pve_segmentation,
                    "T1_brain_mixeltype.nii.gz": result.mixel_type,
                    "T1_brain_bias.nii.gz": result.bias_field,
                    "T1_brain_restore.nii.gz": result.restored,
                    "GM_prob.nii.gz": result.pve_gm,
                }
                for filename, volume in files.items():
                    volume.save(stage / filename)
            seconds = time.perf_counter() - started
            timing = stage / timing_name
            timing.write_text(json.dumps({
                "case_id": case_id, "gpu_gm_raw_seconds": seconds,
                "model_load_seconds": load_seconds, "grid": args.grid,
                "gm_method": args.gm_method,
                "timing_scope": ("gpu_gm_raw_seconds is warm per-case inference and "
                                 "output I/O with persistent models; model_load_seconds "
                                 "is paid once for this multi-case invocation"),
            }, indent=2) + "\n")
            output.mkdir(parents=True, exist_ok=True)
            for name in names:
                os.replace(stage / name, output / name)
            os.replace(timing, subject / timing_name)
        print(f"{case_id} {args.gm_method}_sec={seconds:.3f}", flush=True)


if __name__ == "__main__":
    main()
