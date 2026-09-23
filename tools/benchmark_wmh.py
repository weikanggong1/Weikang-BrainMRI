"""Run matched, fresh-process WMH-SynthSeg commands on a FLAIR directory.

Input filenames must end in _FLAIR.nii.gz. All arms use --crop, save a lesion
probability map and a volume CSV, and use the same official checkpoint.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def make_shadow_home(directory, freesurfer_home, weights):
    directory.mkdir(parents=True, exist_ok=True)
    for name, destination in (("bin", freesurfer_home / "bin"),
                              ("python", freesurfer_home / "python"),
                              ("models", weights)):
        link = directory / name
        if link.is_symlink() and link.resolve() == destination.resolve():
            continue
        if link.exists() or link.is_symlink():
            raise FileExistsError(f"Existing reference-home entry: {link}")
        link.symlink_to(destination, target_is_directory=True)
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--arm", required=True,
                        choices=("official-cpu", "official-cuda", "torch-cpu", "torch-cuda"))
    parser.add_argument("--freesurfer-home", type=Path,
                        default=Path(os.environ.get("FREESURFER_HOME", "/missing")))
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--physical-gpu", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.input_dir = args.input_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.weights = args.weights.resolve()
    args.freesurfer_home = args.freesurfer_home.resolve()
    files = sorted(args.input_dir.glob("*_FLAIR.nii.gz"))
    if not files:
        parser.error("input directory has no *_FLAIR.nii.gz files")
    if not (args.weights / "WMH-SynthSeg_v10_231110.pth").is_file():
        parser.error("official WMH checkpoint is missing in --weights")
    official = args.arm.startswith("official-")
    cuda = args.arm.endswith("cuda")
    if official and not args.freesurfer_home.is_dir():
        parser.error("official arm requires --freesurfer-home")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shadow = (make_shadow_home(args.output_dir / "reference_home", args.freesurfer_home,
                               args.weights) if official else None)
    report_file = args.output_dir / "execution.json"
    records = json.loads(report_file.read_text()) if args.resume and report_file.is_file() else []
    completed = {item["case"] for item in records if item["ok"]}
    allowed = sorted(os.sched_getaffinity(0))
    affinity = allowed[args.threads:2 * args.threads] if cuda else allowed[:args.threads]
    if len(affinity) != args.threads:
        affinity = allowed[:args.threads]
    affinity_arg = ",".join(map(str, affinity))

    for image in files:
        case = image.name.removesuffix("_FLAIR.nii.gz")
        if case in completed:
            continue
        seg = args.output_dir / f"{case}_seg.nii.gz"
        prob = args.output_dir / f"{case}_seg.lesion_probs.nii.gz"
        csv_file = args.output_dir / f"{case}_volumes.csv"
        for output in (seg, prob, csv_file):
            if output.exists():
                raise FileExistsError(f"Existing output for {case}: {output}; use a new directory")
        if args.arm == "official-cpu":
            command = [str(shadow / "bin/mri_WMHsynthseg")]
        elif args.arm == "official-cuda":
            source = args.freesurfer_home / "python/packages/WMHSynthSeg/inference.py"
            command = [sys.executable, str(source)]
        else:
            command = [sys.executable, "-m", "freesurfer_torch.cli", "wmh-synthseg"]
        command += ["--i", str(image), "--o", str(seg), "--csv_vols", str(csv_file),
                    "--device", "cuda:0" if cuda else "cpu", "--threads", str(args.threads),
                    "--crop", "--save_lesion_probabilities"]
        if not official:
            command += ["--weights", str(args.weights)]
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = str(args.physical_gpu) if cuda else ""
        environment["NVIDIA_TF32_OVERRIDE"] = "0"
        if official:
            environment["FREESURFER_HOME"] = str(shadow)
        timed_command = ["taskset", "-c", affinity_arg, *command]
        started = time.perf_counter()
        with (args.output_dir / f"{case}.log").open("w") as log:
            finished = subprocess.run(timed_command, env=environment, stdout=log,
                                      stderr=subprocess.STDOUT, check=False)
        seconds = time.perf_counter() - started
        ok = finished.returncode == 0 and all(path.is_file() for path in (seg, prob, csv_file))
        records.append({"case": case, "arm": args.arm, "seconds": seconds,
                        "returncode": finished.returncode, "ok": ok,
                        "cpu_affinity": affinity, "physical_gpu": args.physical_gpu if cuda else None})
        part = report_file.with_suffix(".json.part")
        part.write_text(json.dumps(records, indent=2) + "\n")
        part.replace(report_file)
        print(case, args.arm, round(seconds, 2), "ok" if ok else "FAILED", flush=True)
        if not ok:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
