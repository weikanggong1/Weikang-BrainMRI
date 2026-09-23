"""Time one SynthSR volume per fresh process, using the same checkpoint in each arm.

Run this script once per arm in a separate output directory. ``official-cuda``
requires --tf-python pointing to a TensorFlow Python with a visible GPU.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def checkpoint_path(location, variant):
    names = {
        "general": "synthsr_v20_230130.h5",
        "lowfield": "synthsr_lowfield_v20_230130.h5",
        "v1": "synthsr_v10_210712.h5",
    }
    path = location / names[variant] if location.is_dir() else location
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.resolve()


def executable(value):
    path = shutil.which(str(value))
    if path is None:
        raise FileNotFoundError(f"Python executable not found: {value}")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True,
                        help="official .h5 file or directory containing the selected variant")
    parser.add_argument("--arm", required=True,
                        choices=("official-cpu", "official-cuda", "torch-cpu", "torch-cuda"))
    parser.add_argument("--freesurfer-home", type=Path,
                        default=os.environ.get("FREESURFER_HOME"))
    parser.add_argument("--official-script", type=Path,
                        help="unmodified official mri_synthsr source; defaults to FreeSurfer's installed script")
    parser.add_argument("--tf-python", help="TensorFlow GPU Python; required for official-cuda")
    parser.add_argument("--torch-python", default=sys.executable,
                        help="Python with this package and PyTorch; defaults to the current interpreter")
    parser.add_argument("--pattern", default="*.nii.gz", help="input filename glob, default: *.nii.gz")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--physical-gpu", type=int, default=0)
    parser.add_argument("--lowfield", action="store_true")
    parser.add_argument("--v1", action="store_true")
    parser.add_argument("--ct", action="store_true")
    parser.add_argument("--disable_flipping", action="store_true")
    parser.add_argument("--disable_sharpening", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.threads < 1 or args.physical_gpu < 0:
        parser.error("--threads must be positive and --physical-gpu nonnegative")
    if shutil.which("taskset") is None:
        parser.error("taskset is required for fixed CPU affinity")
    variant = "v1" if args.v1 else "lowfield" if args.lowfield else "general"
    checkpoint = checkpoint_path(args.weights.expanduser(), variant)
    hasher = hashlib.sha256()
    with checkpoint.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    digest = hasher.hexdigest()
    input_dir = args.input_dir.expanduser().resolve()
    files = sorted(p for p in input_dir.glob(args.pattern) if p.is_file())
    if not files:
        parser.error(f"No files matching {args.pattern} in {args.input_dir}")

    official = args.arm.startswith("official-")
    cuda = args.arm.endswith("cuda")
    fs_home = args.freesurfer_home.expanduser().resolve() if args.freesurfer_home else None
    if official and (fs_home is None or not fs_home.is_dir()):
        parser.error("official arms require --freesurfer-home or FREESURFER_HOME")
    if args.arm == "official-cpu":
        program = fs_home / "bin/mri_synthsr"
        if not program.is_file():
            parser.error(f"Official program not found: {program}")
        prefix = [str(program)]
    elif args.arm == "official-cuda":
        if not args.tf_python:
            parser.error("official-cuda requires --tf-python")
        source = (args.official_script or fs_home / "python/scripts/mri_synthsr").resolve()
        if not source.is_file():
            parser.error(f"Official source not found: {source}")
        prefix = [executable(args.tf_python), str(source)]
    else:
        prefix = [executable(args.torch_python), "-m", "freesurfer_torch.cli", "synthsr"]

    allowed = sorted(os.sched_getaffinity(0))
    if len(allowed) < args.threads:
        parser.error(f"Only {len(allowed)} CPU cores are available to this process")
    affinity = allowed[args.threads:2 * args.threads] if cuda else allowed[:args.threads]
    if len(affinity) < args.threads:
        affinity = allowed[:args.threads]
    affinity_arg = ",".join(map(str, affinity))
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(args.physical_gpu) if cuda else ""
    environment["NVIDIA_TF32_OVERRIDE"] = "0"
    environment["OMP_NUM_THREADS"] = str(args.threads)
    if official:
        environment["FREESURFER_HOME"] = str(fs_home)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.arm == "official-cuda":
        check = subprocess.run(
            [prefix[0], "-c", "import tensorflow as tf; assert tf.config.list_physical_devices('GPU'), 'TensorFlow sees no GPU'"],
            env=environment, capture_output=True, text=True, check=False,
        )
        (output_dir / "cuda_preflight.log").write_text(check.stdout + check.stderr)
        if check.returncode:
            raise SystemExit("TensorFlow GPU preflight failed; see cuda_preflight.log")

    report_file = output_dir / "execution.json"
    records = json.loads(report_file.read_text()) if args.resume and report_file.is_file() else []
    for row in records:
        if (row["arm"], row["variant"], row["weight_sha256"]) != (args.arm, variant, digest):
            raise ValueError("Existing execution.json uses a different arm, variant, or checkpoint")
        if row["ok"] and not (output_dir / f'{row["case"]}_synthsr.nii.gz').is_file():
            raise FileNotFoundError(f'Output missing for completed case: {row["case"]}')
    completed = {row["case"] for row in records if row["ok"]}
    seen = set()
    for image in files:
        relative = image.relative_to(input_dir)
        case = relative.name
        for suffix in (".nii.gz", ".nii", ".mgz", ".npz"):
            if case.endswith(suffix):
                case = case[:-len(suffix)]
                break
        case = "__".join((*relative.parts[:-1], case))
        if case in seen:
            raise ValueError(f"Input paths resolve to the same case name: {case}")
        seen.add(case)
        if case in completed:
            continue
        output = output_dir / f"{case}_synthsr.nii.gz"
        if output.exists():
            raise FileExistsError(f"Existing output: {output}; use a new directory")
        command = [*prefix, "--i", str(image), "--o", str(output),
                   "--threads", str(args.threads)]
        command += ["--model" if official else "--weights", str(checkpoint)]
        if not official:
            command += ["--device", "cuda:0" if cuda else "cpu"]
        elif not cuda:
            command += ["--cpu"]
        for flag in ("lowfield", "v1", "ct", "disable_flipping", "disable_sharpening"):
            if getattr(args, flag):
                command.append("--" + flag)
        started = time.perf_counter()
        with (output_dir / f"{case}.log").open("w") as log:
            process = subprocess.run(["taskset", "-c", affinity_arg, *command],
                                     env=environment, stdout=log, stderr=subprocess.STDOUT,
                                     check=False)
        seconds = time.perf_counter() - started
        ok = process.returncode == 0 and output.is_file() and output.stat().st_size > 0
        records.append({
            "case": case, "arm": args.arm, "variant": variant,
            "seconds": seconds, "returncode": process.returncode, "ok": ok,
            "cpu_affinity": affinity, "physical_gpu": args.physical_gpu if cuda else None,
            "weight_sha256": digest,
        })
        part = report_file.with_suffix(".json.part")
        part.write_text(json.dumps(records, indent=2) + "\n")
        part.replace(report_file)
        print(case, args.arm, round(seconds, 2), "ok" if ok else "FAILED", flush=True)
        if not ok:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
