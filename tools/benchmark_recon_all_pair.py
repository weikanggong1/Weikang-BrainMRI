"""Serial, opt-in end-to-end benchmark of FreeSurfer 8.2 A and GPU package C.

The default invocation only prints a plan. ``--execute`` creates a new output
directory and runs both reconstructions; existing data are never removed.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time


THREADS = {"OMP_NUM_THREADS": "4", "FS_OMP_NUM_THREADS": "4",
           "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS": "1"}
REQUIRED = ("mri/aparc+aseg.mgz", "stats/aseg.stats", "stats/lh.aparc.stats",
            "stats/rh.aparc.stats", "surf/lh.white", "surf/rh.white")
GPU_QUERY = ("--query-gpu=index,uuid,name,"
             "utilization.gpu,memory.used", "--format=csv,noheader,nounits")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def checked_file(value: str | Path) -> Path:
    path = Path(value).expanduser().resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"Expected a file: {path}")
    return path


def gpu_sample(physical_index: int) -> list[str]:
    result = subprocess.run(("nvidia-smi", "-i", str(physical_index), *GPU_QUERY),
                            capture_output=True, text=True,
                            timeout=10, check=True)
    return next(csv.reader([result.stdout.strip()]))


def cpu_times() -> tuple[int, int]:
    fields = [int(value) for value in Path("/proc/stat").read_text().splitlines()[0].split()[1:]]
    return sum(fields), fields[3] + fields[4]


def run_timed(command: list[str], env: dict[str, str], label: str,
              root: Path, subject_dir: Path, gpu_index: int) -> dict:
    stdout = root / f"{label}.stdout.log"
    resource_file = root / f"{label}.resources.csv"
    time_file = root / f"{label}.time.txt"
    started = utc_now()
    began = time.monotonic()
    with stdout.open("w") as output, resource_file.open("w", newline="") as samples:
        writer = csv.writer(samples)
        writer.writerow(("timestamp_utc", "cpu_utilization_percent", "load_1m",
                         "load_5m", "load_15m", "gpu_index", "gpu_uuid",
                         "gpu_name", "gpu_utilization_percent", "gpu_memory_mib"))
        process = subprocess.Popen(
            ["/usr/bin/time", "-v", "-o", str(time_file), *command],
            env=env, stdout=output, stderr=subprocess.STDOUT,
        )
        previous = cpu_times()
        while True:
            load = os.getloadavg()
            current = cpu_times()
            total = current[0] - previous[0]
            usage = "" if total <= 0 else round(100 * (1 - (current[1] - previous[1]) / total), 2)
            previous = current
            try:
                gpu = gpu_sample(gpu_index)
            except (OSError, subprocess.SubprocessError, StopIteration):
                gpu = [""] * 5
            writer.writerow((utc_now(), usage, *[round(v, 2) for v in load], *gpu))
            samples.flush()
            if process.poll() is not None:
                break
            time.sleep(10)
        code = process.wait()
    missing = [name for name in REQUIRED if not (subject_dir / name).is_file()]
    result = {"label": label, "started_utc": started, "ended_utc": utc_now(),
              "elapsed_seconds": time.monotonic() - began, "exit_code": code,
              "missing_required_outputs": missing, "command": command,
              "stdout_log": str(stdout), "time_log": str(time_file),
              "resource_csv": str(resource_file), "subject_dir": str(subject_dir)}
    (root / f"{label}.result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t1", required=True, type=Path)
    parser.add_argument("--official-home", required=True, type=Path)
    parser.add_argument("--candidate-python", required=True, type=Path)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--license", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path,
                        help="new directory; existing paths are refused")
    parser.add_argument("--order", choices=("A-C", "C-A"), default="A-C",
                        help="serial run order (default: A-C)")
    parser.add_argument("--cuda-visible-devices", type=int, metavar="PHYSICAL_INDEX",
                        help="show only this physical GPU to both runs; C uses logical cuda:0")
    parser.add_argument("--execute", action="store_true",
                        help="run the chosen order; without this flag only print the plan")
    args = parser.parse_args()
    if args.cuda_visible_devices is not None and args.cuda_visible_devices < 0:
        parser.error("--cuda-visible-devices must be a nonnegative physical GPU index")

    t1 = checked_file(args.t1)
    official_home = args.official_home.expanduser().resolve(strict=True)
    official_recon = checked_file(official_home / "bin/recon-all")
    build_stamp = checked_file(official_home / "build-stamp.txt").read_text().strip()
    if "8.2.0" not in build_stamp:
        raise ValueError(f"Expected the official FreeSurfer 8.2.0 build: {build_stamp}")
    # Keep the venv entry path: resolving its symlink can bypass pyvenv.cfg.
    candidate_python = args.candidate_python.expanduser().absolute()
    if not candidate_python.is_file():
        raise ValueError(f"Expected a candidate Python interpreter: {candidate_python}")
    bundle = args.bundle.expanduser().resolve(strict=True)
    checked_file(bundle / "bin/recon-all")
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    license_file = checked_file(args.license)
    root = args.output.expanduser().resolve()
    official_subjects = root / "official_subjects"
    candidate_subjects = root / "candidate_subjects"
    official = [str(official_recon), "-i", str(t1), "-s", "a_official",
                "-sd", str(official_subjects), "-all", "-parallel",
                "-openmp", "4", "-itkthreads", "1"]
    visible_devices = (str(args.cuda_visible_devices)
                       if args.cuda_visible_devices is not None
                       else os.environ.get("CUDA_VISIBLE_DEVICES"))
    candidate_device = "cuda:0" if args.cuda_visible_devices is not None else "cuda:1"
    gpu_index = args.cuda_visible_devices if args.cuda_visible_devices is not None else 1
    candidate = [str(candidate_python), "-I", "-m",
                 "freesurfer_torch.recon_all.standalone", "-i", str(t1),
                 "-s", "c_candidate", "-sd", str(candidate_subjects),
                 "--bundle", str(bundle), "--license", str(license_file),
                 "--device", candidate_device, "--threads", "4"]
    run_order = ("A_official", "C_candidate") if args.order == "A-C" else \
                ("C_candidate", "A_official")
    plan = {"order": args.order, "run_order": list(run_order), "t1": str(t1),
            "t1_sha256": sha256(t1), "hostname": socket.gethostname(),
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "cuda_visible_devices": visible_devices,
            "cuda_visibility_source": ("--cuda-visible-devices" if args.cuda_visible_devices is not None
                                       else "inherited environment"),
            "gpu_mapping": {"official_visible_physical_gpu": args.cuda_visible_devices,
                            "candidate_visible_physical_gpu": args.cuda_visible_devices,
                            "candidate_logical_device": candidate_device,
                            "sampled_physical_gpu": gpu_index},
            "threads": THREADS, "gpu_device": candidate_device, "gpu_smi_index": gpu_index,
            "official_home": str(official_home),
            "official_build_stamp": build_stamp,
            "official_recon_sha256": sha256(official_recon),
            "bundle": str(bundle), "bundle_manifest_sha256":
                sha256(manifest_path) if manifest_path.is_file() else None,
            "bundle_manifest_standalone_verified": manifest.get("standalone_verified") is True,
            "candidate_python": str(candidate_python), "output": str(root),
            "output_exists": root.exists(), "commands": {"A_official": official,
                                                        "C_candidate": candidate}}
    if not args.execute:
        print(json.dumps(plan, indent=2))
        print(f"Dry run only. Pass --execute to create a new output directory and run {args.order}.")
        return
    if root.exists():
        raise FileExistsError(f"Refusing an existing output path: {root}")
    if manifest.get("standalone_verified") is not True:
        raise ValueError("C requires a production bundle with standalone_verified=true")
    loaded_home = os.environ.get("FREESURFER_HOME")
    if not loaded_home or Path(loaded_home).resolve() != official_home:
        raise RuntimeError("Load the official FreeSurfer module matching --official-home first")
    if not os.access(candidate_python, os.X_OK) or not os.access(official_recon, os.X_OK):
        raise PermissionError("Both reconstruction entry points must be executable")
    if shutil.which("nvidia-smi") is None or not Path("/usr/bin/time").is_file():
        raise FileNotFoundError("nvidia-smi and /usr/bin/time are required for the benchmark")
    inventory = gpu_sample(gpu_index)
    probe_env = os.environ.copy()
    if args.cuda_visible_devices is not None:
        probe_env["CUDA_VISIBLE_DEVICES"] = visible_devices
    count = subprocess.run([str(candidate_python), "-I", "-c",
                            "import torch, freesurfer_torch.recon_all.standalone; "
                            "print(torch.cuda.device_count())"],
                           env=probe_env, capture_output=True, text=True,
                           check=True, timeout=120)
    if int(count.stdout.strip()) < (1 if args.cuda_visible_devices is not None else 2):
        raise RuntimeError(f"Logical {candidate_device} is unavailable in the candidate Python environment")

    root.mkdir(parents=True, exist_ok=False)
    official_subjects.mkdir()
    candidate_subjects.mkdir()
    plan.update({"created_utc": utc_now(), "gpu_inventory": inventory,
                 "output_exists": False})
    (root / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")

    official_env = os.environ.copy()
    official_env.update(THREADS)
    official_env.update(SUBJECTS_DIR=str(official_subjects), FS_LICENSE=str(license_file))
    candidate_env = {key: os.environ[key] for key in
                     ("HOME", "TMPDIR", "LANG", "LC_ALL", "CUDA_VISIBLE_DEVICES")
                     if key in os.environ}
    candidate_env.update(THREADS)
    candidate_env["PATH"] = "/usr/bin:/bin"
    candidate_env["PYTHONNOUSERSITE"] = "1"
    if args.cuda_visible_devices is not None:
        official_env["CUDA_VISIBLE_DEVICES"] = visible_devices
        candidate_env["CUDA_VISIBLE_DEVICES"] = visible_devices

    runs = {
        "A_official": (official, official_env, official_subjects / "a_official"),
        "C_candidate": (candidate, candidate_env, candidate_subjects / "c_candidate"),
    }
    results = []
    for label in run_order:
        command, env, subject = runs[label]
        if sha256(t1) != plan["t1_sha256"]:
            raise RuntimeError("T1 content changed between runs")
        result = run_timed(command, env, label, root, subject, gpu_index)
        results.append(result)
        (root / "summary.json").write_text(json.dumps({"plan": plan, "results": results},
                                                      indent=2) + "\n")
        if result["exit_code"] or result["missing_required_outputs"]:
            raise RuntimeError(f"{label} failed; inspect {root / f'{label}.result.json'}")
    print(json.dumps({"output": str(root), "results": results}, indent=2))


if __name__ == "__main__":
    main()
