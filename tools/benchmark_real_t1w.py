"""Fresh-process SynthStrip/SynthMorph benchmark; clinical data stay on the server.

prepare creates a private, immutable manifest from the first valid sorted T1w
files. run reads that manifest and immediately saves one private result per
process. summarize writes a shareable report containing case IDs, never input
paths or clinical directory names. Run after loading the FreeSurfer module.

Examples:
  python tools/benchmark_real_t1w.py prepare --manifest benchmark/manifest.private.json
  python tools/benchmark_real_t1w.py run --manifest benchmark/manifest.private.json \
    --output-dir benchmark/runs --weights weights --function strip --arms candidate_cpu \
    --cpu-affinity 8-15 --threads 8
  python tools/benchmark_real_t1w.py run --manifest benchmark/manifest.private.json \
    --output-dir benchmark/runs --weights weights --function morph \
    --arms reference_gpu candidate_gpu --gpu 0 --cpu-affinity 16-23
  python tools/benchmark_real_t1w.py summarize --manifest benchmark/manifest.private.json \
    --output-dir benchmark/runs --report benchmark/summary.public.json

Only summary.public.json is intended for copying off the server. The manifest,
individual result files, logs, and medical images are private server artifacts.
"""

import argparse
from datetime import datetime, timezone
import glob
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time


ARMS = ("reference_cpu", "reference_gpu", "candidate_cpu", "candidate_gpu")
PAIRS = (("reference_cpu", "candidate_cpu"), ("reference_gpu", "candidate_gpu"),
         ("reference_cpu", "reference_gpu"), ("candidate_cpu", "candidate_gpu"))
WEIGHTS = {"strip": ("synthstrip.1.pt",),
           "morph": ("synthmorph.affine.2.h5", "synthmorph.deform.3.h5")}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False))
    temporary.replace(path)


def image_record(path):
    import numpy as np
    import surfa as sf
    path = Path(path).resolve()
    image = sf.load_volume(str(path))
    affine = image.geom.vox2world.matrix
    if len(image.shape) != 3 or min(image.shape) < 2:
        raise ValueError("not a nonempty single-frame 3D volume")
    if (not np.isfinite(affine).all() or abs(np.linalg.det(affine[:3, :3])) < 1e-8
            or not np.isfinite(image.data).all()):
        raise ValueError("nonfinite voxels/geometry or singular geometry")
    stat = path.stat()
    return {"path": str(path), "sha256": sha256(path), "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns, "shape": list(image.shape),
            "dtype": str(image.dtype), "voxsize": np.asarray(image.geom.voxsize).tolist(),
            "vox2world": affine.tolist()}


def prepare(args):
    if args.count < 1:
        raise ValueError("count must be positive")
    manifest_path = Path(args.manifest)
    if manifest_path.exists():
        raise FileExistsError("manifest already exists; run uses the existing manifest read-only")
    cases, excluded = [], []
    for path in sorted(glob.glob(args.input_glob)):
        try:
            record = image_record(path)
        except Exception as exc:
            excluded.append({"path": path, "reason": f"{type(exc).__name__}: {exc}"})
            continue
        record["case_id"] = f"case{len(cases) + 1:02d}"
        cases.append(record)
        if len(cases) == args.count:
            break
    if len(cases) != args.count:
        raise ValueError(f"found only {len(cases)} valid inputs; requested {args.count}")
    manifest = {"version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
                "selection": "first sorted files with valid finite 3D data and nonsingular finite geometry; no output-based selection",
                "input_glob_private": args.input_glob, "cases": cases, "excluded_private": excluded,
                "template": image_record(args.template)}
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("x") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
    print(json.dumps({"prepared_cases": len(cases), "excluded_inputs": len(excluded)}), flush=True)


def validate_manifest(manifest, cases):
    for record in [manifest["template"], *cases]:
        path = Path(record["path"])
        stat = path.stat()
        if stat.st_size != record["size_bytes"] or stat.st_mtime_ns != record["mtime_ns"]:
            if sha256(path) != record["sha256"]:
                raise RuntimeError("manifest input changed; prepare a new manifest for changed data")


def provenance(args):
    weights = {name: sha256(Path(args.weights) / name) for name in WEIGHTS[args.function]}
    package = Path(__file__).resolve().parents[1] / "src" / "fnit"
    sources = {path.relative_to(package).as_posix(): sha256(path) for path in sorted(package.rglob("*.py"))}
    fs_home = Path(args.freesurfer_home) if args.freesurfer_home else None
    official = None if fs_home is None else fs_home / "python" / "scripts" / f"mri_synth{args.function}"
    return {"weights_sha256": weights, "candidate_sources_sha256": sources,
            "official_script_sha256": sha256(official) if official and official.is_file() else None,
            "candidate_python": str(Path(args.python).resolve()),
            "official_script": str(official) if official else None}


def command_for(args, arm, case, template, outputs, provenance_data):
    gpu = arm.endswith("gpu")
    candidate = arm.startswith("candidate")
    weights = Path(args.weights).resolve()
    if candidate:
        command = [args.python, "-m", "fnit.cli", f"synth{args.function}"]
        command += ["--weights", str(weights), "--device", "cuda:0" if gpu else "cpu", "-j", str(args.threads)]
    elif args.function == "strip" and gpu:
        source = provenance_data["official_script"]
        if not source or not Path(source).is_file():
            raise FileNotFoundError("the unmodified official SynthStrip Python source is required")
        command = [args.python, source, "-g", "-t", str(args.threads), "--model", str(weights / WEIGHTS["strip"][0])]
    else:
        if not args.freesurfer_home:
            raise ValueError("load FreeSurfer or pass --freesurfer-home for reference arms")
        command = [str(Path(args.freesurfer_home) / "bin" / f"mri_synth{args.function}")]
        if args.function == "strip":
            command += ["-t", str(args.threads), "--model", str(weights / WEIGHTS["strip"][0])]
        else:
            command += ["register", "-j", str(args.threads)]
            for name in WEIGHTS["morph"]:
                command += ["-w", str(weights / name)]
        if gpu:
            command.append("-g")
    if args.function == "strip":
        command += ["-i", case["path"], "-o", outputs["image"], "-m", outputs["mask"], "-d", outputs["distance"]]
    else:
        command += ["-m", "joint", "-e", "256", "-n", "7", "-r", "0.5",
                    "-o", outputs["moved"], "-t", outputs["transform"], case["path"], template["path"]]
    if args.cpu_affinity:
        command = ["taskset", "-c", args.cpu_affinity, *command]
    return command


def stop_process(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
    except ProcessLookupError:
        process.wait()


def execute_case(args, arm, case, manifest, provenance_data, *, warmup=False):
    case_id = case["case_id"]
    directory = Path(args.output_dir).resolve() / args.function / arm / (("warmup_" if warmup else "") + case_id)
    directory.mkdir(parents=True, exist_ok=True)
    names = ("image", "mask", "distance") if args.function == "strip" else ("moved", "transform")
    outputs = {name: str(directory / f"{name}.nii.gz") for name in names}
    command = command_for(args, arm, case, manifest["template"], outputs, provenance_data)
    environment = os.environ.copy()
    environment.update({key: str(args.threads) for key in
                        ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                         "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS")})
    environment["CUDA_VISIBLE_DEVICES"] = args.gpu if arm.endswith("gpu") else ""
    environment["NVIDIA_TF32_OVERRIDE"] = "0"
    if args.freesurfer_home:
        environment["FREESURFER_HOME"] = args.freesurfer_home
    if arm.startswith("candidate") or (args.function == "strip" and arm.endswith("gpu")):
        environment.pop("PYTHONPATH", None)
    fingerprint = {"command": command, "input_sha256": case["sha256"], "template_sha256": manifest["template"]["sha256"],
                   "provenance": provenance_data, "gpu_visibility": environment["CUDA_VISIBLE_DEVICES"],
                   "threads": args.threads, "cpu_affinity": args.cpu_affinity, "warmup_runs_requested": args.warmup,
                   "nvidia_tf32_override": environment["NVIDIA_TF32_OVERRIDE"]}
    signature = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()
    result_path, log_path = directory / "result.json", directory / "stdout_stderr.log"
    if result_path.is_file():
        previous = json.loads(result_path.read_text())
        if (previous.get("status") == "success" and previous.get("signature") == signature
                and log_path.is_file() and all(Path(path).is_file() and Path(path).stat().st_size > 0 for path in outputs.values())):
            print(f"resume {args.function} {arm} {case_id}{' warmup' if warmup else ''}", flush=True)
            return previous
    for path in outputs.values():
        Path(path).unlink(missing_ok=True)
    mode = "official_source_cuda" if args.function == "strip" and arm == "reference_gpu" else (
        "official_command" if arm.startswith("reference") else "standalone_pytorch")
    result = {"case_id": case_id, "function": args.function, "arm": arm, "implementation": mode,
              "warmup_discarded": warmup, "status": "running", "signature": signature,
              "command_private": command, "provenance_private": provenance_data,
              "threads": args.threads, "cpu_affinity": args.cpu_affinity,
              "warmup_runs_requested": args.warmup,
              "nvidia_tf32_override": environment["NVIDIA_TF32_OVERRIDE"],
              "cuda_visible_devices": environment["CUDA_VISIBLE_DEVICES"], "outputs_private": outputs,
              "started_at": time.time(), "returncode": None, "wall_seconds": None, "max_rss_kib": None}
    write_json(result_path, result)
    timed_command = command
    resource_path = directory / "max_rss_kib.txt"
    resource_path.unlink(missing_ok=True)
    if Path("/usr/bin/time").is_file():
        timed_command = ["/usr/bin/time", "-f", "%M", "-o", str(resource_path), *command]
    print(f"start {args.function} {arm} {case_id}{' warmup' if warmup else ''}", flush=True)
    with log_path.open("w") as log:
        started = time.perf_counter()
        process = None
        try:
            process = subprocess.Popen(timed_command, stdout=log, stderr=subprocess.STDOUT,
                                       env=environment, start_new_session=True)
            result["process_group_pid"] = process.pid
            result["returncode"] = process.wait(timeout=args.timeout)
            result["status"] = "success" if result["returncode"] == 0 else "failed"
        except subprocess.TimeoutExpired:
            stop_process(process)
            result.update(status="timed_out", returncode=process.returncode)
        except KeyboardInterrupt:
            if process is not None:
                stop_process(process)
            result.update(status="interrupted", returncode=process.returncode if process else None)
        except OSError as exc:
            result.update(status="launch_failed", error_private=f"{type(exc).__name__}: {exc}")
            log.write(result["error_private"] + "\n")
        finally:
            result["wall_seconds"] = time.perf_counter() - started
            result["finished_at"] = time.time()
    if result["status"] == "success" and not all(Path(path).is_file() and Path(path).stat().st_size > 0 for path in outputs.values()):
        result["status"] = "missing_outputs"
    if resource_path.is_file():
        try:
            result["max_rss_kib"] = int(resource_path.read_text().splitlines()[-1])
        except (ValueError, IndexError):
            pass
    write_json(result_path, result)
    print(f"finish {args.function} {arm} {case_id}: {result['status']} {result['wall_seconds']:.3f}s", flush=True)
    if result["status"] == "interrupted":
        raise KeyboardInterrupt
    return result


def run(args):
    if args.threads < 1 or args.start < 0 or (args.limit is not None and args.limit < 1):
        raise ValueError("threads/limit must be positive; start is zero-based and nonnegative")
    if args.cpu_affinity and (not re.fullmatch(r"[0-9,-]+", args.cpu_affinity) or not shutil.which("taskset")):
        raise ValueError("--cpu-affinity requires taskset and a CPU list such as 0-7")
    manifest = json.loads(Path(args.manifest).read_text())
    cases = manifest["cases"][args.start:None if args.limit is None else args.start + args.limit]
    if not cases:
        raise ValueError("selected case range is empty")
    validate_manifest(manifest, cases)
    provenance_data = provenance(args)
    failed = 0
    for arm in args.arms:
        if args.warmup:
            execute_case(args, arm, cases[0], manifest, provenance_data, warmup=True)
        for case in cases:
            result = execute_case(args, arm, case, manifest, provenance_data)
            failed += result["status"] != "success"
    return int(failed > 0)


def numeric_difference(first, second):
    import numpy as np
    if first.shape != second.shape:
        return {"shape_equal": False}
    difference = first.astype(np.float64) - second.astype(np.float64)
    absolute = np.abs(difference)
    rmse = float(np.sqrt(np.mean(difference * difference)))
    norm = float(np.sqrt(np.mean(first.astype(np.float64) ** 2)))
    return {"shape_equal": True, "max_abs_error": float(absolute.max()),
            "mean_abs_error": float(absolute.mean()), "rmse": rmse,
            "nrmse_reference_rms": rmse / max(norm, 1e-12),
            "exactly_equal": bool(np.array_equal(first, second))}


def geometry_difference(first, second):
    import numpy as np
    return {"shape_equal": bool(np.array_equal(first.shape, second.shape)),
            "vox2world_max_abs_error": float(np.max(np.abs(first.vox2world.matrix - second.vox2world.matrix)))}


def warp_jacobian(warp):
    import numpy as np
    import surfa as sf
    displacement = warp.convert(format=sf.Warp.Format.disp_ras).data.astype(np.float32)
    gradient = np.empty(displacement.shape[:3] + (3, 3), dtype=np.float32)
    for component in range(3):
        for axis, derivative in enumerate(np.gradient(displacement[..., component])):
            gradient[..., component, axis] = derivative
    derivative_world = gradient @ np.linalg.inv(warp.target.vox2world.matrix[:3, :3])
    determinant = np.linalg.det(derivative_world + np.eye(3))
    return {"fraction_nonpositive": float(np.mean(determinant <= 0)),
            "minimum": float(determinant.min()), "median": float(np.median(determinant)),
            "finite_fraction": float(np.mean(np.isfinite(determinant))),
            "domain": "all forward-warp target-grid voxels; physical RAS derivative"}


def compare_outputs(function, first, second):
    import numpy as np
    import surfa as sf
    if function == "strip":
        outputs = {}
        for name in ("image", "mask", "distance"):
            outputs[name] = numeric_difference(first[name].data, second[name].data)
            outputs[name]["geometry"] = geometry_difference(first[name].geom, second[name].geom)
        if first["mask"].shape == second["mask"].shape:
            a, b = first["mask"].data != 0, second["mask"].data != 0
            total = int(a.sum()) + int(b.sum())
            outputs["mask"]["dice"] = float(2 * np.count_nonzero(a & b) / total) if total else 1.0
            outputs["mask"]["disagreeing_voxels"] = int(np.count_nonzero(a != b))
        return outputs
    a = first["transform"].convert(format=sf.Warp.Format.disp_ras)
    b = second["transform"].convert(format=sf.Warp.Format.disp_ras)
    warp = numeric_difference(a.data, b.data)
    warp["units"] = "RAS displacement millimetres"
    warp["source_geometry"] = geometry_difference(a.source, b.source)
    warp["target_geometry"] = geometry_difference(a.target, b.target)
    if a.shape == b.shape:
        norm = np.linalg.norm(a.data.astype(np.float64) - b.data.astype(np.float64), axis=-1)
        warp.update(vector_error_mean_mm=float(norm.mean()), vector_error_p95_mm=float(np.percentile(norm, 95)),
                    vector_error_max_mm=float(norm.max()))
    moved = numeric_difference(first["moved"].data, second["moved"].data)
    moved["geometry"] = geometry_difference(first["moved"].geom, second["moved"].geom)
    return {"moved": moved, "transform": warp}


def summarize(args):
    import numpy as np
    import surfa as sf
    manifest = json.loads(Path(args.manifest).read_text())
    report = {"case_count": len(manifest["cases"]), "case_ids": [case["case_id"] for case in manifest["cases"]],
              "selection": manifest["selection"], "excluded_input_count": len(manifest["excluded_private"]),
              "timing_scope": "fresh command process including Python/framework startup, weight/input loading, inference and output saving; quality analysis excluded",
              "timing_limitations": "filesystem caches are not reset; runs may share host resources; these are descriptive elapsed times, not a controlled cold-cache speedup experiment",
              "registration_settings": {"model": "joint", "extent": 256, "steps": 7, "hyper": 0.5},
              "reference_strip_gpu": "official_source_cuda: unchanged official script executed with the project CUDA-enabled Python; not the bundled FreeSurfer command runtime",
              "image_nrmse_definition": "RMSE(candidate-reference) / max(RMS(reference), 1e-12)",
              "functions": {}}
    all_complete = True
    for function in args.function:
        timing = {arm: [] for arm in args.arms}
        source_records = {arm: [] for arm in args.arms}
        entries = []
        for case in manifest["cases"]:
            case_id = case["case_id"]
            entry, loaded = {"case_id": case_id, "runs": {}, "comparisons": {}}, {}
            for arm in args.arms:
                path = Path(args.output_dir) / function / arm / case_id / "result.json"
                if not path.is_file():
                    entry["runs"][arm] = {"status": "not_run"}
                    all_complete = False
                    continue
                record = json.loads(path.read_text())
                source = {key: record["provenance_private"].get(key) for key in
                          ("weights_sha256", "candidate_sources_sha256", "official_script_sha256")}
                if source not in source_records[arm]:
                    source_records[arm].append(source)
                public = {key: record.get(key) for key in ("status", "implementation", "returncode", "wall_seconds", "max_rss_kib",
                                                          "threads", "cpu_affinity", "cuda_visible_devices", "warmup_runs_requested", "nvidia_tf32_override")}
                entry["runs"][arm] = public
                if record["status"] != "success":
                    all_complete = False
                    continue
                timing[arm].append(record["wall_seconds"])
                try:
                    loaded[arm] = {name: (sf.load_warp(path) if name == "transform" else sf.load_volume(path))
                                   for name, path in record["outputs_private"].items()}
                    if not all(np.isfinite(volume.data).all() for volume in loaded[arm].values()):
                        raise ValueError("nonfinite output")
                    if function == "morph":
                        public["forward_warp_jacobian"] = warp_jacobian(loaded[arm]["transform"])
                except Exception as exc:
                    public["quality_error"] = type(exc).__name__
                    loaded.pop(arm, None)
                    all_complete = False
            for first, second in PAIRS:
                if first not in args.arms or second not in args.arms:
                    continue
                key = f"{first}__{second}"
                if first in loaded and second in loaded:
                    try:
                        entry["comparisons"][key] = {"available": True, "outputs": compare_outputs(function, loaded[first], loaded[second])}
                    except Exception as exc:
                        entry["comparisons"][key] = {"available": False, "error_type": type(exc).__name__}
                        all_complete = False
                else:
                    entry["comparisons"][key] = {"available": False}
            entries.append(entry)
        aggregate = {}
        for arm, values in timing.items():
            aggregate[arm] = {"successful_cases": len(values), "requested_cases": len(manifest["cases"])}
            if values:
                aggregate[arm].update(median_seconds=float(np.median(values)),
                                      q25_seconds=float(np.percentile(values, 25)), q75_seconds=float(np.percentile(values, 75)),
                                      mean_seconds=float(np.mean(values)), min_seconds=min(values), max_seconds=max(values),
                                      total_seconds=sum(values))
        report["functions"][function] = {"time_summary": aggregate, "source_records": source_records, "cases": entries}
    report["all_requested_runs_and_comparisons_available"] = all_complete
    write_json(args.report, report)
    print(json.dumps({"case_count": report["case_count"], "complete": all_complete,
                      "time_summary": {name: data["time_summary"] for name, data in report["functions"].items()}}, indent=2), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    modes = parser.add_subparsers(dest="mode", required=True)
    select = modes.add_parser("prepare", help="create a private manifest without running either model")
    select.add_argument("--manifest", required=True)
    select.add_argument("--input-glob", required=True)
    select.add_argument("--template", required=True)
    select.add_argument("--count", type=int, default=12)
    execute = modes.add_parser("run", help="run or resume fresh-process inference from the private manifest")
    execute.add_argument("--manifest", required=True)
    execute.add_argument("--output-dir", required=True)
    execute.add_argument("--weights", required=True)
    execute.add_argument("--function", choices=("strip", "morph"), required=True)
    execute.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    execute.add_argument("--threads", type=int, default=8)
    execute.add_argument("--cpu-affinity", help="taskset CPU list, e.g. 0-7")
    execute.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES value; subprocess uses cuda:0")
    execute.add_argument("--start", type=int, default=0, help="zero-based offset in the immutable manifest")
    execute.add_argument("--limit", "--count", dest="limit", type=int)
    execute.add_argument("--warmup", type=int, choices=(0, 1), default=0, help="one discarded fresh-process first-case run per arm")
    execute.add_argument("--timeout", type=float, help="per-process seconds; default unlimited")
    execute.add_argument("--python", default=sys.executable, help="project venv Python for candidate and official_source_cuda")
    execute.add_argument("--freesurfer-home", default=os.environ.get("FREESURFER_HOME"))
    collect = modes.add_parser("summarize", help="write case-ID-only timing and correctness results")
    collect.add_argument("--manifest", required=True)
    collect.add_argument("--output-dir", required=True)
    collect.add_argument("--report", required=True)
    collect.add_argument("--function", nargs="+", choices=("strip", "morph"), default=["strip", "morph"])
    collect.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args)
        return 0
    return run(args) if args.mode == "run" else summarize(args)


if __name__ == "__main__":
    raise SystemExit(main())
