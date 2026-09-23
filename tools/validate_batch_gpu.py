"""Compare serial inference with independent GPU workers on four real jobs.

Example, two processes on one GPU:
    python tools/validate_batch_gpu.py --image template.nii.gz --weights weights \
        --out-dir validation/batch_same_gpu --devices cuda:0 --workers-per-device 2

Example, one process on each of two GPUs:
    python tools/validate_batch_gpu.py --image template.nii.gz --weights weights \
        --out-dir validation/batch_two_gpu --devices cuda:0 cuda:1 --workers-per-device 1

The two strip jobs use --image. The two registration jobs use --moving and
--fixed, both defaulting to --image. Inputs are never changed by this script.
"""

import argparse
from dataclasses import asdict
import gc
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
import surfa as sf
import torch

from freesurfer_torch import BatchResult, BatchRunner, SynthMorph, SynthStrip


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_jobs(root, image, moving, fixed, weights):
    jobs = []
    for index in range(4):
        strip = index < 2
        directory = root / f"{index}_{'synthstrip' if strip else 'synthmorph'}"
        directory.mkdir(parents=True, exist_ok=True)
        names = ("image", "mask", "distance") if strip else ("moved", "fixed_moved", "transform", "inverse")
        jobs.append({
            "task": "synthstrip" if strip else "synthmorph",
            "model": {"weights": str(weights)} if strip else {
                "weights": str(weights), "model": "deform", "extent": 192,
            },
            "kwargs": {"image": str(image)} if strip else {"moving": str(moving), "fixed": str(fixed)},
            "outputs": {name: str(directory / f"{name}.nii.gz") for name in names},
        })
    return jobs


def run_serial(jobs, device):
    models, outcomes = {}, []
    for index, job in enumerate(jobs):
        outcome = BatchResult(index, job["task"], device, pid=os.getpid(), started_at=time.time())
        try:
            key = (job["task"], json.dumps(job["model"], sort_keys=True))
            if key not in models:
                constructor = SynthStrip if job["task"] == "synthstrip" else SynthMorph
                models[key] = constructor(device=device, **job["model"])
            result = models[key](**job["kwargs"])
            for name, path in job["outputs"].items():
                getattr(result, name).save(path)
                outcome.outputs[name] = path
        except Exception as exc:
            outcome.error = f"{type(exc).__name__}: {exc}"
            outcome.traceback = traceback.format_exc()
        finally:
            torch.cuda.synchronize(device)
            outcome.finished_at = time.time()
        outcomes.append(outcome)
        print(f"serial {index} {job['task']}: {'ok' if outcome.ok else outcome.error}", flush=True)
    return outcomes


def compare(reference_path, candidate_path, *, warp, mask, atol, rtol):
    loader = sf.load_warp if warp else sf.load_volume
    reference, candidate = loader(reference_path), loader(candidate_path)
    first, second = reference.data, candidate.data
    result = {"reference": reference_path, "candidate": candidate_path,
              "reference_shape": list(first.shape), "candidate_shape": list(second.shape),
              "shape_equal": first.shape == second.shape}
    if not result["shape_equal"]:
        result["passed"] = False
        return result
    difference = np.abs(first.astype(np.float64) - second.astype(np.float64))
    result.update({
        "finite": bool(np.isfinite(first).all() and np.isfinite(second).all()),
        "max_abs_error": float(difference.max()),
        "mean_abs_error": float(difference.mean()),
        "rmse": float(np.sqrt(np.mean(difference * difference))),
        "unequal_voxels": int(np.count_nonzero(first != second)),
        "bitwise_equal": bool(np.array_equal(first, second)),
        "allclose": bool(np.allclose(first, second, atol=atol, rtol=rtol)),
    })
    if warp:
        geometries = (("source", reference.source, candidate.source),
                      ("target", reference.target, candidate.target))
        result["warp_format_equal"] = bool(reference.format == candidate.format)
    else:
        geometries = (("image", reference.geom, candidate.geom),)
    result["geometry"] = {}
    for name, ref_geom, cand_geom in geometries:
        result["geometry"][name] = {
            "shape_equal": bool(np.array_equal(ref_geom.shape, cand_geom.shape)),
            "affine_max_abs_error": float(np.max(np.abs(
                ref_geom.vox2world.matrix - cand_geom.vox2world.matrix))),
        }
    geometry_ok = all(item["shape_equal"] and item["affine_max_abs_error"] <= 1e-5
                      for item in result["geometry"].values())
    if mask:
        ref_mask, cand_mask = first != 0, second != 0
        total = int(ref_mask.sum()) + int(cand_mask.sum())
        result["dice"] = float(2 * np.count_nonzero(ref_mask & cand_mask) / total) if total else 1.0
    result["passed"] = bool(result["finite"] and geometry_ok
                            and result.get("warp_format_equal", True)
                            and (result["bitwise_equal"] if mask else result["allclose"]))
    return result


def overlapping_jobs(outcomes):
    overlaps = []
    for index, first in enumerate(outcomes):
        if not first.ok or first.started_at is None or first.finished_at is None:
            continue
        for second in outcomes[index + 1:]:
            if not second.ok or first.pid == second.pid or second.started_at is None or second.finished_at is None:
                continue
            duration = min(first.finished_at, second.finished_at) - max(first.started_at, second.started_at)
            if duration > 0:
                overlaps.append({"indices": [first.index, second.index], "pids": [first.pid, second.pid],
                                 "devices": [first.device, second.device], "overlap_seconds": duration})
    return overlaps


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", required=True)
    parser.add_argument("--moving")
    parser.add_argument("--fixed")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--devices", nargs="+", default=["cuda:0"])
    parser.add_argument("--workers-per-device", type=int, default=2)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-5)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("real CUDA hardware is required")
    if args.workers_per_device < 1 or len(args.devices) * args.workers_per_device < 2:
        parser.error("use at least two workers in total to validate parallel dispatch")
    for device in args.devices:
        if not device.startswith("cuda:"):
            parser.error("devices must use cuda:<index>")
        torch.cuda.get_device_properties(device)
    image = Path(args.image).resolve()
    moving, fixed = Path(args.moving or image).resolve(), Path(args.fixed or image).resolve()
    weights, root = Path(args.weights).resolve(), Path(args.out_dir).resolve()
    for path in (image, moving, fixed, weights / "synthstrip.1.pt", weights / "synthmorph.deform.3.h5"):
        if not path.is_file():
            parser.error(f"missing input or weights: {path}")
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        parser.error("out-dir must be empty to keep this validation independent")
    torch.set_num_threads(args.threads)
    torch.cuda.set_device(args.devices[0])
    report = {
        "test": "serial versus spawned GPU batch workers",
        "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
        "devices": [{"device": device, "name": torch.cuda.get_device_name(device)} for device in args.devices],
        "workers_per_device": args.workers_per_device, "threads_per_worker": args.threads,
        "atol": args.atol, "rtol": args.rtol, "mask_requires_exact_equality": True,
        "inputs": {name: {"path": str(path), "sha256": sha256(path)}
                   for name, path in (("image", image), ("moving", moving), ("fixed", fixed))},
        "weights": {name: sha256(weights / name) for name in ("synthstrip.1.pt", "synthmorph.deform.3.h5")},
        "timing_scope": "Job intervals include model loading, inference and file saving. "
                        "Overlap demonstrates concurrent independent processes, not kernel-level GPU overlap. "
                        "Wall-clock measurements are descriptive and are not a speedup benchmark.",
        "phase": "serial_running",
    }
    report_path = root / "report.json"
    report_path.write_text(json.dumps(report, indent=2))
    serial_jobs = make_jobs(root / "serial", image, moving, fixed, weights)
    started = time.perf_counter()
    serial = run_serial(serial_jobs, args.devices[0])
    report["serial_wall_seconds"] = time.perf_counter() - started
    report["serial"] = [asdict(item) for item in serial]
    # Release all serial models before allocating separate worker copies.
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(args.devices[0])
    report["phase"] = "batch_running"
    report_path.write_text(json.dumps(report, indent=2))
    batch_jobs = make_jobs(root / "batch", image, moving, fixed, weights)
    started = time.perf_counter()
    with BatchRunner(args.devices, args.workers_per_device, args.threads) as runner:
        parallel = runner.run(batch_jobs)
    report["batch_wall_seconds"] = time.perf_counter() - started
    report["batch"] = [asdict(item) for item in parallel]
    report["independent_process_overlaps"] = overlapping_jobs(parallel)
    report["observed_worker_pids"] = sorted({item.pid for item in parallel if item.pid is not None})
    report["observed_devices"] = sorted({item.device for item in parallel if item.device is not None})
    report["comparisons"] = []
    for baseline, candidate in zip(serial, parallel):
        item = {"index": candidate.index, "task": candidate.task,
                "serial_error": baseline.error, "batch_error": candidate.error, "outputs": {}}
        if baseline.ok and candidate.ok:
            for name in baseline.outputs:
                try:
                    item["outputs"][name] = compare(
                        baseline.outputs[name], candidate.outputs[name],
                        warp=name in ("transform", "inverse"), mask=name == "mask",
                        atol=args.atol, rtol=args.rtol,
                    )
                except Exception as exc:
                    item["outputs"][name] = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}
        item["passed"] = bool(item["outputs"] and all(v["passed"] for v in item["outputs"].values()))
        report["comparisons"].append(item)
    report["outputs_passed"] = all(item["passed"] for item in report["comparisons"])
    report["parallel_processes_observed"] = bool(report["independent_process_overlaps"])
    report["all_requested_devices_observed"] = set(report["observed_devices"]) == set(args.devices)
    report["passed"] = (report["outputs_passed"] and report["parallel_processes_observed"]
                        and report["all_requested_devices_observed"])
    report["phase"] = "complete"
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps({"report": str(report_path), "passed": report["passed"],
                      "outputs_passed": report["outputs_passed"],
                      "parallel_processes_observed": report["parallel_processes_observed"],
                      "worker_pids": report["observed_worker_pids"],
                      "serial_wall_seconds": report["serial_wall_seconds"],
                      "batch_wall_seconds": report["batch_wall_seconds"]}, indent=2), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
