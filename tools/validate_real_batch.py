"""Validate 24 real-image jobs in persistent workers, then compare saved outputs.

Run on the server; no clinical input is copied. Example:
  CUDA_VISIBLE_DEVICES=1 taskset -c 32-39 python tools/validate_real_batch.py run \
    --manifest benchmark/manifest.private.json --weights weights \
    --out-dir benchmark/real_batch
  python tools/validate_real_batch.py compare --out-dir benchmark/real_batch \
    --baseline-dir benchmark/runs

The run uses cuda:0 within the supplied CUDA_VISIBLE_DEVICES mapping, two
workers, four CPU threads per worker, and one persistent BatchRunner. The two
calls process SynthStrip and joint SynthMorph, respectively. Compare can be
rerun as candidate_gpu baseline cases finish. Exit codes: 0 = passed, 1 =
failure, 2 = pending baseline. Only *.public.json files may be shared; medical
outputs and execution.private.json remain server-private.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import re
import time
import traceback


OUTPUT_NAMES = {"strip": ("image", "mask", "distance"), "morph": ("moved", "transform")}
TIMING_NOTE = (
    "GPU 1 is shared with other work in the intended deployment. Device cuda:0 is relative to "
    "CUDA_VISIBLE_DEVICES. Durations are descriptive and are not an exclusive-hardware timing "
    "comparison with the baseline. Job intervals include model loading, inference and saving; "
    "overlap establishes concurrent independent processes, not simultaneous GPU kernel execution."
)


def write_json(path, data):
    path = Path(path)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False))
    temporary.replace(path)


def error_type(error):
    match = re.match(r"([A-Za-z_][A-Za-z_0-9]*):", error or "")
    return match.group(1) if match else ("WorkerFailure" if error else None)


def public_job(item):
    result = item["result"]
    start, end = result.get("started_at"), result.get("finished_at")
    return {"case_id": item["case_id"], "task": item["task"], "pid": result.get("pid"),
            "device": result.get("device"), "started_at": start, "finished_at": end,
            "elapsed_seconds": end - start if start is not None and end is not None else None,
            "status": "failure" if result.get("error") else "success",
            "error_type": error_type(result.get("error"))}


def overlaps(jobs):
    found = []
    for index, first in enumerate(jobs):
        if first["status"] != "success" or first["started_at"] is None or first["finished_at"] is None:
            continue
        for second in jobs[index + 1:]:
            if (second["status"] != "success" or first["pid"] == second["pid"]
                    or second["started_at"] is None or second["finished_at"] is None):
                continue
            duration = min(first["finished_at"], second["finished_at"]) - max(first["started_at"], second["started_at"])
            if duration > 0:
                found.append({"first": {"case_id": first["case_id"], "task": first["task"], "pid": first["pid"]},
                              "second": {"case_id": second["case_id"], "task": second["task"], "pid": second["pid"]},
                              "overlap_seconds": duration})
    return found


def public_execution(private):
    jobs = [public_job(item) for item in private["jobs"]]
    pids = {task: sorted({job["pid"] for job in jobs if job["task"] == task and job["pid"] is not None})
            for task in OUTPUT_NAMES}
    report = {"phase": private["phase"], "expected_jobs": 24, "recorded_jobs": len(jobs),
              "workers_per_device": 2, "threads_per_worker": 4, "devices": ["cuda:0"],
              "cuda_visible_devices": private["cuda_visible_devices"], "cpu_affinity": private["cpu_affinity"],
              "nvidia_tf32_override": "0",
              "timing_note": TIMING_NOTE, "task_wall_seconds": private["task_wall_seconds"],
              "wall_seconds": private.get("wall_seconds"), "jobs": jobs,
              "worker_pids_by_task": pids, "same_worker_processes_across_calls": bool(pids["strip"] and pids["strip"] == pids["morph"]),
              "independent_process_overlaps": overlaps(jobs),
              "errors": [{"task": item["task"], "error_type": item["error_type"]} for item in private.get("errors", [])]}
    report["execution_passed"] = (private["phase"] == "complete" and len(jobs) == 24
                                  and all(job["status"] == "success" for job in jobs)
                                  and len(pids["strip"]) == len(pids["morph"]) == 2
                                  and report["same_worker_processes_across_calls"]
                                  and bool(report["independent_process_overlaps"]) and not report["errors"])
    return report


def build_jobs(manifest, root, weights, task):
    jobs = []
    for case in manifest["cases"]:
        output = root / task / case["case_id"]
        model = {"weights": str(weights)}
        if task == "morph":
            model.update(model="joint", extent=256, steps=7, hyper=.5)
        jobs.append({"task": "synthstrip" if task == "strip" else "synthmorph", "model": model,
                     "kwargs": {"image": case["path"]} if task == "strip" else {
                         "moving": case["path"], "fixed": manifest["template"]["path"]},
                     "outputs": {name: str(output / f"{name}.nii.gz") for name in OUTPUT_NAMES[task]}})
    return jobs


def run(args):
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    from freesurfer_torch import BatchRunner
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text())
    cases = manifest["cases"]
    if (len(cases) != 12 or len({case["case_id"] for case in cases}) != 12
            or any(not re.fullmatch(r"case[0-9]+", case["case_id"]) for case in cases)):
        raise ValueError("expected the fixed twelve-case public-ID manifest")
    for record in [manifest["template"], *cases]:
        if not Path(record["path"]).is_file():
            raise FileNotFoundError("a server-private manifest input is missing")
    root, weights = Path(args.out_dir).resolve(), Path(args.weights).resolve()
    for name in ("synthstrip.1.pt", "synthmorph.affine.2.h5", "synthmorph.deform.3.h5"):
        if not (weights / name).is_file():
            raise FileNotFoundError(f"missing model weight: {name}")
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ValueError("out-dir must be empty for a new run; use compare for existing outputs")
    private = {"phase": "starting", "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
               "case_ids": [case["case_id"] for case in cases], "jobs": [], "errors": [], "task_wall_seconds": {},
               "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "not_set"),
               "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None}
    started = time.perf_counter()
    with BatchRunner(devices=["cuda:0"], workers_per_device=2, threads_per_worker=4) as runner:
        for task in OUTPUT_NAMES:
            private["phase"] = f"{task}_running"
            write_json(root / "execution.private.json", private)
            write_json(root / "execution.public.json", public_execution(private))
            print(f"start {task}: 12 jobs", flush=True)
            task_started = time.perf_counter()
            try:
                results = runner.run(build_jobs(manifest, root, weights, task))
                for case, result in zip(cases, results):
                    private["jobs"].append({"case_id": case["case_id"], "task": task, "result": asdict(result)})
            except Exception as exc:
                private["errors"].append({"task": task, "error_type": type(exc).__name__, "traceback_private": traceback.format_exc()})
            private["task_wall_seconds"][task] = time.perf_counter() - task_started
            write_json(root / "execution.private.json", private)
            write_json(root / "execution.public.json", public_execution(private))
            print(f"finish {task}: {private['task_wall_seconds'][task]:.3f}s", flush=True)
    private["phase"] = "complete"
    private["wall_seconds"] = time.perf_counter() - started
    report = public_execution(private)
    write_json(root / "execution.private.json", private)
    write_json(root / "execution.public.json", report)
    print(json.dumps({"execution_passed": report["execution_passed"], "recorded_jobs": len(private["jobs"]),
                      "worker_pids_by_task": report["worker_pids_by_task"], "wall_seconds": report["wall_seconds"]}), flush=True)
    return 0 if report["execution_passed"] else 1


def geometry(reference, candidate):
    import numpy as np
    first, second = reference.vox2world.matrix, candidate.vox2world.matrix
    finite = bool(np.isfinite(first).all() and np.isfinite(second).all())
    shape_equal = bool(np.array_equal(reference.shape, candidate.shape))
    return {"shape_equal": shape_equal, "finite": finite,
            "vox2world_max_abs_error": float(np.max(np.abs(first - second))) if finite else None,
            "passed": bool(shape_equal and finite and np.allclose(first, second, atol=1e-5, rtol=0))}


def compare_file(reference_path, candidate_path, name):
    import numpy as np
    import surfa as sf
    warp = name == "transform"
    loader = sf.load_warp if warp else sf.load_volume
    first, second = loader(reference_path), loader(candidate_path)
    if warp:
        first = first.convert(format=sf.Warp.Format.disp_ras)
        second = second.convert(format=sf.Warp.Format.disp_ras)
        geometries = {"source": geometry(first.source, second.source), "target": geometry(first.target, second.target)}
    else:
        geometries = {"image": geometry(first.geom, second.geom)}
    a, b = first.data, second.data
    metrics = {"reference_shape": list(a.shape), "batch_shape": list(b.shape), "shape_equal": a.shape == b.shape,
               "units": "RAS displacement mm" if warp else ("mm" if name == "distance" else ("binary" if name == "mask" else "input intensity")),
               "geometry": geometries, "passed": False}
    if a.shape != b.shape:
        metrics["error_type"] = "ShapeMismatch"
        return metrics
    metrics["finite"] = bool(np.isfinite(a).all() and np.isfinite(b).all())
    if not metrics["finite"]:
        metrics["error_type"] = "NonfiniteOutput"
        return metrics
    difference = a.astype(np.float64) - b.astype(np.float64)
    absolute = np.abs(difference)
    metrics.update(max_abs_error=float(absolute.max()), mean_abs_error=float(absolute.mean()),
                   rmse=float(np.sqrt(np.mean(difference * difference))),
                   exactly_equal=bool(np.array_equal(a, b)), allclose=bool(np.allclose(a, b, atol=1e-4, rtol=1e-5)))
    if name == "mask":
        metrics["disagreeing_voxels"] = int(np.count_nonzero(a != b))
    metrics["passed"] = all(value["passed"] for value in geometries.values()) and (
        metrics["exactly_equal"] if name == "mask" else metrics["allclose"])
    return metrics


def compare(args):
    root = Path(args.out_dir).resolve()
    private = json.loads((root / "execution.private.json").read_text())
    execution = public_execution(private)
    observed = {(item["task"], item["case_id"]): item for item in private["jobs"]}
    rows = []
    for task in OUTPUT_NAMES:
        for case_id in private["case_ids"]:
            batch = observed.get((task, case_id))
            row = {"case_id": case_id, "task": task, "status": "pending", "error_type": None, "metrics": {}}
            if batch is None:
                row["error_type"] = "BatchResultPending" if private["phase"] != "complete" else "MissingBatchResult"
                if private["phase"] == "complete":
                    row["status"] = "failure"
                rows.append(row)
                continue
            row.update(public_job(batch))
            row["status"] = "pending"
            if batch["result"].get("error"):
                row.update(status="failure", error_type=error_type(batch["result"]["error"]))
                rows.append(row)
                continue
            baseline_path = Path(args.baseline_dir) / task / "candidate_gpu" / case_id / "result.json"
            if not baseline_path.is_file():
                row["error_type"] = "BaselinePending"
                rows.append(row)
                continue
            baseline = json.loads(baseline_path.read_text())
            state = baseline.get("status")
            if state != "success":
                row.update(status="pending" if state in (None, "not_run", "running") else "failure",
                           error_type="BaselinePending" if state in (None, "not_run", "running") else "BaselineFailure")
                rows.append(row)
                continue
            row["baseline_wall_seconds"] = baseline.get("wall_seconds")
            for name in OUTPUT_NAMES[task]:
                try:
                    row["metrics"][name] = compare_file(baseline["outputs_private"][name], batch["result"]["outputs"][name], name)
                except Exception as exc:
                    row["metrics"][name] = {"passed": False, "error_type": type(exc).__name__}
            row["status"] = "success" if all(value["passed"] for value in row["metrics"].values()) else "failure"
            if row["status"] == "failure":
                row["error_type"] = "OutputComparisonFailure"
            rows.append(row)
    counts = {state: sum(row["status"] == state for row in rows) for state in ("success", "pending", "failure")}
    report = {"reference": "candidate_gpu fresh-process outputs on benchmark GPU 0",
              "cuda_visible_devices": private["cuda_visible_devices"], "timing_note": TIMING_NOTE,
              "atol": 1e-4, "rtol": 1e-5, "geometry_atol": 1e-5, "mask_requires_exact_equality": True,
              "expected_jobs": 24, "counts": counts, "execution_passed": execution["execution_passed"],
              "jobs": rows, "passed": counts["success"] == 24 and execution["execution_passed"]}
    write_json(root / "comparison.public.json", report)
    print(json.dumps({"passed": report["passed"], "counts": counts}), flush=True)
    execution_failed = private["phase"] == "complete" and not execution["execution_passed"]
    return 0 if report["passed"] else (1 if counts["failure"] or execution_failed else 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    start = commands.add_parser("run", help="execute 12 strip and 12 joint-256 registration jobs")
    start.add_argument("--manifest", required=True)
    start.add_argument("--weights", required=True)
    start.add_argument("--out-dir", required=True)
    check = commands.add_parser("compare", help="compare saved batch outputs against candidate_gpu baseline outputs")
    check.add_argument("--out-dir", required=True)
    check.add_argument("--baseline-dir", required=True)
    args = parser.parse_args()
    return run(args) if args.command == "run" else compare(args)


if __name__ == "__main__":
    raise SystemExit(main())
