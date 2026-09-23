"""Spawned GPU workers that reuse models and return ordered, per-job results.

Each job has ``task``, ``kwargs``, ``outputs``, and optional ``model`` keys.
``model`` contains constructor arguments other than ``device``; ``kwargs``
contains arguments to the model's call. Outputs map result attribute names to
file paths. Arrays stay in the worker: only paths and errors cross processes.
"""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
import json
import multiprocessing as mp
import os
from pathlib import Path
import re
import time
import traceback


_OUTPUTS = {
    "synthstrip": {"image", "mask", "distance"},
    "synthmorph": {"moved", "fixed_moved", "transform", "inverse"},
    "wmh_synthseg": {"segmentation", "lesion_probability"},
}
_models = {}
_device = None
_initialization_error = None


@dataclass
class BatchResult:
    """One job's outcome; outputs lists only files successfully saved."""

    index: int
    task: str
    device: str | None
    outputs: dict[str, str] = field(default_factory=dict)
    error: str | None = None
    traceback: str | None = None
    pid: int | None = None
    started_at: float | None = None
    finished_at: float | None = None

    @property
    def ok(self):
        return self.error is None


def _init_worker(devices, threads):
    global _device, _models, _initialization_error
    _device = devices.get()
    _models = {}
    _initialization_error = None
    try:
        import torch
        torch.set_num_threads(threads)
        if _device.startswith("cuda:"):
            torch.cuda.set_device(_device)
    except Exception as exc:
        _initialization_error = (f"{type(exc).__name__}: {exc}", traceback.format_exc())


def _model_class(task):
    if task == "synthstrip":
        from .synthstrip import SynthStrip
        return SynthStrip
    if task == "synthmorph":
        from .synthmorph import SynthMorph
        return SynthMorph
    if task == "wmh_synthseg":
        from .wmh_synthseg import WMHSynthSeg
        return WMHSynthSeg
    raise ValueError(f"Unknown task: {task}")


def _run_job(index, job):
    outcome = BatchResult(
        index=index, task=job["task"], device=_device,
        pid=os.getpid(), started_at=time.time(),
    )
    try:
        if _initialization_error is not None:
            outcome.error, outcome.traceback = _initialization_error
            return outcome
        options = job.get("model", {})
        key = (job["task"], json.dumps(options, sort_keys=True, default=os.fspath))
        if key not in _models:
            _models[key] = _model_class(job["task"])(device=_device, **options)
        kwargs = dict(job["kwargs"])
        if job["task"] == "wmh_synthseg" and "lesion_probability" in job["outputs"]:
            kwargs["save_lesion_probabilities"] = True
        result = _models[key](**kwargs)
        for name, path in job["outputs"].items():
            getattr(result, name).save(path)
            outcome.outputs[name] = path
    except Exception as exc:
        outcome.error = f"{type(exc).__name__}: {exc}"
        outcome.traceback = traceback.format_exc()
    finally:
        outcome.finished_at = time.time()
    return outcome


def _prepare_jobs(jobs, overwrite):
    prepared, paths = [], set()
    for index, job in enumerate(jobs):
        prefix = f"job {index}"
        if not isinstance(job, dict):
            raise TypeError(f"{prefix}: expected a job dictionary")
        unknown = set(job) - {"task", "model", "kwargs", "outputs"}
        if unknown:
            raise ValueError(f"{prefix}: unknown job keys: {sorted(unknown)}")
        task = job.get("task")
        if task not in _OUTPUTS:
            raise ValueError(f"{prefix}: task must be one of {sorted(_OUTPUTS)}")
        options, kwargs, outputs = (job.get(k, {}) for k in ("model", "kwargs", "outputs"))
        if not all(isinstance(v, dict) for v in (options, kwargs, outputs)):
            raise TypeError(f"{prefix}: model, kwargs and outputs must be dictionaries")
        if "device" in options:
            raise ValueError(f"{prefix}: set devices on BatchRunner, not job model")
        json.dumps(options, sort_keys=True, default=os.fspath)
        if not outputs or set(outputs) - _OUTPUTS[task]:
            raise ValueError(f"{prefix}: outputs must use {sorted(_OUTPUTS[task])}")
        normalized = {}
        kwargs = dict(kwargs)
        destinations = list(outputs.items())
        if task == "synthmorph" and kwargs.get("output_dir"):
            debug_directory = Path(kwargs["output_dir"]).expanduser().resolve()
            kwargs["output_dir"] = str(debug_directory)
            destinations.extend((None, debug_directory / name) for name in
                                ("inp_1.nii.gz", "inp_2.nii.gz", "network_transforms.npz"))
        for name, value in destinations:
            path = Path(value).expanduser().resolve()
            if path in paths:
                raise ValueError(f"{prefix}: duplicate output path: {path}")
            if path.exists() and (not overwrite or path.is_dir()):
                raise FileExistsError(f"{prefix}: output already exists: {path}")
            paths.add(path)
            if name is not None:
                normalized[name] = str(path)
        prepared.append({"task": task, "model": dict(options), "kwargs": kwargs, "outputs": normalized})
    # Validate the whole batch before making any output directories.
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    return prepared


class BatchRunner:
    """A persistent pool with one model cache per spawned worker.

    ``devices`` contains strings such as ``("cuda:0", "cuda:1")`` or ``("cpu",)``.
    GPU indices refer to the process's CUDA_VISIBLE_DEVICES mapping. The default
    is one worker per GPU; increasing workers_per_device runs separate models
    concurrently on that GPU and consumes additional GPU memory. Jobs are
    distributed dynamically, and results retain input order. Use a context
    manager or call close() when finished. Calls to run() should be sequential.

    All processes are spawned, never forked. In a Python script construct the
    runner under ``if __name__ == "__main__":``. In notebooks use the CLI or an
    importable script. Each job must save at least one output.
    """

    def __init__(self, devices=("cuda:0",), workers_per_device=1, threads_per_worker=1):
        if isinstance(devices, str):
            raise TypeError("devices must be a sequence, such as ('cuda:0',)")
        devices = tuple(devices)
        if not devices or any(not isinstance(d, str) or not re.fullmatch(r"cpu|cuda:\d+", d) for d in devices):
            raise ValueError("devices must contain cpu or cuda:<index>")
        if len(set(devices)) != len(devices):
            raise ValueError("duplicate devices: use workers_per_device instead")
        if not isinstance(workers_per_device, int) or workers_per_device < 1:
            raise ValueError("workers_per_device must be a positive integer")
        if not isinstance(threads_per_worker, int) or threads_per_worker < 1:
            raise ValueError("threads_per_worker must be a positive integer")
        context = mp.get_context("spawn")
        self._devices = context.Queue()
        for _ in range(workers_per_device):
            for device in devices:
                self._devices.put(device)
        self._pool = ProcessPoolExecutor(
            max_workers=len(devices) * workers_per_device,
            mp_context=context,
            initializer=_init_worker,
            initargs=(self._devices, threads_per_worker),
        )

    def run(self, jobs, *, overwrite=False):
        """Run file-output jobs; inference failures become BatchResult errors.

        Invalid manifests and conflicting output paths raise before dispatch.
        SynthMorph's three output_dir debug files participate in these checks.
        ``overwrite=True`` allows existing output files, but never duplicate
        output paths within a batch. Successfully saved files remain if a later
        output from the same job fails. A worker crash is reported for affected
        jobs; close and recreate the runner before retrying a crashed pool.
        """
        if self._pool is None:
            raise RuntimeError("BatchRunner is closed")
        jobs = _prepare_jobs(jobs, overwrite)
        futures, outcomes = {}, [None] * len(jobs)
        for index, job in enumerate(jobs):
            try:
                futures[index] = self._pool.submit(_run_job, index, job)
            except Exception as exc:
                outcomes[index] = BatchResult(
                    index=index, task=job["task"], device=None,
                    error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc(),
                )
        for index, future in futures.items():
            try:
                outcomes[index] = future.result()
            except Exception as exc:
                outcomes[index] = BatchResult(
                    index=index, task=jobs[index]["task"], device=None,
                    error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc(),
                )
        return outcomes

    def close(self):
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None
            self._devices.close()
            self._devices.join_thread()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


def run_batch(jobs, *, devices=("cuda:0",), workers_per_device=1, threads_per_worker=1, overwrite=False):
    """Process one batch; use BatchRunner to reuse models across several batches."""
    with BatchRunner(devices, workers_per_device, threads_per_worker) as runner:
        return runner.run(jobs, overwrite=overwrite)
