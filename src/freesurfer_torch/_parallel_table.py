"""Run one model's table across the current process and spawned workers."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def run_parallel(cases, model, task, workers, threads_per_worker, make_job, run_local):
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError('workers must be a positive integer')
    if isinstance(threads_per_worker, bool) or not isinstance(threads_per_worker, int) or threads_per_worker < 1:
        raise ValueError('threads_per_worker must be a positive integer')
    if workers == 1 or len(cases) < 2:
        return [run_local(case) for case in cases]

    import torch
    from .batch import BatchRunner

    count = min(workers, len(cases))
    device = str(model.device)
    if device == 'cuda':
        device = f'cuda:{torch.cuda.current_device()}'
    child_indices = [index for index in range(len(cases)) if index % count]
    child_jobs = [make_job(cases[index]) for index in child_indices]
    saved = [None] * len(cases)
    original_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(threads_per_worker)
        with BatchRunner(devices=(device,), workers_per_device=count - 1,
                         threads_per_worker=threads_per_worker) as runner:
            with ThreadPoolExecutor(max_workers=1) as dispatch:
                future = dispatch.submit(runner.run, child_jobs, overwrite=True)
                try:
                    for index in range(0, len(cases), count):
                        saved[index] = run_local(cases[index])
                except Exception:
                    try:
                        future.result()
                    except Exception:
                        pass
                    raise
                else:
                    outcomes = future.result()
    finally:
        torch.set_num_threads(original_threads)
    for index, outcome in zip(child_indices, outcomes):
        if outcome.error:
            raise RuntimeError(f'{task} row {index}: {outcome.error}')
        saved[index] = {name: Path(path) for name, path in outcome.outputs.items()}
    return saved
