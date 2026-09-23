"""Run the three public T1w examples with two persistent GPU workers."""

from dataclasses import asdict
import json
from pathlib import Path

from freesurfer_torch import BatchRunner


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUTPUT = ROOT / "results" / "python"


def main():
    jobs = []
    for subject in ("sub-01", "sub-02", "sub-03"):
        moving = str(DATA / f"{subject}_T1w.nii.gz")
        jobs.append({
            "task": "synthstrip",
            "kwargs": {"image": moving},
            "outputs": {
                "image": str(OUTPUT / f"{subject}_brain.nii.gz"),
                "mask": str(OUTPUT / f"{subject}_mask.nii.gz"),
            },
        })
        if subject != "sub-01":
            jobs.append({
                "task": "synthmorph",
                "model": {"model": "joint"},
                "kwargs": {
                    "moving": moving,
                    "fixed": str(DATA / "sub-01_T1w.nii.gz"),
                },
                "outputs": {
                    "moved": str(OUTPUT / f"{subject}_in_sub-01.nii.gz"),
                    "transform": str(OUTPUT / f"{subject}_to_sub-01.mgz"),
                },
            })

    with BatchRunner(devices=("cuda:0", "cuda:1"),
                     workers_per_device=1, threads_per_worker=4) as runner:
        results = runner.run(jobs)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "batch_report.json").write_text(
        json.dumps([asdict(result) for result in results], indent=2) + "\n"
    )
    for result in results:
        print(result.index, result.task, result.device, result.outputs, result.error)
    if any(not result.ok for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
