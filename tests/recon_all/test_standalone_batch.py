"""Batch recon-all keeps subjects isolated and schedules one job per GPU."""

import threading
import time

import pytest

from freesurfer_torch.recon_all import standalone


def jobs_in(tmp_path, count):
    jobs = []
    for index in range(count):
        image = tmp_path / f"input{index}.nii.gz"
        image.write_bytes(f"t1-{index}".encode())
        jobs.append({"t1": image, "subject": f"sub{index}",
                     "subjects_dir": tmp_path / f"output{index}" / "subjects"})
    return jobs


@pytest.mark.parametrize("invalid", ["same_root", "nested_root", "nonempty_root",
                                     "missing_input", "inside_bundle"])
def test_batch_rejects_invalid_jobs_before_dispatch(tmp_path, monkeypatch, invalid):
    jobs = jobs_in(tmp_path, 2)
    if invalid == "same_root":
        jobs[1]["subjects_dir"] = jobs[0]["subjects_dir"]
    elif invalid == "nested_root":
        jobs[1]["subjects_dir"] = jobs[0]["subjects_dir"] / "nested"
    elif invalid == "nonempty_root":
        jobs[1]["subjects_dir"].mkdir(parents=True)
        (jobs[1]["subjects_dir"] / "existing.txt").write_text("keep")
    elif invalid == "inside_bundle":
        jobs[1]["subjects_dir"] = tmp_path / "bundle" / "new"
    else:
        jobs[1]["t1"] = tmp_path / "missing.nii.gz"

    called = []
    monkeypatch.setattr(standalone, "run_recon_all", lambda *a, **k: called.append((a, k)))
    with pytest.raises((ValueError, FileNotFoundError, FileExistsError)):
        standalone.run_recon_all_batch(jobs, tmp_path / "bundle", devices=("cuda:0", "cuda:1"))
    assert not called
    assert not jobs[0]["subjects_dir"].exists()


def test_batch_uses_distinct_gpus_concurrently_and_preserves_order(tmp_path, monkeypatch):
    jobs = jobs_in(tmp_path, 2)
    bundle = tmp_path / "bundle"
    license_file = tmp_path / "license.txt"
    barrier = threading.Barrier(2)
    calls = []
    lock = threading.Lock()

    def fake_run(t1, subject, subjects_dir, bundle_root, *, device, threads,
                 license_file, development_bundle):
        with lock:
            calls.append((t1, subject, subjects_dir, bundle_root, device, threads,
                          license_file, development_bundle))
        barrier.wait(timeout=3)
        return {"subject": subject, "device": device, "subjects_dir": str(subjects_dir)}

    monkeypatch.setattr(standalone, "run_recon_all", fake_run)
    reports = standalone.run_recon_all_batch(
        jobs, bundle, devices=("cuda:0", "cuda:1"), threads=4,
        license_file=license_file)
    assert [report["subject"] for report in reports] == [job["subject"] for job in jobs]
    assert [report["device"] for report in reports] == ["cuda:0", "cuda:1"]
    assert {(call[1], call[4]) for call in calls} == {("sub0", "cuda:0"), ("sub1", "cuda:1")}
    assert all(call[0] == jobs[int(call[1][-1])]["t1"].resolve()
               and call[2] == jobs[int(call[1][-1])]["subjects_dir"].resolve()
               and call[3] == bundle and call[5:] == (4, license_file, False)
               for call in calls)


def test_batch_runs_jobs_on_one_gpu_in_sequence(tmp_path, monkeypatch):
    jobs = jobs_in(tmp_path, 3)
    active = maximum = 0
    order = []
    lock = threading.Lock()

    def fake_run(t1, subject, subjects_dir, bundle_root, **kwargs):
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
            order.append(subject)
        time.sleep(0.02)
        with lock:
            active -= 1
        assert kwargs["device"] == "cuda:0"
        return {"subject": subject}

    monkeypatch.setattr(standalone, "run_recon_all", fake_run)
    reports = standalone.run_recon_all_batch(jobs, tmp_path / "bundle", devices=("cuda:0",))
    assert maximum == 1
    assert order == [job["subject"] for job in jobs]
    assert [report["subject"] for report in reports] == order


def test_batch_finishes_other_jobs_before_raising_aggregate_error(tmp_path, monkeypatch):
    jobs = jobs_in(tmp_path, 3)
    completed = []
    lock = threading.Lock()

    def fake_run(t1, subject, subjects_dir, bundle_root, **kwargs):
        if subject == "sub0":
            raise RuntimeError("deliberate failure")
        with lock:
            completed.append(subject)
        return {"subject": subject}

    monkeypatch.setattr(standalone, "run_recon_all", fake_run)
    with pytest.raises(RuntimeError, match=r"job 0 \(sub0\).*deliberate failure"):
        standalone.run_recon_all_batch(jobs, tmp_path / "bundle", devices=("cuda:0", "cuda:1"))
    assert set(completed) == {"sub1", "sub2"}
