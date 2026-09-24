"""Dispatcher tests use tiny fake models; no GPU or checkpoints are required."""

import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import textwrap
from types import SimpleNamespace
import unittest
from unittest import mock


SOURCE = Path(__file__).resolve().parents[2] / "src" / "freesurfer_torch" / "batch.py"
SPEC = importlib.util.spec_from_file_location("batch_under_test", SOURCE)
batch = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = batch
SPEC.loader.exec_module(batch)


class BatchValidationTests(unittest.TestCase):
    def test_duplicate_outputs_rejected_before_directories_created(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "new" / "mask.nii.gz"
            job = {"task": "synthstrip", "kwargs": {"image": "input.nii.gz"}, "outputs": {"mask": path}}
            with self.assertRaisesRegex(ValueError, "duplicate output"):
                batch._prepare_jobs([job, job], overwrite=True)
            self.assertFalse(path.parent.exists())

    def test_existing_output_requires_explicit_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "mask.nii.gz"
            path.write_text("old")
            job = {"task": "synthstrip", "outputs": {"mask": path}}
            with self.assertRaises(FileExistsError):
                batch._prepare_jobs([job], overwrite=False)
            self.assertEqual(batch._prepare_jobs([job], overwrite=True)[0]["outputs"]["mask"], str(path))
            self.assertEqual(path.read_text(), "old")

    def test_wmh_outputs_are_validated_before_dispatch(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            seg = root / "case_seg.nii.gz"
            prob = root / "case_seg.lesion_probs.nii.gz"
            job = {"task": "wmh_synthseg", "kwargs": {"image": "case.nii.gz", "crop": True},
                   "outputs": {"segmentation": seg, "lesion_probability": prob}}
            prepared = batch._prepare_jobs([job], overwrite=False)
            self.assertEqual(set(prepared[0]["outputs"]), {"segmentation", "lesion_probability"})
            with self.assertRaisesRegex(ValueError, "duplicate output"):
                batch._prepare_jobs([job, job], overwrite=True)

    def test_synthsr_has_one_saved_output(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "case_synthsr.nii.gz"
            job = {"task": "synthsr", "kwargs": {"image": "case_FLAIR.nii.gz"},
                   "outputs": {"image": output}}
            prepared = batch._prepare_jobs([job], overwrite=False)
            self.assertEqual(prepared[0]["outputs"], {"image": str(output)})
            with self.assertRaises(ValueError):
                batch._prepare_jobs([{"task": "synthsr", "outputs": {"mask": output}}], False)

    def test_fast_accepts_tissue_and_bias_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            job = {
                "task": "fast",
                "kwargs": {"image": "T1_brain.nii.gz"},
                "outputs": {
                    "pve_gm": root / "T1_brain_pve_1.nii.gz",
                    "bias_field": root / "T1_brain_bias.nii.gz",
                    "restored": root / "T1_brain_restore.nii.gz",
                },
            }
            prepared = batch._prepare_jobs([job], overwrite=False)
            self.assertEqual(set(prepared[0]["outputs"]),
                             {"pve_gm", "bias_field", "restored"})
            with self.assertRaises(ValueError):
                batch._prepare_jobs([
                    {"task": "fast", "outputs": {"lesion_probability": root / "bad.nii.gz"}}
                ], False)

    def test_fast_vbm_accepts_input_and_template_grid_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            job = {
                "task": "fast_vbm",
                "kwargs": {"image": "T1w.nii.gz", "template": "GM_template.nii.gz"},
                "outputs": {
                    "brain_mask": root / "mask.nii.gz",
                    "pve_gm": root / "pve_1.nii.gz",
                    "warped_gm": root / "warped.nii.gz",
                    "jacobian": root / "jacobian.nii.gz",
                    "modulated_gm": root / "modulated.nii.gz",
                },
            }
            prepared = batch._prepare_jobs([job], overwrite=False)
            self.assertEqual(set(prepared[0]["outputs"]), set(job["outputs"]))
            with self.assertRaises(ValueError):
                batch._prepare_jobs([
                    {"task": "fast_vbm", "outputs": {"distance": root / "bad.nii.gz"}}
                ], False)

    def test_fast_dispatch_reuses_model_and_saves_atomically(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            first_output = root / "first_pve_1.nii.gz"
            second_output = root / "second_pve_1.nii.gz"
            first_output.write_text("old")
            initialized, calls, saved = [], [], []

            class Volume:
                def __init__(self, value):
                    self.value = value

                def save(self, path):
                    path = Path(path)
                    saved.append(path)
                    path.write_text(self.value)

            class FakeFast:
                def __init__(self, device, bias_fwhm_mm):
                    initialized.append((device, bias_fwhm_mm))

                def __call__(self, image, mask=None):
                    calls.append((image, mask))
                    return SimpleNamespace(pve_gm=Volume(image))

            jobs = batch._prepare_jobs([
                {
                    "task": "fast", "model": {"bias_fwhm_mm": 0.0},
                    "kwargs": {"image": "first.nii.gz", "mask": "mask.nii.gz"},
                    "outputs": {"pve_gm": first_output},
                },
                {
                    "task": "fast", "model": {"bias_fwhm_mm": 0.0},
                    "kwargs": {"image": "second.nii.gz"},
                    "outputs": {"pve_gm": second_output},
                },
            ], overwrite=True)
            with mock.patch.object(batch, "_device", "cpu"), \
                    mock.patch.object(batch, "_models", {}), \
                    mock.patch.object(batch, "_initialization_error", None), \
                    mock.patch.object(batch, "_model_class", return_value=FakeFast):
                outcomes = [batch._run_job(index, job) for index, job in enumerate(jobs)]

            self.assertTrue(all(outcome.ok for outcome in outcomes))
            self.assertEqual(initialized, [("cpu", 0.0)])
            self.assertEqual(calls, [("first.nii.gz", "mask.nii.gz"),
                                     ("second.nii.gz", None)])
            self.assertEqual(first_output.read_text(), "first.nii.gz")
            self.assertEqual(second_output.read_text(), "second.nii.gz")
            self.assertEqual(outcomes[0].outputs, {"pve_gm": str(first_output.resolve())})
            self.assertEqual(outcomes[1].outputs, {"pve_gm": str(second_output.resolve())})
            self.assertTrue(all(path.parent == root and path.name.endswith(".nii.gz")
                                for path in saved))
            self.assertTrue(all(path not in (first_output, second_output) for path in saved))
            self.assertFalse(any(path.exists() for path in saved))

    def test_fast_vbm_dispatch_reuses_pipeline_and_selects_volume_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            initialized, calls = [], []

            class Volume:
                def __init__(self, value):
                    self.value = value

                def save(self, path):
                    Path(path).write_text(self.value)

            class Result:
                def __init__(self, image):
                    self.image = image

                def volumes(self):
                    return {
                        "pve_gm": Volume(f"{self.image}:pve"),
                        "modulated_gm": Volume(f"{self.image}:modulated"),
                    }

                def report(self):
                    return {"case": self.image, "registration": {"jacobian_min": 0.2}}

            class FakeFastVBM:
                def __init__(self, device, registration_backend):
                    initialized.append((device, registration_backend))

                def __call__(self, image, template, brain_mask=None):
                    calls.append((image, template, brain_mask))
                    return Result(image)

            jobs = batch._prepare_jobs([
                {
                    "task": "fast_vbm", "model": {
                        "registration_backend": "fnirt"
                    },
                    "kwargs": {"image": "first.nii.gz", "template": "template.nii.gz"},
                    "outputs": {"pve_gm": root / "first_pve.nii.gz"},
                },
                {
                    "task": "fast_vbm", "model": {
                        "registration_backend": "fnirt"
                    },
                    "kwargs": {
                        "image": "second.nii.gz", "template": "template.nii.gz",
                        "brain_mask": "second_mask.nii.gz",
                    },
                    "outputs": {"modulated_gm": root / "second_mod.nii.gz"},
                },
            ], overwrite=False)
            with mock.patch.object(batch, "_device", "cuda:0"), \
                    mock.patch.object(batch, "_models", {}), \
                    mock.patch.object(batch, "_initialization_error", None), \
                    mock.patch.object(batch, "_model_class", return_value=FakeFastVBM):
                outcomes = [batch._run_job(index, job) for index, job in enumerate(jobs)]

            self.assertTrue(all(outcome.ok for outcome in outcomes))
            self.assertEqual(initialized, [("cuda:0", "fnirt")])
            self.assertEqual(calls, [
                ("first.nii.gz", "template.nii.gz", None),
                ("second.nii.gz", "template.nii.gz", "second_mask.nii.gz"),
            ])
            self.assertEqual((root / "first_pve.nii.gz").read_text(), "first.nii.gz:pve")
            self.assertEqual((root / "second_mod.nii.gz").read_text(), "second.nii.gz:modulated")
            self.assertEqual(outcomes[0].metadata["registration"]["jacobian_min"], 0.2)

    def test_synthmorph_debug_outputs_participate_in_collision_checks(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            debug = root / "debug"
            first = {"task": "synthmorph", "kwargs": {"output_dir": debug},
                     "outputs": {"moved": root / "first.nii.gz"}}
            second = {"task": "synthmorph", "kwargs": {"output_dir": debug},
                      "outputs": {"moved": root / "second.nii.gz"}}
            with self.assertRaisesRegex(ValueError, "duplicate output"):
                batch._prepare_jobs([first, second], overwrite=True)
            self.assertFalse(debug.exists())
            explicit = {"task": "synthstrip", "outputs": {"mask": debug / "inp_1.nii.gz"}}
            with self.assertRaisesRegex(ValueError, "duplicate output"):
                batch._prepare_jobs([explicit, first], overwrite=True)
            self.assertFalse(debug.exists())
            debug.mkdir()
            (debug / "network_transforms.npz").write_text("old debug data")
            with self.assertRaises(FileExistsError):
                batch._prepare_jobs([first], overwrite=False)
            prepared = batch._prepare_jobs([first], overwrite=True)
            self.assertEqual(prepared[0]["kwargs"]["output_dir"], str(debug))
            self.assertEqual((debug / "network_transforms.npz").read_text(), "old debug data")

    def test_invalid_manifest_and_device_assignment(self):
        invalid = [
            {"task": "unknown", "outputs": {"mask": "x"}},
            {"task": "synthstrip", "outputs": {}},
            {"task": "synthstrip", "outputs": {"transform": "x"}},
            {"task": "synthstrip", "outputs": {"mask": "x"}, "model": {"device": "cuda:0"}},
        ]
        for job in invalid:
            with self.subTest(job=job), self.assertRaises(ValueError):
                batch._prepare_jobs([job], overwrite=False)
        for devices in [(), ("cuda",), ("cuda:0", "cuda:0")]:
            with self.subTest(devices=devices), self.assertRaises(ValueError):
                batch.BatchRunner(devices)

    def test_empty_and_closed_runner(self):
        with batch.BatchRunner(devices=("cpu",)) as runner:
            self.assertEqual(runner.run([]), [])
        with self.assertRaisesRegex(RuntimeError, "closed"):
            runner.run([])


class BatchSpawnTests(unittest.TestCase):
    def test_spawn_reuse_errors_order_and_device_distribution(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            package = root / "freesurfer_torch"
            package.mkdir()
            (package / "__init__.py").write_text("")
            shutil.copyfile(SOURCE, package / "batch.py")
            (root / "torch.py").write_text(textwrap.dedent('''
                def set_num_threads(value):
                    assert value > 0
                class cuda:
                    @staticmethod
                    def set_device(device):
                        assert device.startswith("cuda:")
                        if device == "cuda:99":
                            raise RuntimeError("invalid simulated CUDA device")
            '''))
            (package / "synthstrip.py").write_text(textwrap.dedent('''
                import json
                import os
                from pathlib import Path
                import time
                from types import SimpleNamespace

                class Volume:
                    def __init__(self, value):
                        self.value = value
                    def save(self, path):
                        Path(path).write_text(json.dumps(self.value))

                class SynthStrip:
                    def __init__(self, weights=None, device="cpu"):
                        self.calls = 0
                        self.weights = weights
                        self.device = device
                    def __call__(self, image):
                        self.calls += 1
                        if image == "fail":
                            raise ValueError("deliberate job failure")
                        if image == "crash":
                            os._exit(9)
                        time.sleep(0.05)
                        value = dict(pid=os.getpid(), calls=self.calls,
                                     weights=self.weights, device=self.device, image=image)
                        volume = Volume(value)
                        return SimpleNamespace(image=volume, mask=volume, distance=volume)
            '''))
            (package / "synthmorph.py").write_text(textwrap.dedent('''
                from .synthstrip import SynthStrip
                class SynthMorph(SynthStrip):
                    def __call__(self, moving, fixed):
                        result = super().__call__(moving)
                        result.moved = result.image
                        result.fixed_moved = result.image
                        result.transform = result.image
                        result.inverse = result.image
                        return result
            '''))
            (root / "exercise.py").write_text(textwrap.dedent('''
                from dataclasses import asdict
                import json
                from pathlib import Path
                from freesurfer_torch.batch import BatchRunner, run_batch

                def strip(name, image="good", weights="first"):
                    return dict(task="synthstrip", model=dict(weights=weights),
                                kwargs=dict(image=image), outputs=dict(mask=f"out/{name}.json"))

                if __name__ == "__main__":
                    with BatchRunner(devices=("cpu",)) as runner:
                        first = runner.run([
                            strip("a"),
                            dict(task="synthmorph", model=dict(weights="morph"),
                                 kwargs=dict(moving="moving", fixed="fixed"),
                                 outputs=dict(transform="out/transform.json")),
                            strip("fail", "fail"), strip("b"), strip("other", weights="second"),
                        ])
                        second = runner.run([strip("c")])
                    parallel = run_batch([strip(f"parallel-{i}", str(i)) for i in range(8)],
                                         devices=("cuda:0", "cuda:1"))
                    same_gpu = run_batch([strip(f"same-{i}", str(i)) for i in range(8)],
                                         devices=("cpu",), workers_per_device=2)
                    crashed = run_batch([strip("crash", "crash"), strip("after-crash")],
                                        devices=("cpu",))
                    bad_device = run_batch([strip("bad-device")], devices=("cuda:99",))
                    report = dict(first=first, second=second, parallel=parallel,
                                  same_gpu=same_gpu, crashed=crashed, bad_device=bad_device)
                    Path("report.json").write_text(json.dumps(
                        {key: [asdict(v) for v in values] for key, values in report.items()}))
            '''))
            completed = subprocess.run(
                [sys.executable, str(root / "exercise.py")], cwd=root,
                text=True, capture_output=True, timeout=40,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            report = json.loads((root / "report.json").read_text())
            self.assertEqual([item["index"] for item in report["first"]], list(range(5)))
            failed = report["first"][2]
            self.assertIn("ValueError: deliberate job failure", failed["error"])
            self.assertIn("Traceback", failed["traceback"])
            self.assertEqual(failed["outputs"], {})
            for name, count in [("a", 1), ("b", 3), ("c", 4), ("other", 1)]:
                data = json.loads((root / "out" / f"{name}.json").read_text())
                self.assertEqual(data["calls"], count)
            self.assertEqual({item["device"] for item in report["parallel"]}, {"cuda:0", "cuda:1"})
            self.assertEqual([item["index"] for item in report["parallel"]], list(range(8)))
            pids = {json.loads((root / "out" / f"same-{i}.json").read_text())["pid"] for i in range(8)}
            self.assertEqual(len(pids), 2)
            self.assertEqual({item["pid"] for item in report["same_gpu"]}, pids)
            for item in report["first"] + report["same_gpu"] + report["bad_device"]:
                self.assertIsInstance(item["pid"], int)
                self.assertGreater(item["started_at"], 0)
                self.assertGreaterEqual(item["finished_at"], item["started_at"])
            overlaps = [
                min(first["finished_at"], second["finished_at"]) - max(first["started_at"], second["started_at"])
                for index, first in enumerate(report["same_gpu"])
                for second in report["same_gpu"][index + 1:]
                if first["pid"] != second["pid"]
            ]
            self.assertTrue(any(value > 0 for value in overlaps))
            self.assertTrue(all(item["error"] is None for item in report["parallel"]))
            self.assertTrue(all(item["error"] is not None for item in report["crashed"]))
            self.assertIn("invalid simulated CUDA device", report["bad_device"][0]["error"])


if __name__ == "__main__":
    unittest.main()
