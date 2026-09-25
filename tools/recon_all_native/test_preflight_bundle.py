"""External model checks for bundles whose weights live outside the native tree."""

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from preflight_bundle import MODEL_FILES


SCRIPT = Path(__file__).with_name("preflight_bundle.py")
MISSING = object()


class ExternalModelsPreflight(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.bundle = self.root / "bundle"
        self.bundle.mkdir()
        self.models = self.root / "models"
        self.models.mkdir()
        self.output = self.root / "preflight.json"

    def row(self, name, data=None):
        data = data if data is not None else name.encode()
        path = self.models / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return {"path": "models/" + name, "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest()}

    def rows(self):
        return [self.row(name) for name in MODEL_FILES]

    def run_preflight(self, rows=MISSING, models_dir=None):
        manifest = {"files": [], "libraries": []}
        if rows is not MISSING:
            manifest["external_models"] = rows
        (self.bundle / "manifest.json").write_text(json.dumps(manifest))
        command = [sys.executable, str(SCRIPT), "--bundle", str(self.bundle),
                   "--output", str(self.output)]
        if models_dir is not None:
            command.extend(["--models-dir", str(models_dir)])
        process = subprocess.run(command, capture_output=True, text=True, check=False)
        return process.returncode, json.loads(self.output.read_text())

    def test_legacy_bundle_needs_no_models_dir(self):
        code, report = self.run_preflight()
        self.assertEqual(code, 0, report["errors"])
        self.assertTrue(report["hash_and_linkage_passed"])
        self.assertEqual(report["checked_external_models"], 0)

    def test_present_external_models_must_be_full_list(self):
        for rows in ([], "models/synthstrip.1.pt", None):
            with self.subTest(rows=rows):
                code, report = self.run_preflight(rows, self.models)
                self.assertEqual(code, 2)
                self.assertFalse(report["hash_and_linkage_passed"])

    def test_fixed_inventory_matches_runner(self):
        runner = Path(__file__).resolve().parents[2] / "src/fnit/recon_all/standalone.py"
        spec = importlib.util.spec_from_file_location("standalone_model_inventory", runner)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(MODEL_FILES, module.MODEL_FILES)

    def test_exact_13_models_have_valid_size_and_sha(self):
        code, report = self.run_preflight(self.rows(), self.models)
        self.assertEqual(code, 0, report["errors"])
        self.assertEqual(report["checked_external_models"], 13)

    def test_models_dir_is_required_for_external_models(self):
        code, report = self.run_preflight(self.rows())
        self.assertEqual(code, 2)
        self.assertIn("--models-dir", report["errors"][0]["error"])
        self.assertEqual(report["checked_external_models"], 0)

    def test_size_and_hash_mismatch_fail(self):
        rows = self.rows()
        rows[0]["bytes"] += 1
        rows[0]["sha256"] = "0" * 64
        code, report = self.run_preflight(rows, self.models)
        self.assertEqual(code, 2)
        self.assertEqual(report["checked_external_models"], 13)
        self.assertEqual({error["error"] for error in report["errors"]},
                         {"External model size mismatch", "External model hash mismatch"})

    def test_rejects_incomplete_duplicate_or_unexpected_inventory(self):
        rows = self.rows()
        cases = (rows[:-1], rows[:-1] + [rows[0].copy()],
                 rows + [{"path": "models/unexpected.h5", "bytes": 1, "sha256": "0" * 64}],
                 rows[:-1] + [{"path": "models/../outside.h5", "bytes": 1,
                               "sha256": "0" * 64}])
        for changed in cases:
            with self.subTest(paths=[row["path"] for row in changed]):
                code, report = self.run_preflight(changed, self.models)
                self.assertEqual(code, 2)
                self.assertEqual(report["checked_external_models"], 0)
                self.assertIn("inventory", report["errors"][0]["error"])

    def test_rejects_all_model_file_symlinks(self):
        rows = self.rows()
        name = MODEL_FILES[0]
        model = self.models / name
        data = model.read_bytes()
        model.unlink()
        outside = self.root / "outside.h5"
        outside.write_bytes(data)
        model.symlink_to(outside)
        code, report = self.run_preflight(rows, self.models)
        self.assertEqual(code, 2)
        self.assertEqual(report["checked_external_models"], 12)
        self.assertIn("symlinked", report["errors"][0]["error"])

        model.unlink()
        shadow = self.models / "shadow.h5"
        shadow.write_bytes(data)
        model.symlink_to(shadow)
        code, report = self.run_preflight(rows, self.models)
        self.assertEqual(code, 2)
        self.assertEqual(report["checked_external_models"], 12)
        self.assertIn("symlinked", report["errors"][0]["error"])

    def test_missing_model_and_invalid_schema_fail(self):
        rows = self.rows()
        (self.models / MODEL_FILES[0]).unlink()
        rows[1].pop("bytes")
        code, report = self.run_preflight(rows, self.models)
        self.assertEqual(code, 2)
        self.assertEqual(report["checked_external_models"], 11)
        self.assertEqual(len(report["errors"]), 2)


if __name__ == "__main__":
    unittest.main()
