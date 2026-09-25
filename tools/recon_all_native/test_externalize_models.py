"""Externalization must fail closed and never alter the certified source bundle."""

from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import externalize_models as converter


def digest(data):
    return hashlib.sha256(data).hexdigest()


class ExternalizeModels(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.source = root / "certified-bundle"
        self.copy = root / "bundle-copy"
        self.external = root / "weights"
        for directory in (self.source, self.copy, self.external):
            directory.mkdir()
        rows = []
        for index, name in enumerate(converter.MODEL_FILES):
            data = f"certified model {index}\n".encode()
            row = {"path": f"models/{name}", "bytes": len(data),
                   "sha256": digest(data), "kind": "asset", "staged": True}
            rows.append(row)
            for bundle in (self.source, self.copy):
                path = bundle / row["path"]
                path.parent.mkdir(exist_ok=True)
                path.write_bytes(data)
            (self.external / name).write_bytes(data)
        other = {"path": "bin/recon-all", "bytes": 3,
                 "sha256": digest(b"bin"), "kind": "elf", "staged": True}
        for bundle in (self.source, self.copy):
            (bundle / other["path"]).parent.mkdir()
            (bundle / other["path"]).write_bytes(b"bin")
        manifest = {"standalone_verified": True, "candidate_bundle": True,
                    "files": [*rows, other],
                    "required_resources": [row["path"] for row in rows] + ["bin/recon-all"],
                    "verification": {"profile_id": "certified", "evidence": {"run": "old"}}}
        self.source_manifest = self.source / "manifest.json"
        self.source_manifest.write_text(json.dumps(manifest))
        (self.copy / "manifest.json").write_bytes(self.source_manifest.read_bytes())

    def arguments(self, *extra):
        return ["--certified-manifest", str(self.source_manifest),
                "--bundle-copy", str(self.copy), "--models-dir", str(self.external), *extra]

    def invoke(self, *extra):
        output = io.StringIO()
        with redirect_stdout(output):
            code = converter.main(self.arguments(*extra))
        return code, json.loads(output.getvalue())

    def test_check_only_then_convert_preserves_source_and_marks_candidate(self):
        original = self.source_manifest.read_bytes()
        code, report = self.invoke("--check-only")
        self.assertEqual(code, 0)
        self.assertFalse(report["written"])
        self.assertEqual((self.copy / "manifest.json").read_bytes(), original)

        code, report = self.invoke()
        self.assertEqual(code, 0)
        self.assertTrue(report["written"])
        candidate = json.loads((self.copy / "manifest.json").read_text())
        self.assertIs(candidate["standalone_verified"], False)
        self.assertIs(candidate["candidate_bundle"], True)
        self.assertNotIn("verification", candidate)
        self.assertEqual(len(candidate["external_models"]), converter.MODEL_COUNT)
        self.assertEqual(candidate["externalization"]["source_certified_bundle_manifest_sha256"],
                         digest(original))
        self.assertIs(candidate["externalization"]["requires_recertification"], True)
        self.assertEqual(candidate["externalization"]["source_verification"],
                         json.loads(original)["verification"])
        self.assertEqual([row["path"] for row in candidate["files"]], ["bin/recon-all"])
        self.assertEqual(candidate["required_resources"], ["bin/recon-all"])
        self.assertEqual(list((self.copy / "models").iterdir()), [])
        self.assertEqual(len(list(self.external.iterdir())), converter.MODEL_COUNT)
        self.assertEqual(self.source_manifest.read_bytes(), original)
        self.assertEqual(len(list((self.source / "models").iterdir())), converter.MODEL_COUNT)

    def test_bad_external_bytes_and_symlink_fail_without_mutation(self):
        original = (self.copy / "manifest.json").read_bytes()
        model = self.external / converter.MODEL_FILES[0]
        model.write_bytes(b"X" * model.stat().st_size)
        code, report = self.invoke()
        self.assertEqual(code, 2)
        self.assertIn("differs from certified manifest", report["error"])
        self.assertEqual((self.copy / "manifest.json").read_bytes(), original)
        self.assertEqual(len(list((self.copy / "models").iterdir())), converter.MODEL_COUNT)

        model.unlink()
        outside = self.external.parent / "outside.h5"
        outside.write_bytes((self.copy / "models" / converter.MODEL_FILES[0]).read_bytes())
        model.symlink_to(outside)
        code, report = self.invoke()
        self.assertEqual(code, 2)
        self.assertIn("not a local regular file", report["error"])
        self.assertEqual((self.copy / "manifest.json").read_bytes(), original)

    def test_uncertified_source_or_nonidentical_copy_is_rejected(self):
        original = self.source_manifest.read_bytes()
        (self.copy / "manifest.json").write_bytes(original + b" ")
        self.assertEqual(self.invoke()[0], 2)
        (self.copy / "manifest.json").write_bytes(original)
        source = json.loads(original)
        source["standalone_verified"] = False
        self.source_manifest.write_text(json.dumps(source))
        (self.copy / "manifest.json").write_bytes(self.source_manifest.read_bytes())
        code, report = self.invoke()
        self.assertEqual(code, 2)
        self.assertIn("not certified", report["error"])

    def test_unlink_failure_keeps_bundle_unverified(self):
        original_unlink = Path.unlink

        def fail_on_model(path, *args, **kwargs):
            if path.name == converter.MODEL_FILES[0] and path.parent == self.copy / "models":
                raise PermissionError("fixture cannot remove model")
            return original_unlink(path, *args, **kwargs)

        with patch.object(Path, "unlink", fail_on_model):
            code, report = self.invoke()
        self.assertEqual(code, 2)
        self.assertIn("fixture cannot remove model", report["error"])
        candidate = json.loads((self.copy / "manifest.json").read_text())
        self.assertIs(candidate["standalone_verified"], False)


if __name__ == "__main__":
    unittest.main()
