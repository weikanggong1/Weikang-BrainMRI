"""A derived release must retain the certified evidence and reject stale links."""

from argparse import Namespace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import derive_equivalent_bundle as release
from attest_v07_runtime_equivalence import package_files, sha256, tree_hash


class DerivedBundle(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        old_root, new_root = root / "old-source", root / "new-source"
        old_root.mkdir()
        new_root.mkdir()
        (new_root / "__init__.py").write_text("__version__ = '0.7.0'\n")
        old_tree = "a" * 64
        new_files = package_files(new_root)
        new_tree = tree_hash(new_files)
        environment = {"python": "3.11.0", "platform": "Linux-test",
                       "torch_cuda": "12.4", "cudnn": 90100, "cuda_available": True}
        old_software = {**environment, "package_tree_sha256": old_tree,
                        "versions": {"freesurfer-torch": "0.6.0", "torch": "2.5.1"}}
        new_software = {**environment, "package_root": str(new_root),
                        "package_files_sha256": new_files,
                        "package_tree_sha256": new_tree,
                        "versions": {"freesurfer-torch": "0.7.0", "torch": "2.5.1"}}
        source = {"standalone_verified": True, "files": [{"path": "bin/recon-all", "sha256": "b" * 64}],
                  "verification": {"profile_id": "fixed-profile", "software": old_software,
                                   "evidence": {"comparison": {"sha256": "c" * 64}}}}
        source_path = root / "certified.json"
        source_path.write_text(json.dumps(source))
        copy = root / "release-bundle"
        copy.mkdir()
        (copy / "manifest.json").write_bytes(source_path.read_bytes())
        frozen_path = root / "frozen.json"
        frozen_path.write_text(json.dumps({"bundle_manifest_sha256": sha256(source_path),
                                           "software": new_software}))
        report = {"old_verified_bundle_manifest_sha256": sha256(source_path),
                  "old_package_tree_sha256": old_tree,
                  "new_code_manifest_sha256": sha256(frozen_path),
                  "new_package_tree_sha256": new_tree, "new_version": "0.7.0"}
        attestation_path = root / "attestation.json"
        attestation_path.write_text(json.dumps(report))
        old_code_path = root / "old-code.json"
        old_code_path.write_text("{}")
        pyproject_path = root / "pyproject.toml"
        pyproject_path.write_text('[project]\nversion = "0.7.0"\n')
        self.args = Namespace(certified_bundle_manifest=source_path, bundle_copy=copy,
                              old_code_manifest=old_code_path, old_package_root=old_root,
                              new_package_root=new_root, attestation=attestation_path,
                              new_code_manifest=frozen_path, new_pyproject=pyproject_path)
        self.report = report

    def test_derived_manifest_retains_old_evidence_and_marks_no_new_run(self):
        with patch.object(release, "attest", return_value=self.report):
            path, original, derived = release.derive(self.args)
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(derived["verification"]["certification_kind"], "source_equivalent_release")
        self.assertEqual(derived["verification"]["source_certification"]["verification"],
                         json.loads(original)["verification"])
        self.assertIn("No 0.7 reconstruction", derived["verification"]["release_equivalence"]["boundary"])
        self.assertEqual(derived["verification"]["software"]["package_tree_sha256"],
                         self.report["new_package_tree_sha256"])

    def test_tampered_copy_attestation_and_code_snapshot_are_rejected(self):
        with patch.object(release, "attest", return_value=self.report):
            path = self.args.bundle_copy / "manifest.json"
            path.write_text(path.read_text() + " ")
            with self.assertRaisesRegex(ValueError, "no longer matches"):
                release.derive(self.args)
            path.write_bytes(self.args.certified_bundle_manifest.read_bytes())

            self.args.attestation.write_text(json.dumps({**self.report, "new_version": "0.8.0"}))
            with self.assertRaisesRegex(ValueError, "stale or altered"):
                release.derive(self.args)
            self.args.attestation.write_text(json.dumps(self.report))

            frozen = json.loads(self.args.new_code_manifest.read_text())
            frozen["software"]["package_tree_sha256"] = "0" * 64
            self.args.new_code_manifest.write_text(json.dumps(frozen))
            with self.assertRaisesRegex(ValueError, "another 0.7 code snapshot"):
                release.derive(self.args)

    def test_changed_torch_version_is_rejected(self):
        frozen = json.loads(self.args.new_code_manifest.read_text())
        frozen["software"]["versions"]["torch"] = "2.6.0"
        self.args.new_code_manifest.write_text(json.dumps(frozen))
        report = {**self.report, "new_code_manifest_sha256": sha256(self.args.new_code_manifest)}
        self.args.attestation.write_text(json.dumps(report))
        with patch.object(release, "attest", return_value=report):
            with self.assertRaisesRegex(ValueError, "Runtime dependency versions changed"):
                release.derive(self.args)


if __name__ == "__main__":
    unittest.main()
