"""An unrelated 0.7 change must not mask a changed recon-all runtime."""

from contextlib import contextmanager
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import attest_v07_runtime_equivalence as attestation


def refresh_new_manifest(root, path):
    files = attestation.package_files(root)
    path.write_text(json.dumps({"software": {
        "package_files_sha256": files,
        "package_tree_sha256": attestation.tree_hash(files),
        "versions": {"fudan-neuroimaging-toolkit": "0.7.0"},
    }}))


@contextmanager
def fixture():
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        old, new = root / "old", root / "new"
        for name in attestation.RUNTIME_FILES:
            for package in (old, new):
                path = package / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("# frozen\n")
        old_init = ("__version__ = '0.6.0'\n"
                    "def __getattr__(name):\n"
                    "    if name in ('FastVBM',\n"
                    "                'register_gm'):\n"
                    "        return name\n")
        (old / "__init__.py").write_text(old_init)
        (new / "__init__.py").write_text(
            old_init.replace("__version__ = '0.6.0'", "__version__ = '0.7.0'")
                    .replace("'register_gm'):",
                             "'LinearRegistrationResult', 'register_affine', 'register_gm'):"))
        old_weights = ('MODEL_FILES = {"fast-vbm": ("synthstrip.1.pt",),}\n'
                       'def resolve_weights(filename, explicit=None):\n'
                       '    return explicit or filename\n')
        (old / "weights.py").write_text(old_weights)
        (new / "weights.py").write_text(old_weights.replace(
            '"fast-vbm": ("synthstrip.1.pt",),',
            '"fast-vbm": ("synthstrip.1.pt", "synthmorph.deform.3.h5"),'))
        unrelated = new / "fast_vbm/linear.py"
        unrelated.parent.mkdir(parents=True)
        unrelated.write_text("# new unrelated module\n")
        files = attestation.package_files(old)
        software = {"package_files_sha256": files,
                    "package_tree_sha256": attestation.tree_hash(files),
                    "versions": {"fudan-neuroimaging-toolkit": "0.6.0"}}
        code = root / "code.json"
        code.write_text(json.dumps({"software": software}))
        bundle = root / "bundle.json"
        bundle.write_text(json.dumps({"standalone_verified": True,
                                      "verification": {"software": software,
                                                       "evidence": {"code_manifest": {
                                                           "sha256": attestation.sha256(code)}}}}))
        new_code = root / "new_code.json"
        refresh_new_manifest(new, new_code)
        pyproject = root / "pyproject.toml"
        pyproject.write_text('[project]\nversion = "0.7.0"\n')
        yield old, new, code, bundle, new_code, pyproject


class RuntimeEquivalence(unittest.TestCase):
    def test_exact_delta_passes_but_runtime_or_link_changes_fail(self):
        with fixture() as (old, new, code, bundle, new_code, pyproject):
            report = attestation.attest(code, bundle, old, new, new_code, pyproject)
            self.assertEqual(report["changed_runtime_files"], ["__init__.py", "weights.py"])
            self.assertEqual(report["other_changed_package_files"], ["fast_vbm/linear.py"])
            path = new / "synthseg_parc/postprocess.py"
            path.write_text("# changed segmentation\n")
            refresh_new_manifest(new, new_code)
            with self.assertRaisesRegex(ValueError, "Unexpected recon-all runtime changes"):
                attestation.attest(code, bundle, old, new, new_code, pyproject)

        with fixture() as (old, new, code, bundle, new_code, pyproject):
            path = new / "weights.py"
            path.write_text(path.read_text() + "# extra change\n")
            refresh_new_manifest(new, new_code)
            with self.assertRaisesRegex(ValueError, "Weight resolver has more"):
                attestation.attest(code, bundle, old, new, new_code, pyproject)

        with fixture() as (old, new, code, bundle, new_code, pyproject):
            record = json.loads(bundle.read_text())
            record["verification"]["evidence"]["code_manifest"]["sha256"] = "0" * 64
            bundle.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "not linked"):
                attestation.attest(code, bundle, old, new, new_code, pyproject)

    def test_unlisted_package_change_and_v07_version_mismatch_fail(self):
        with fixture() as (old, new, code, bundle, new_code, pyproject):
            path = new / "synthsr/new_backend.py"
            path.parent.mkdir(parents=True)
            path.write_text("# unrelated package addition\n")
            refresh_new_manifest(new, new_code)
            with self.assertRaisesRegex(ValueError, "Unexpected non-runtime package changes"):
                attestation.attest(code, bundle, old, new, new_code, pyproject)

        with fixture() as (old, new, code, bundle, new_code, pyproject):
            pyproject.write_text('[project]\nversion = "0.8.0"\n')
            with self.assertRaisesRegex(ValueError, "versions do not agree"):
                attestation.attest(code, bundle, old, new, new_code, pyproject)

        with fixture() as (old, new, code, bundle, new_code, pyproject):
            record = json.loads(new_code.read_text())
            record["software"]["package_tree_sha256"] = "0" * 64
            new_code.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "differs from the v0.7 code snapshot"):
                attestation.attest(code, bundle, old, new, new_code, pyproject)


if __name__ == "__main__":
    unittest.main()
