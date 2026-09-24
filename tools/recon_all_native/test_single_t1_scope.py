"""The branch audit must fail closed when its source or invocation changes."""

from contextlib import contextmanager
import copy
import hashlib
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from runtime_dispatch import file_row
from script_closure import scan_scripts
import single_t1_scope as scope


@contextmanager
def scoped_fixture(*, version_command=False):
    with tempfile.TemporaryDirectory() as temporary:
        bundle = Path(temporary) / "bundle"
        manifest = {"files": [], "script_dispatch": []}
        contents = {path: "fixture\n" for path in scope.PINNED_FILES}
        for item in scope.RULES:
            contents.setdefault(item["script"], "fixture\n")
            contents[item["script"]] += item["command"] + "\n"
        for path in {row["script"] for row in scope.RULES}:
            contents[path] = "#!/bin/sh\n" + contents[path]
        if version_command:
            contents["upstream_scripts/bin/recon-all"] += (
                "set allinfocmds = (\\\n mri_deface \\\n)\n"
                "foreach cmd ($allinfocmds)\n $cmd -all-info\nend\n")
        for name in scope.GUI_COMMANDS:
            contents["bin/" + name] = "#!/bin/sh\nexit 0\n"
            contents["upstream_scripts/bin/" + name] = "#!/bin/sh\nisargflag\nmri_coreg\n"
        for relative, content in contents.items():
            path = bundle / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
            manifest["files"].append(file_row(bundle, path, "script"))
        pins = {path: hashlib.sha256(contents[path].encode()).hexdigest() for path in scope.PINNED_FILES}
        config = Path(temporary) / "reference.yaml"
        config.write_text("UseSynthSeg:\n    value: True\nDoSynthSR:\n    value: False\nUseStopMaskSCM:\n    value: False\n")
        with patch.object(scope, "PINNED_FILES", pins), \
                patch.object(scope, "REFERENCE_CONFIG_SHA256", hashlib.sha256(config.read_bytes()).hexdigest()):
            scope.configure_scope(bundle, manifest, config)
            yield bundle, manifest, config


def scan(bundle, manifest):
    return scan_scripts(bundle, manifest["files"], runtime_profile=manifest["runtime_profile"],
                        inactive_command_audit=manifest["inactive_command_audit"])


class SingleT1Scope(unittest.TestCase):
    def test_only_reviewed_script_command_pairs_are_suppressed(self):
        with scoped_fixture() as (bundle, manifest, _):
            report = scan(bundle, manifest)
            self.assertTrue(report["script_resource_checks_passed"], report["errors"])
            self.assertEqual(len(report["inactive_commands"]), 16)
            ordinary = scan_scripts(bundle, manifest["files"])
            self.assertEqual(len([row for row in ordinary["errors"] if "command" in row]), 16)
            extra = bundle / "bin/new-command.sh"
            extra.write_text("#!/bin/sh\nmri_deface\n")
            manifest["files"].append(file_row(bundle, extra, "script"))
            report = scan(bundle, manifest)
            self.assertEqual([row["script"] for row in report["errors"]], ["bin/new-command.sh"])

    def test_changed_source_or_configuration_invalidates_entire_audit(self):
        for relative in ("upstream_scripts/bin/recon-all", "etc/global-expert-options.v8.txt",
                         "etc/recon-config.yaml", scope.REFERENCE_CONFIG):
            with self.subTest(relative=relative), scoped_fixture() as (bundle, manifest, _):
                path = bundle / relative
                path.write_text(path.read_text() + "changed\n")
                report = scan(bundle, manifest)
                self.assertFalse(report["script_resource_checks_passed"])
                self.assertEqual(report["inactive_commands"], [])
                self.assertTrue(any(row.get("file") == relative for row in report["errors"]))

    def test_changed_profile_or_evidence_does_not_create_an_exemption(self):
        with scoped_fixture() as (bundle, manifest, _):
            for field in ("runtime_profile", "inactive_command_audit", "boolean_type"):
                changed = copy.deepcopy(manifest)
                if field == "runtime_profile":
                    changed[field]["argv_suffix"].append("-synthsr")
                elif field == "inactive_command_audit":
                    changed[field]["rules"][0]["command"] = "mri_unreviewed"
                else:
                    changed["runtime_profile"]["fresh_subject"] = 1
                report = scan(bundle, changed)
                self.assertFalse(report["script_resource_checks_passed"])
                self.assertEqual(report["inactive_commands"], [])

    def test_version_inventory_remains_required(self):
        with scoped_fixture(version_command=True) as (bundle, manifest, _):
            report = scan(bundle, manifest)
            self.assertTrue(any(row.get("command") == "mri_deface" for row in report["errors"]))
            self.assertFalse(any(row["command"] == "mri_deface" for row in report["inactive_commands"]))

    def test_gui_entry_rejects_execution_and_keeps_original_provenance(self):
        with scoped_fixture() as (bundle, manifest, config):
            for name in scope.GUI_COMMANDS:
                result = subprocess.run(["/bin/sh", str(bundle / "bin" / name)], capture_output=True, text=True)
                self.assertEqual(result.returncode, 64)
                self.assertIn("GUI commands are unavailable", result.stderr)
                original = next(row for row in manifest["files"] if row["path"] == "upstream_scripts/bin/" + name)
                self.assertIs(original["executed"], False)
                self.assertEqual((bundle / original["path"]).read_text(), "#!/bin/sh\nisargflag\nmri_coreg\n")
                archive = bundle / "upstream_scripts/scoped_gui_dispatch/bin" / name
                self.assertEqual(archive.read_text(), "#!/bin/sh\nexit 0\n")
            before = len(manifest["files"])
            scope.configure_scope(bundle, manifest, config)
            self.assertEqual(len(manifest["files"]), before)
            self.assertIs(manifest["standalone_verified"], False)
            self.assertEqual(scope.validate_scope_audit(bundle, manifest["runtime_profile"], manifest["inactive_command_audit"])[1], [])


if __name__ == "__main__":
    unittest.main()
