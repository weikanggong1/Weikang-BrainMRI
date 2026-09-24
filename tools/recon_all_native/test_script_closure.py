"""Regression fixtures derived from the first real candidate preflight."""

import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from runtime_dispatch import (AUXILIARY_ASSETS, classify_linux_startup, complete_runtime_dispatch,
                              file_row, replace_auxiliary)
from audit_bundle import sha256
from script_closure import STANDARD_UTILITIES, executable_text, mandatory_version_commands, scan_scripts


class ScriptClosure(unittest.TestCase):
    def test_ascii_label_is_data_but_malformed_label_still_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            label = root / "lh.test.label"
            label.write_text("#!ascii label test , from subject fsaverage\n1\n0 1 2 3 0\n")
            rows = [{"path": label.name}]
            report = scan_scripts(root, rows)
            self.assertTrue(report["script_resource_checks_passed"])
            self.assertEqual(len(report["recognized_data_files"]), 1)
            label.write_text("#!ascii label, from file MPM_lh.FG1.label\n1\n0 1 2 3 0\n")
            report = scan_scripts(root, rows)
            self.assertTrue(report["script_resource_checks_passed"])
            self.assertEqual(len(report["recognized_data_files"]), 1)
            label.write_text("#!ascii label test\n2\n0 1 2 3 0\n")
            self.assertFalse(scan_scripts(root, rows)["script_resource_checks_passed"])

    def test_help_requires_unconditional_exit_and_reader(self):
        code = "#!/bin/tcsh\nmri_real\ncat $0 | awk 'BEGINHELP'\nexit 1;\n# comment\nBEGINHELP\nmri_help\n"
        actual, evidence = executable_text(code, "bin/probe")
        self.assertNotIn("mri_help", actual)
        self.assertIn("mri_real", actual)
        self.assertIsNotNone(evidence)
        self.assertEqual(executable_text(code.replace("exit 1;", "echo continue"), "bin/probe")[0],
                         code.replace("exit 1;", "echo continue"))

    def test_version_information_loop_is_a_real_dependency(self):
        code = "set allinfocmds = (\\\n mri_gcut \\\n tkregister2_cmdl \\\n)\nforeach cmd ($allinfocmds)\n $cmd -all-info\nend\n"
        self.assertEqual(mandatory_version_commands(code), ["mri_gcut", "tkregister2_cmdl"])
        self.assertEqual(mandatory_version_commands(code.replace("$cmd -all-info", "echo metadata")), [])

    def test_required_bc_probe_is_checked(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "probe.sh").write_text("#!/bin/sh\nset cmd = (which bc)\n")
            with patch("script_closure.shutil.which", side_effect=lambda name, **kwargs: None if name == "bc" else "/usr/bin/" + name):
                report = scan_scripts(root, [{"path": "probe.sh"}])
            self.assertTrue(any(row.get("command") == "bc" for row in report["errors"]))

    def test_host_utility_check_uses_runtime_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ambient = root / "ambient"
            ambient.mkdir()
            (root / "bin").mkdir()
            command = "codex_fixture_base_utility"
            utility = ambient / command
            utility.write_text("#!/bin/sh\nexit 0\n")
            utility.chmod(0o755)
            (root / "probe.sh").write_text(f"#!/usr/bin/env bash\n{command}\n")
            with patch.dict(os.environ, {"PATH": str(ambient)}), \
                    patch("script_closure.STANDARD_UTILITIES", STANDARD_UTILITIES | {command}), \
                    patch("script_closure.shutil.which", wraps=shutil.which) as which:
                self.assertEqual(shutil.which(command), str(utility))
                which.reset_mock()
                report = scan_scripts(root, [{"path": "probe.sh"}])
                self.assertTrue(any(row.get("command") == command for row in report["errors"]))
                self.assertEqual({call.args[0] for call in which.call_args_list}, {"bash", command})
                for call in which.call_args_list:
                    self.assertEqual(call.kwargs["path"], f"{root / 'bin'}:/usr/bin:/bin")
                shutil.copy2(utility, root / "bin" / command)
                report = scan_scripts(root, [{"path": "probe.sh"}])
                self.assertTrue(report["script_resource_checks_passed"], report["errors"])

    def test_python_check_matches_isolated_dispatch_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directories = [root / "python/packages", root / "python/scripts", root / "ambient"]
            modules = ["codex_fixture_bundle_package", "codex_fixture_bundle_script", "codex_fixture_pythonpath"]
            for directory, module in zip(directories, modules):
                directory.mkdir(parents=True)
                (directory / f"{module}.py").write_text("fixture = True\n")
            (root / "probe.py").write_text("#!/usr/bin/env python3\nimport json\nimport " + ", ".join(modules) + "\n")
            with patch.dict(os.environ, {"PYTHONPATH": str(directories[-1])}):
                report = scan_scripts(root, [{"path": "probe.py"}], script_dispatch=[
                    {"executed_path": "probe.py", "interpreter": "FS_TORCH_PYTHON"}])
            self.assertTrue(report["python_modules"]["json"]["available"])
            self.assertEqual({row["python_module"] for row in report["errors"]}, set(modules))
            self.assertTrue(all(not report["python_modules"][module]["available"] for module in modules))

    def test_csh_python_and_private_python_dispatch_keep_upstream(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "bin").mkdir()
            (root / "libexec").mkdir()
            (root / "libexec/tcsh").write_text("fixture interpreter")
            contents = {"bin/UpdateNeeded": "#!/bin/csh -f\necho ok\n",
                        "bin/csvprint": "#!/usr/bin/env python3\nimport csv\n",
                        "bin/fspython": "#!/bin/bash\nexec /external/python\n"}
            manifest = {"files": [], "script_dispatch": []}
            for relative, text in contents.items():
                path = root / relative
                path.write_text(text)
                manifest["files"].append(file_row(root, path, "script"))
            complete_runtime_dispatch(root, manifest)
            for relative, text in contents.items():
                self.assertEqual((root / "upstream_scripts" / relative).read_text(), text)
            self.assertTrue((root / "bin/tcsh").is_file())
            self.assertTrue((root / "bin/csh").is_file())
            report = scan_scripts(root, manifest["files"], script_dispatch=manifest["script_dispatch"])
            self.assertTrue(report["script_resource_checks_passed"], report["errors"])
            before = len(manifest["files"])
            complete_runtime_dispatch(root, manifest)
            self.assertEqual(len(manifest["files"]), before)

    def test_linux_startup_classification_requires_exact_guard(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            contents = {"sources.csh": '#!/bin/tcsh\nif ("`uname -s`" == "Darwin") then\n  source $FREESURFER_HOME/SetUpFreeSurfer.csh\nendif\n',
                        "SetUpFreeSurfer.csh": "#!/bin/tcsh\nsource $FREESURFER_HOME/FreeSurferEnv.csh\n",
                        "FreeSurferEnv.csh": "#!/bin/tcsh\necho environment\n"}
            manifest = {"files": []}
            for relative, text in contents.items():
                path = root / relative
                path.write_text(text)
                manifest["files"].append(file_row(root, path, "script"))
            classify_linux_startup(root, manifest)
            self.assertTrue(all(row.get("executed") is False for row in manifest["files"][1:]))
            for row in manifest["files"]:
                row.pop("executed", None)
            (root / "sources.csh").write_text(contents["sources.csh"].replace("Darwin", "Linux"))
            classify_linux_startup(root, manifest)
            self.assertTrue(all("executed" not in row for row in manifest["files"]))

    def test_auxiliary_replacements_archive_tf_only_after_all_callers_replaced(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = {"files": [], "neural_replacements": [{"path": "bin/mri_entowm_seg"}]}
            contents = {"libexec/tcsh": "fixture", "bin/mri_mcadura_seg": "#!/bin/tcsh -f\nmri_sclimbic_seg\n",
                        "bin/mri_vsinus_seg": "#!/bin/tcsh -f\nmri_sclimbic_seg\n",
                        "bin/mri_sclimbic_seg": "#!/bin/sh\nexec python helper.py\n",
                        "python/scripts/mri_sclimbic_seg": "#!/usr/bin/env python\nimport tensorflow\n"}
            for resources in AUXILIARY_ASSETS.values():
                contents.update({name: "fixture" for name in resources})
            for relative, text in contents.items():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text)
                manifest["files"].append(file_row(root, path, "script" if relative.startswith(("bin/", "python/")) else "asset"))
            complete_runtime_dispatch(root, manifest)
            replace_auxiliary(root, manifest, ["mcadura"])
            helper = next(row for row in manifest["files"] if row["path"] == "python/scripts/mri_sclimbic_seg")
            self.assertIsNot(helper.get("executed"), False)
            replace_auxiliary(root, manifest, ["vsinus"])
            self.assertIs(helper.get("executed"), False)
            self.assertEqual(len(manifest["neural_replacements"]), 3)
            for row in manifest["files"]:
                self.assertEqual(sha256(root / row["path"]), row["sha256"])


if __name__ == "__main__":
    unittest.main()
