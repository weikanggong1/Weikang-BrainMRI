"""Promotion rejects missing, stale and cross-run evidence before any mutation."""

import argparse
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import shlex
import sys
import unittest
from unittest.mock import patch

from audit_bundle import sha256
from check_comparison_gate import check_report
import promote_bundle as promotion
from test_comparison_gate import passing_report
from test_single_t1_scope import scoped_fixture


@contextmanager
def evidence_fixture():
    with scoped_fixture() as (bundle, manifest, _):
        root = bundle.parent
        manifest.update(candidate_bundle=True, fs_home=str(root / "FreeSurfer"))
        manifest_path = bundle / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))
        candidate = root / "subjects/sub01"
        for name in promotion.required_checks() | {"mri/orig.mgz", "mri/ribbon.mgz", "mri/wmparc.mgz", "surf/lh.sphere.reg", "surf/rh.sphere.reg"}:
            path = candidate / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(name)
        config = candidate / "scripts/recon-config.yaml"
        config.parent.mkdir(parents=True)
        config.write_bytes((bundle / "etc/scoped-reference-recon-config.yaml").read_bytes())
        image, license_file = root / "t1.nii.gz", root / "FreeSurfer/license.txt"
        image.write_text("T1")
        license_file.parent.mkdir()
        license_file.write_text("private fixture key")
        preflight = {"hash_and_linkage_passed": True, "script_resource_checks_passed": True,
                     "errors": [], "script_resource_closure": {"errors": []},
                     "checked_staged_files": len(manifest["files"])}
        software = {"python": "3.11", "versions": {"torch": "2.5.1"},
                    "package_files_sha256": {"recon_all/gpu_tools.py": "frozen hash"}}
        now = datetime.now(timezone.utc)
        run = {"bundle": str(bundle), "subject_dir": str(candidate), "subject": "sub01",
               "return_code": 0, "missing_outputs": [], "input_sha256": sha256(image),
               "threads": 4, "itk_threads": 1, "parallel_hemispheres": True, "device": "cuda:1",
               "started_utc": now.isoformat(), "elapsed_seconds": 4000,
               "effective_config_matches_profile": True, "effective_config_sha256": sha256(config)}
        comparison = passing_report()
        comparison.update(candidate=str(candidate), reference=str(root / "reference"), tolerances={})
        for name in promotion.required_checks():
            comparison["checks"].setdefault(name, {}).update(status="passed")
            if name.startswith("label/") and name.endswith(".annot") or name.startswith("mri/"):
                comparison["checks"][name]["min_dice"] = 0.995
        args = argparse.Namespace(bundle=bundle, package_python=Path(sys.executable), trace_cwd=root,
                                  input=image, license_file=license_file, check_only=False)
        reports = {"preflight": preflight, "run": run, "comparison": comparison, "tolerances": {},
                   "code_manifest": {"captured_utc": (now - timedelta(seconds=10)).isoformat(),
                                     "bundle_manifest_sha256": sha256(manifest_path), "software": software}}
        for name, report in reports.items():
            path = root / (name + ".json")
            path.write_text(json.dumps(report))
            setattr(args, name, path)
        args.aggregate = root / "aggregate.json"
        aggregate = check_report(comparison)
        aggregate["comparison_sha256"] = sha256(args.comparison)
        args.aggregate.write_text(json.dumps(aggregate))
        args.trace = root / "trace.log"
        args.trace.write_text(f'1 execve("{bundle}/bin/recon-all", ["recon-all", "-s", "sub01", "-sd", "{candidate.parent}", "-i", "{image}"], 0x1) = 0\n'
                              f'1 openat(AT_FDCWD, "{license_file}", O_RDONLY) = 3\n')
        args.trace_command_file = root / "command.txt"
        command = ["env", "-i", "HOME=$HOME", "strace", "-f", "-qq", "-e", "trace=file,process", "-s", "256",
                   "-o", str(args.trace), "fs-torch-recon-all", "-i", str(image), "-s", "sub01", "-sd", str(candidate.parent),
                   "--bundle", str(bundle), "--license", str(license_file), "--device", "cuda:1", "--threads", "4", "--development-bundle"]
        args.trace_command_file.write_text(shlex.join(command))
        yield args, manifest, preflight, software


class PromoteBundle(unittest.TestCase):
    def test_linked_evidence_passes_without_mutating_manifest(self):
        with evidence_fixture() as (args, manifest, preflight, software):
            result = promotion.validate(args, manifest, preflight, software)
            self.assertEqual(len(result["candidate_outputs_sha256"]), 57)
            self.assertTrue(result["trace_audit"]["passed"])
            self.assertIs(json.loads((args.bundle / "manifest.json").read_text())["standalone_verified"], False)

    def test_missing_failed_or_stale_reports_fail(self):
        cases = [("comparison", lambda r: r["checks"].pop("stats/synthseg.vol.csv")),
                 ("comparison", lambda r: r.update(candidate="/another/subject")),
                 ("aggregate", lambda r: r.update(comparison_sha256="0" * 64)),
                 ("preflight", lambda r: r.update(checked_staged_files=1)),
                 ("run", lambda r: r.update(return_code=1)),
                 ("run", lambda r: r.update(effective_config_matches_profile=False)),
                 ("code_manifest", lambda r: r.update(bundle_manifest_sha256="0" * 64)),
                 ("code_manifest", lambda r: r["software"].update(python="different")),
                 ("code_manifest", lambda r: r.update(captured_utc="2999-01-01T00:00:00+00:00"))]
        for name, change in cases:
            with self.subTest(name=name), evidence_fixture() as (args, manifest, preflight, software):
                path = getattr(args, name)
                report = json.loads(path.read_text())
                change(report)
                path.write_text(json.dumps(report))
                with self.assertRaises(ValueError):
                    promotion.validate(args, manifest, preflight, software)

    def test_visible_external_access_and_unfinished_trace_fail(self):
        for extra in ('2 openat(AT_FDCWD, "/opt/fsl/data/atlas.nii", O_RDONLY) = 3\n',
                      '2 openat(AT_FDCWD, "/opt/FreeSurfer/license.txt", O_RDONLY) = 3\n',
                      '2 openat(AT_FDCWD, "/env/site-packages/tensorflow/model.py", O_RDONLY) = 4\n',
                      '2 execve("/opt/FreeSurfer/bin/mri_info", [], 0x1) = 0\n',
                      '2 openat(AT_FDCWD, "/opt/fsl/data.nii", O_RDONLY <unfinished ...>\n',
                      '2 openat(AT_FDCWD, "/long/path"..., O_RDONLY) = 4\n'):
            with self.subTest(extra=extra), evidence_fixture() as (args, manifest, preflight, software):
                with args.trace.open("a") as stream:
                    stream.write(extra)
                with self.assertRaises(ValueError):
                    promotion.validate(args, manifest, preflight, software)

    def test_other_subject_trace_and_changed_inventory_fail(self):
        with evidence_fixture() as (args, manifest, preflight, software):
            original = args.trace.read_text()
            args.trace.write_text(original.replace('"sub01"', '"other-subject"'))
            with self.assertRaisesRegex(ValueError, "matching this subject"):
                promotion.validate(args, manifest, preflight, software)
            args.trace.write_text(original)
            (args.bundle / "bin/untracked-tool").write_text("unexpected")
            with self.assertRaisesRegex(ValueError, "Untracked bundle file"):
                promotion.validate(args, manifest, preflight, software)

    def test_failed_external_probe_and_argv_mentions_are_not_accesses(self):
        with evidence_fixture() as (args, manifest, preflight, software):
            with args.trace.open("a") as stream:
                stream.write('2 openat(AT_FDCWD, "/opt/fsl/absent", O_RDONLY) = -1 ENOENT\n'
                             '2 execve("/bin/echo", ["echo", "/opt/FreeSurfer"], 0x1) = 0\n'
                             '3 openat(AT_FDCWD, "/project/freesurfer_torch/module.py", O_RDONLY <unfinished ...>\n'
                             '3 <... openat resumed>) = 5\n'
                             '4 openat(AT_FDCWD, "/env/site-packages/nibabel/freesurfer/io.py", O_RDONLY) = 6\n')
            self.assertTrue(promotion.validate(args, manifest, preflight, software)["trace_audit"]["passed"])

    def test_successful_promotion_records_original_and_report_hashes(self):
        with evidence_fixture() as (args, manifest, preflight, software):
            def preflight_run(command, **kwargs):
                Path(command[-1]).write_text(json.dumps(preflight))
            argv = [value for key in ("bundle", "preflight", "run", "comparison", "aggregate", "trace", "trace_command_file",
                                      "trace_cwd", "package_python", "input", "license_file", "code_manifest", "tolerances")
                    for value in ("--" + key.replace("_", "-"), str(getattr(args, key)))]
            with patch.object(promotion, "collect_software", return_value=software), \
                    patch.object(promotion.subprocess, "run", side_effect=preflight_run):
                self.assertEqual(promotion.main(argv), 0)
            promoted = json.loads((args.bundle / "manifest.json").read_text())
            self.assertIs(promoted["standalone_verified"], True)
            self.assertEqual(promoted["verification"]["evidence"]["comparison"]["sha256"], sha256(args.comparison))
            self.assertEqual(promoted["verification"]["original_manifest_sha256"],
                             sha256(args.bundle / "metadata/manifest.before-promotion.json"))


if __name__ == "__main__":
    unittest.main()
