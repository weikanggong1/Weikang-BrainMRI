"""Aggregate checks must reject failures that per-element limits can permit."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

from check_comparison_gate import check_report, main


def passing_report():
    checks = {}
    for hemi in ("lh", "rh"):
        for surface in ("white", "pial"):
            checks[f"surf/{hemi}.{surface}"] = {"displacement_mm": {"mae": 0.01, "p99_abs_error": 0.05}}
        checks[f"surf/{hemi}.thickness"] = {"mae": 0.005}
        for atlas in ("aparc", "aparc.DKTatlas", "aparc.a2009s"):
            checks[f"label/{hemi}.{atlas}.annot"] = {"agreement": 0.999}
    for volume in ("aseg.mgz", "aparc+aseg.mgz"):
        checks[f"mri/{volume}"] = {"labels": {"0": {"dice": 0.0}, "1": {"dice": 0.999}, "2": {"dice": 0.999}}}
    return {"passed": True, "checks": checks, "failed_checks": []}


class ComparisonGate(unittest.TestCase):
    def test_exact_boundaries_pass_and_background_is_excluded(self):
        result = check_report(passing_report())
        self.assertTrue(result["passed"], result["failed_checks"])
        self.assertEqual(len(result["checks"]), 19)

    def test_each_aggregate_can_fail_an_otherwise_passing_comparator(self):
        changes = [
            ("surf/lh.white", ("displacement_mm", "mae"), 0.01001),
            ("surf/rh.pial", ("displacement_mm", "p99_abs_error"), 0.05001),
            ("surf/rh.thickness", ("mae",), 0.00501),
            ("label/rh.aparc.a2009s.annot", ("agreement",), 0.99899),
            ("mri/aseg.mgz", ("labels", "2", "dice"), 0.9989),
        ]
        for key, path, value in changes:
            with self.subTest(key=key, path=path):
                report = passing_report()
                target = report["checks"][key]
                for part in path[:-1]:
                    target = target[part]
                target[path[-1]] = value
                result = check_report(report)
                self.assertFalse(result["passed"])
                self.assertEqual(len(result["failed_checks"]), 1)

    def test_missing_nonfinite_and_empty_foreground_fail(self):
        base = passing_report()
        for invalid in (None, float("nan"), float("inf"), -0.1, True, "0.0"):
            with self.subTest(invalid=invalid):
                report = copy.deepcopy(base)
                report["checks"]["surf/lh.white"]["displacement_mm"]["mae"] = invalid
                self.assertFalse(check_report(report)["passed"])
        del base["checks"]["label/lh.aparc.annot"]
        base["checks"]["mri/aseg.mgz"]["labels"] = {"0": {"dice": 1}}
        self.assertEqual(len(check_report(base)["failed_checks"]), 2)

    def test_comparator_failure_cannot_be_overridden(self):
        report = passing_report()
        report["passed"] = False
        report["failed_checks"] = ["stats/aseg.stats"]
        self.assertEqual(check_report(report)["failed_checks"], ["comparator"])

    def test_cli_records_source_digest_and_exit_status(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, output = Path(temporary) / "comparison.json", Path(temporary) / "gate.json"
            report = passing_report()
            source.write_text(json.dumps(report))
            self.assertEqual(main([str(source), "--output", str(output)]), 0)
            saved = json.loads(output.read_text())
            self.assertEqual(len(saved["comparison_sha256"]), 64)
            report["passed"] = False
            source.write_text(json.dumps(report))
            self.assertEqual(main([str(source), "--output", str(output)]), 1)
            self.assertFalse(json.loads(output.read_text())["passed"])


if __name__ == "__main__":
    unittest.main()
