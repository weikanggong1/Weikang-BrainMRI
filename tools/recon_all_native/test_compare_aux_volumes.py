"""Auxiliary voxel comparison must catch errors hidden by summary metrics."""

import json
from pathlib import Path
import tempfile
import unittest

import nibabel as nib
import numpy as np

from compare_aux_volumes import compare_aux_volumes, main, sha256


class CompareAuxVolumes(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.reference, self.candidate = root / "reference", root / "candidate"
        for subject in (self.reference, self.candidate):
            (subject / "mri").mkdir(parents=True)
            for name in ("ribbon.mgz", "wmparc.mgz"):
                self.write(subject, name, self.data())

    @staticmethod
    def data():
        values = np.zeros((4, 4, 4), dtype=np.int16)
        values[:2] = 2
        values[2:] = 41
        return values

    @staticmethod
    def write(subject, name, data, affine=None):
        nib.save(nib.MGHImage(data, np.eye(4) if affine is None else affine),
                 subject / "mri" / name)

    def test_identical_volumes_pass_with_file_hashes(self):
        report = compare_aux_volumes(self.reference, self.candidate)
        self.assertTrue(report["passed"])
        self.assertEqual(report["failed_checks"], [])
        self.assertEqual(set(report["checks"]), {"mri/ribbon.mgz", "mri/wmparc.mgz"})
        for name, check in report["checks"].items():
            self.assertEqual(check["foreground_macro_dice"], 1)
            self.assertEqual(check["mismatch_count"], 0)
            self.assertEqual(check["reference_sha256"], sha256(self.reference / name))
            self.assertEqual(check["candidate_sha256"], sha256(self.candidate / name))
        json.dumps(report, allow_nan=False)

    def test_changed_voxel_reports_index_and_fails_label_dice(self):
        values = self.data()
        values[1, 2, 3] = 41
        self.write(self.candidate, "ribbon.mgz", values)
        report = compare_aux_volumes(self.reference, self.candidate)
        check = report["checks"]["mri/ribbon.mgz"]
        self.assertFalse(report["passed"])
        self.assertEqual(report["failed_checks"], ["mri/ribbon.mgz"])
        self.assertEqual(check["outlier_voxels"], [[1, 2, 3]])
        self.assertEqual(check["mismatch_count"], 1)
        self.assertLess(check["labels"]["2"]["dice"], 0.995)

    def test_new_label_fails_even_if_only_one_voxel(self):
        values = self.data()
        values[0, 0, 0] = 1001
        self.write(self.candidate, "wmparc.mgz", values)
        check = compare_aux_volumes(self.reference, self.candidate)["checks"]["mri/wmparc.mgz"]
        self.assertEqual(check["status"], "failed")
        self.assertFalse(check["labels"]["1001"]["present_in_both"])

    def test_geometry_and_noninteger_labels_fail(self):
        affine = np.diag([2, 1, 1, 1]).astype(float)
        self.write(self.candidate, "ribbon.mgz", self.data(), affine)
        values = self.data().astype(np.float32)
        values[1, 2, 3] = 2.5
        self.write(self.candidate, "wmparc.mgz", values)
        report = compare_aux_volumes(self.reference, self.candidate)
        self.assertEqual(set(report["failed_checks"]), set(report["checks"]))
        self.assertFalse(report["checks"]["mri/ribbon.mgz"]["affine_equal"])
        self.assertFalse(report["checks"]["mri/ribbon.mgz"]["header_zooms_equal"])
        self.assertIn("finite integers", report["checks"]["mri/wmparc.mgz"]["reason"])

    def test_missing_volume_exits_nonzero_and_writes_report(self):
        (self.candidate / "mri/wmparc.mgz").unlink()
        output = Path(self.temporary.name) / "out.json"
        self.assertEqual(main([str(self.reference), str(self.candidate), "--output", str(output)]), 1)
        report = json.loads(output.read_text())
        self.assertEqual(report["failed_checks"], ["mri/wmparc.mgz"])
        self.assertEqual(report["checks"]["mri/wmparc.mgz"]["status"], "missing")


if __name__ == "__main__":
    unittest.main()
