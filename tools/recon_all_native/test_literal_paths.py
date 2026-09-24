"""Regression fixtures for source wrappers pointing at ../fspython."""

from pathlib import Path
import tempfile
import unittest

from package_candidate import expand_literal_script_files
from script_closure import scan_scripts


class LiteralResourcePaths(unittest.TestCase):
    def test_parent_paths_are_normalized_or_reported_without_escaping(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "suite"
            (root / "bin").mkdir(parents=True)
            (root / "models").mkdir()
            (root / "models/model.h5").write_text("inside")
            (base / "fspython").mkdir()
            (base / "fspython/helper.py").write_text("outside")
            (root / "models/link.h5").symlink_to(base / "fspython/helper.py")
            (root / "bin/probe").write_text(
                '#!/bin/sh\n'
                'cat "$FREESURFER_HOME/models/../models/model.h5"\n'
                'cat "$FREESURFER_HOME/../fspython/helper.py"\n'
                'cat "$FREESURFER_HOME/models/link.h5"\n')
            selected, outside = expand_literal_script_files(root, {"bin/probe"})
            self.assertEqual(selected, {"bin/probe", "models/model.h5"})
            self.assertEqual({row["reference"] for row in outside},
                             {"../fspython/helper.py", "models/link.h5"})
            closure = scan_scripts(root, [{"path": "bin/probe"}])
            self.assertFalse(closure["script_resource_checks_passed"])
            self.assertEqual(
                {row["resource"] for row in closure["errors"]
                 if row["error"] == "Literal resource resolves outside the package"},
                {"../fspython/helper.py", "models/link.h5"})


if __name__ == "__main__":
    unittest.main()
