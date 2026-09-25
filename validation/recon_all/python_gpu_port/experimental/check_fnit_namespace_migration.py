"""Attest a mechanical recon-all port from freesurfer_torch to fnit."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-root", type=Path, required=True)
    parser.add_argument("--new-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    paths = subprocess.check_output(
        ["git", "ls-files", "-z", "--others", "--exclude-standard",
         "src/freesurfer_torch/recon_all", "tests/recon_all"],
        cwd=args.old_root,
    ).decode().split("\0")
    rows = []
    for name in paths:
        if not name or not name.endswith(".py"):
            continue
        relative = Path(name)
        mapped = (Path("src", "fnit", *relative.parts[2:])
                  if relative.parts[:2] == ("src", "freesurfer_torch") else relative)
        original = (args.old_root / relative).read_bytes()
        expected = original.replace(b"freesurfer_torch", b"fnit")
        if mapped.name in ("assets.py", "test_assets.py"):
            expected = expected.replace(b"FREESURFER_TORCH_ASSETS", b"FNIT_ASSETS")
        actual = (args.new_root / mapped).read_bytes()
        rows.append({"old": name, "new": str(mapped),
                     "old_sha256": sha256(original), "new_sha256": sha256(actual),
                     "exact_expected_transform": expected == actual})
    report = {"scope": "new Python source and tests only",
              "transform": "freesurfer_torch -> fnit; assets.py and test_assets.py also rename the asset environment variable",
              "checked": len(rows), "exact": sum(row["exact_expected_transform"] for row in rows),
              "files": rows}
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{report['exact']}/{report['checked']} transformed files exact")
    if report["exact"] != report["checked"]:
        raise AssertionError("namespace migration has additional source edits")


if __name__ == "__main__":
    main()
