#!/usr/bin/env python3
"""Verify that the FNIT rename changed Python branding and paths only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


BASELINE_COMMIT = "6a408c8e10d2e0c3dcfe49d9ffe651e4da3b7470"
BASELINE_PACKAGE_PREFIX = "src/freesurfer_torch/"
CURRENT_PACKAGE_PREFIX = "src/fnit/"
BASELINE_PACKAGE_SHA256 = (
    "2b51c1203d01651b4aa060b7e96f0a63fa483acfc11078a5f1ddecb2165e8ca3"
)
CURRENT_PACKAGE_SHA256 = (
    "03b6b2df8a020126f1978cae0d7d49b217b7ef80ab8f692028c43c1ba014dbe4"
)
BASELINE_ATTESTATION = "validation/fast_vbm/fnirt_source_equivalence.v0.9.public.json"
BASELINE_ATTESTATION_SHA256 = (
    "65a80f1cf5572371bb5756fbbc834eb902b999fee9195e58eb94bddbe2e60d25"
)
BRAND_REPLACEMENTS = (
    (b"FREESURFER_TORCH", b"FNIT"),
    (b"freesurfer_torch", b"fnit"),
    (b"freesurfer-torch", b"fudan-neuroimaging-toolkit"),
    (b"fs-torch", b"fnit"),
    (b"Weikang-BrainMRI", b"Fudan-Neuroimaging-toolkit"),
)


def _git(repo: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout


def _git_source(repo: Path, commit: str, path: str) -> bytes:
    return _git(repo, "show", f"{commit}:{path}")


def _baseline_files(repo: Path) -> list[str]:
    listing = _git(
        repo,
        "ls-tree",
        "-r",
        "--name-only",
        BASELINE_COMMIT,
        "--",
        BASELINE_PACKAGE_PREFIX.rstrip("/"),
    ).decode()
    return sorted(
        path.removeprefix(BASELINE_PACKAGE_PREFIX)
        for path in listing.splitlines()
        if path.endswith(".py")
    )


def _digest(entries: list[tuple[str, bytes]]) -> str:
    digest = hashlib.sha256()
    for relative, source in sorted(entries):
        digest.update(relative.encode())
        digest.update(source)
    return digest.hexdigest()


def _renamed(source: bytes) -> bytes:
    for old, new in BRAND_REPLACEMENTS:
        source = source.replace(old, new)
    return source


def build_attestation(repo: Path) -> dict:
    package = repo / CURRENT_PACKAGE_PREFIX
    baseline_names = _baseline_files(repo)
    current_names = sorted(
        path.relative_to(package).as_posix() for path in package.rglob("*.py")
    )
    if baseline_names != current_names:
        raise RuntimeError("Python file sets differ beyond the package-directory rename")

    baseline_entries = [
        (
            relative,
            _git_source(repo, BASELINE_COMMIT, BASELINE_PACKAGE_PREFIX + relative),
        )
        for relative in baseline_names
    ]
    current_entries = [
        (relative, (package / relative).read_bytes()) for relative in current_names
    ]
    baseline_digest = _digest(baseline_entries)
    current_digest = _digest(current_entries)
    if baseline_digest != BASELINE_PACKAGE_SHA256:
        raise RuntimeError(f"unexpected baseline digest: {baseline_digest}")
    if current_digest != CURRENT_PACKAGE_SHA256:
        raise RuntimeError(f"unexpected current digest: {current_digest}")

    changed = []
    for (relative, old), (_, new) in zip(baseline_entries, current_entries):
        if _renamed(old) != new:
            raise RuntimeError(f"non-branding Python change: {relative}")
        if old != new:
            changed.append(relative)

    prior_bytes = _git_source(repo, BASELINE_COMMIT, BASELINE_ATTESTATION)
    if hashlib.sha256(prior_bytes).hexdigest() != BASELINE_ATTESTATION_SHA256:
        raise RuntimeError("baseline FNIRT attestation hash differs")
    prior = json.loads(prior_bytes)
    if not prior["checks"]["passed"]:
        raise RuntimeError("baseline FNIRT source attestation did not pass")

    return {
        "schema_version": 2,
        "attestation": "FNIT rename source-equivalence inheritance",
        "baseline": {
            "git_commit": BASELINE_COMMIT,
            "package_import": "freesurfer_torch",
            "package_python_source_sha256": baseline_digest,
            "fnirt_source_attestation_sha256": BASELINE_ATTESTATION_SHA256,
        },
        "final": {
            "package_import": "fnit",
            "package_python_source_sha256": current_digest,
        },
        "python_sources": {
            "file_count": len(current_names),
            "branding_changed_file_count": len(changed),
            "branding_changed_files": changed,
        },
        "checks": {
            "baseline_fnirt_attestation_passed": True,
            "python_file_sets_equal_after_package_rename": True,
            "all_python_sources_identical_after_brand_normalization": True,
            "expected_package_digests": True,
            "passed": True,
        },
        "evidence_inherited_from": [
            "report.v0.9.public.json",
            "fnirt_fsl_10case.v0.9.public.json",
        ],
        "interpretation": {
            "new_numerical_run": False,
            "fsl_numerical_equivalent": False,
            "scope": (
                "The FNIT package inherits the published single-subject numerical "
                "evidence because every Python source file is unchanged after "
                "normalizing the package, distribution, CLI, repository, and weight "
                "configuration names."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    report = json.dumps(build_attestation(repo), indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(report, end="")
    else:
        args.output.write_text(report)


if __name__ == "__main__":
    main()
