#!/usr/bin/env python3
"""Verify that the FNIRT implementation is unchanged after source relocation."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess


BASELINE_COMMIT = "aece9c7e4cc2c030a8ebb925c4d670f7c841a8ea"
BASELINE_PACKAGE_SHA256 = (
    "ab87e9fdc53f1c70991fd038b5abef0cebff6b288a76a1d202bd9756198c325c"
)
FINAL_PACKAGE_SHA256 = (
    "8bea5a883d0043a5d1264994dddf0fcbdb88fbdf3c1d22a753107754b24b6438"
)
PACKAGE_PREFIX = "src/freesurfer_torch/"
FNIRT_PREFIX = f"{PACKAGE_PREFIX}fnirt/"
RELOCATED_IMPORTS = {
    "registration.py": (
        "fast_vbm.linear",
        "flirt.coordinates",
        ("voxel_to_fsl_scaled_mm", "world_to_flirt_affine"),
    ),
    "standalone.py": (
        "fast_vbm.linear",
        "flirt.coordinates",
        ("flirt_to_world_affine",),
    ),
}
COORDINATE_FUNCTIONS = (
    "voxel_to_fsl_scaled_mm",
    "flirt_to_world_affine",
    "world_to_flirt_affine",
)
COORDINATE_HELPERS = ("_numpy_affine",)


def _git(repo: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout


def _git_source(repo: Path, commit: str, path: str) -> bytes:
    return _git(repo, "show", f"{commit}:{path}")


def _baseline_python_files(repo: Path, commit: str) -> list[str]:
    listing = _git(
        repo,
        "ls-tree",
        "-r",
        "--name-only",
        commit,
        "--",
        PACKAGE_PREFIX.rstrip("/"),
    ).decode("utf-8")
    return sorted(path for path in listing.splitlines() if path.endswith(".py"))


def _package_digest_from_git(repo: Path, commit: str) -> str:
    digest = hashlib.sha256()
    for path in _baseline_python_files(repo, commit):
        digest.update(path.removeprefix(PACKAGE_PREFIX).encode("utf-8"))
        digest.update(_git_source(repo, commit, path))
    return digest.hexdigest()


def _package_digest_from_worktree(package: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(package.rglob("*.py")):
        digest.update(path.relative_to(package).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


class _NormalizeCoordinateImport(ast.NodeTransformer):
    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.ImportFrom:
        node = self.generic_visit(node)
        if node.level == 2 and node.module in {
            "fast_vbm.linear",
            "flirt.coordinates",
        }:
            node.module = "SOURCE_RELOCATION.coordinate_functions"
        return node


def _normalized_module_ast(source: bytes) -> str:
    tree = ast.parse(source.decode("utf-8"))
    tree = _NormalizeCoordinateImport().visit(tree)
    ast.fix_missing_locations(tree)
    return ast.dump(tree, annotate_fields=True, include_attributes=False)


def _definition_ast(source: bytes, name: str) -> str:
    matches = [
        node
        for node in ast.parse(source.decode("utf-8")).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one top-level definition named {name!r}")
    return ast.dump(matches[0], annotate_fields=True, include_attributes=False)


def _imported_names(source: bytes, module: str) -> tuple[str, ...]:
    matches = [
        tuple(alias.name for alias in node.names)
        for node in ast.walk(ast.parse(source.decode("utf-8")))
        if isinstance(node, ast.ImportFrom)
        and node.level == 2
        and node.module == module
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one relative import from {module!r}")
    return matches[0]


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def build_attestation(repo: Path) -> dict:
    package = repo / PACKAGE_PREFIX
    baseline_digest = _package_digest_from_git(repo, BASELINE_COMMIT)
    final_digest = _package_digest_from_worktree(package)
    if baseline_digest != BASELINE_PACKAGE_SHA256:
        raise RuntimeError(
            f"baseline package digest is {baseline_digest}, expected "
            f"{BASELINE_PACKAGE_SHA256}"
        )
    if final_digest != FINAL_PACKAGE_SHA256:
        raise RuntimeError(
            f"final package digest is {final_digest}, expected {FINAL_PACKAGE_SHA256}"
        )

    baseline_files = [
        path
        for path in _baseline_python_files(repo, BASELINE_COMMIT)
        if path.startswith(FNIRT_PREFIX)
    ]
    current_files = sorted(
        f"{FNIRT_PREFIX}{path.relative_to(package / 'fnirt').as_posix()}"
        for path in (package / "fnirt").rglob("*.py")
    )
    if baseline_files != current_files:
        raise RuntimeError("baseline and final FNIRT Python file sets differ")

    byte_identical = []
    relocations = []
    for path in baseline_files:
        name = Path(path).name
        old = _git_source(repo, BASELINE_COMMIT, path)
        new = (repo / path).read_bytes()
        if name not in RELOCATED_IMPORTS:
            if old != new:
                raise RuntimeError(f"unexpected FNIRT source change: {path}")
            byte_identical.append(path.removeprefix(PACKAGE_PREFIX))
            continue

        old_module, new_module, expected_names = RELOCATED_IMPORTS[name]
        if _imported_names(old, old_module) != expected_names:
            raise RuntimeError(f"unexpected baseline import in {path}")
        if _imported_names(new, new_module) != expected_names:
            raise RuntimeError(f"unexpected final import in {path}")
        old_ast = _normalized_module_ast(old)
        new_ast = _normalized_module_ast(new)
        if old_ast != new_ast:
            raise RuntimeError(f"normalized AST differs: {path}")
        relocations.append(
            {
                "file": path.removeprefix(PACKAGE_PREFIX),
                "baseline_module": old_module,
                "final_module": new_module,
                "imported_names": list(expected_names),
                "normalized_ast_sha256": _sha256_text(old_ast),
            }
        )

    old_coordinates = _git_source(
        repo,
        BASELINE_COMMIT,
        f"{PACKAGE_PREFIX}fast_vbm/linear.py",
    )
    new_coordinates = (package / "flirt" / "coordinates.py").read_bytes()
    definitions = {}
    for name in (*COORDINATE_FUNCTIONS, *COORDINATE_HELPERS):
        old_ast = _definition_ast(old_coordinates, name)
        new_ast = _definition_ast(new_coordinates, name)
        if old_ast != new_ast:
            raise RuntimeError(f"relocated coordinate definition differs: {name}")
        definitions[name] = _sha256_text(old_ast)

    return {
        "schema_version": 1,
        "attestation": "FNIRT source-equivalence inheritance",
        "baseline": {
            "git_commit": BASELINE_COMMIT,
            "package_python_source_sha256": baseline_digest,
        },
        "final": {
            "package_python_source_sha256": final_digest,
        },
        "fnirt_python_sources": {
            "file_count": len(baseline_files),
            "byte_identical_files": byte_identical,
            "import_relocations": relocations,
        },
        "coordinate_source_relocation": {
            "baseline_file": "fast_vbm/linear.py",
            "final_file": "flirt/coordinates.py",
            "functions": list(COORDINATE_FUNCTIONS),
            "support_definitions": list(COORDINATE_HELPERS),
            "definition_ast_sha256": definitions,
        },
        "checks": {
            "expected_package_digests": True,
            "fnirt_python_file_sets_equal": True,
            "unchanged_fnirt_files_byte_identical": True,
            "import_relocations_normalized_ast_identical": True,
            "coordinate_definitions_ast_identical": True,
            "passed": True,
        },
        "evidence_inherited_from": "fnirt_fsl_10case.v0.9.public.json",
        "interpretation": {
            "new_numerical_run": False,
            "fsl_numerical_equivalent": False,
            "scope": (
                "The final package inherits the baseline FNIRT numerical results "
                "because FNIRT executable source is unchanged apart from verified "
                "import and coordinate-function relocation. This attestation is "
                "not a new numerical run."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="write the public JSON here; otherwise print it to stdout",
    )
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    report = json.dumps(build_attestation(repo), indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(report, end="")
    else:
        args.output.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
