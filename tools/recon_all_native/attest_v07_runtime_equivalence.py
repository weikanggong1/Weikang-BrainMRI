#!/usr/bin/env python3
"""Attest the exact 0.6-to-0.7 recon-all Python source delta, without running it."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re


# Imported by the pinned single-T1 entry, its three neural launchers, or the
# auxiliary segmentation launchers. Include the comparison modules as well.
RUNTIME_FILES = {
    "__init__.py", "weights.py", "_batch_table.py", "_parallel_table.py",
    *(f"recon_all/{name}.py" for name in (
        "__init__", "standalone", "gpu_tools", "aux_seg", "sclimbic",
        "compare_subject", "spatial_vertex_compare")),
    *(f"synthstrip/{name}.py" for name in ("__init__", "model", "pipeline")),
    *(f"synthmorph/{name}.py" for name in ("__init__", "models", "pipeline", "spatial")),
    *(f"synthseg_parc/{name}.py" for name in (
        "__init__", "model", "pipeline", "postprocess", "preprocess", "segment")),
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def package_files(root):
    return {p.relative_to(root).as_posix(): sha256(p) for p in sorted(root.rglob("*.py"))}


def tree_hash(files):
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Expected exactly one occurrence of {old!r}")
    return text.replace(old, new, 1)


def function_ast(source, name):
    nodes = [node for node in ast.parse(source).body
             if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name]
    if len(nodes) != 1:
        raise ValueError(f"Expected one {name} function")
    return ast.dump(nodes[0], include_attributes=False)


def project_version(path):
    section = re.search(r"(?ms)^\[project\]\s*$\n(.*?)(?=^\[|\Z)", path.read_text())
    versions = re.findall(r'(?m)^version\s*=\s*"([^"]+)"\s*$', section[1] if section else "")
    if len(versions) != 1:
        raise ValueError("Expected one [project] version")
    return versions[0]


def attest(old_manifest_path, bundle_manifest_path, old_root, new_root,
           new_manifest_path, new_pyproject_path):
    old_manifest = json.loads(old_manifest_path.read_text())
    bundle_manifest = json.loads(bundle_manifest_path.read_text())
    software = old_manifest["software"]
    if bundle_manifest.get("standalone_verified") is not True:
        raise ValueError("The old bundle is not independently verified")
    verification = bundle_manifest["verification"]
    if verification["software"] != software or \
            verification["evidence"]["code_manifest"]["sha256"] != sha256(old_manifest_path):
        raise ValueError("The frozen code manifest is not linked to the verified bundle")

    old_files, new_files = package_files(old_root), package_files(new_root)
    new_software = json.loads(new_manifest_path.read_text())["software"]
    if old_files != software["package_files_sha256"] or \
            tree_hash(old_files) != software["package_tree_sha256"]:
        raise ValueError("Old source checkout differs from the certified code snapshot")
    if new_files != new_software["package_files_sha256"] or \
            tree_hash(new_files) != new_software["package_tree_sha256"]:
        raise ValueError("New source checkout differs from the v0.7 code snapshot")
    if not RUNTIME_FILES.issubset(old_files.keys() & new_files.keys()):
        raise ValueError("A pinned recon-all runtime module is missing")
    changed_runtime = sorted(name for name in RUNTIME_FILES if old_files[name] != new_files[name])
    if changed_runtime != ["__init__.py", "weights.py"]:
        raise ValueError(f"Unexpected recon-all runtime changes: {changed_runtime}")

    old_init = (old_root / "__init__.py").read_text()
    new_init = (new_root / "__init__.py").read_text()
    expected_init = replace_once(old_init, "__version__ = '0.6.0'", "__version__ = '0.7.0'")
    expected_init = replace_once(
        expected_init, "                'register_gm'):",
        "                'LinearRegistrationResult', 'register_affine', 'register_gm'):")
    if new_init != expected_init:
        raise ValueError("Root package initializer has more than the allowed version/lazy-export change")
    if software["versions"]["freesurfer-torch"] != "0.6.0":
        raise ValueError("Certified package version is not 0.6.0")
    if new_software["versions"]["freesurfer-torch"] != "0.7.0" or \
            project_version(new_pyproject_path) != "0.7.0":
        raise ValueError("v0.7 package snapshot and pyproject versions do not agree")

    old_weights = (old_root / "weights.py").read_text()
    new_weights = (new_root / "weights.py").read_text()
    expected_weights = replace_once(
        old_weights, '"fast-vbm": ("synthstrip.1.pt",),',
        '"fast-vbm": ("synthstrip.1.pt", "synthmorph.deform.3.h5"),')
    if new_weights != expected_weights:
        raise ValueError("Weight resolver has more than the allowed FastVBM asset-list change")
    if function_ast(old_weights, "resolve_weights") != function_ast(new_weights, "resolve_weights"):
        raise ValueError("resolve_weights callable changed")

    changed_package = sorted(name for name in old_files.keys() | new_files.keys()
                             if old_files.get(name) != new_files.get(name))
    other_changed = sorted(set(changed_package) - RUNTIME_FILES)
    if any(name != "cli.py" and not name.startswith("fast_vbm/") for name in other_changed):
        raise ValueError(f"Unexpected non-runtime package changes: {other_changed}")
    return {
        "schema_version": 1,
        "old_code_manifest_sha256": sha256(old_manifest_path),
        "old_verified_bundle_manifest_sha256": sha256(bundle_manifest_path),
        "new_code_manifest_sha256": sha256(new_manifest_path),
        "old_package_tree_sha256": tree_hash(old_files),
        "new_package_tree_sha256": tree_hash(new_files),
        "old_version": "0.6.0", "new_version": "0.7.0",
        "runtime_files_checked": len(RUNTIME_FILES),
        "changed_runtime_files": changed_runtime,
        "other_changed_package_files": other_changed,
        "checks": {
            "certified_old_snapshot_linked": True,
            "old_source_matches_snapshot": True,
            "other_runtime_sources_byte_identical": True,
            "root_initializer_exact_known_delta": True,
            "weights_exact_fast_vbm_asset_delta": True,
            "resolve_weights_ast_identical": True,
            "other_changes_limited_to_cli_and_fast_vbm": True,
            "new_source_matches_v07_snapshot": True,
            "v07_versions_agree": True,
        },
        "boundary": "Source equivalence for the pinned recon-all path; no 0.7 reconstruction or numerical comparison was run.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-code-manifest", type=Path, required=True)
    parser.add_argument("--verified-bundle-manifest", type=Path, required=True)
    parser.add_argument("--old-package-root", type=Path, required=True)
    parser.add_argument("--new-package-root", type=Path, required=True)
    parser.add_argument("--new-code-manifest", type=Path, required=True)
    parser.add_argument("--new-pyproject", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = attest(args.old_code_manifest, args.verified_bundle_manifest,
                    args.old_package_root, args.new_package_root,
                    args.new_code_manifest, args.new_pyproject)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"attested": True, "output": str(args.output),
                      "new_package_tree_sha256": result["new_package_tree_sha256"]}))


if __name__ == "__main__":
    main()
