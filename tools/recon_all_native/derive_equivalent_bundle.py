#!/usr/bin/env python3
"""Bind a copied 0.6 certified bundle to an equivalent 0.7 recon-all release.

This derives a release manifest from existing certification and exact source
equivalence. It does not assert that a 0.7 reconstruction was run.
"""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import stat
import tempfile

from attest_v07_runtime_equivalence import attest, package_files, sha256, tree_hash


def require(condition, message):
    if not condition:
        raise ValueError(message)


def derive(args):
    source_path = args.certified_bundle_manifest.resolve(strict=True)
    copy_path = args.bundle_copy.resolve(strict=True) / "manifest.json"
    require(source_path != copy_path, "The release bundle must be a separate copy")
    original = source_path.read_bytes()
    require(copy_path.read_bytes() == original, "The release bundle copy no longer matches the certified manifest")

    report = json.loads(args.attestation.read_text())
    recomputed = attest(args.old_code_manifest, source_path,
                        args.old_package_root, args.new_package_root,
                        args.new_code_manifest, args.new_pyproject)
    require(report == recomputed, "The source-equivalence attestation is stale or altered")
    require(report["old_verified_bundle_manifest_sha256"] == sha256(source_path),
            "Attestation refers to another certified bundle")
    require(report["new_code_manifest_sha256"] == sha256(args.new_code_manifest),
            "Attestation refers to another 0.7 code snapshot")

    frozen = json.loads(args.new_code_manifest.read_text())
    software = frozen["software"]
    new_files = package_files(args.new_package_root)
    require(frozen["bundle_manifest_sha256"] == sha256(source_path),
            "The 0.7 code snapshot was frozen against another bundle manifest")
    require(software["package_files_sha256"] == new_files
            and software["package_tree_sha256"] == tree_hash(new_files)
            and software["package_tree_sha256"] == report["new_package_tree_sha256"],
            "The 0.7 code snapshot differs from the release source")
    require(Path(software["package_root"]).resolve() == args.new_package_root.resolve()
            and software["versions"]["fudan-neuroimaging-toolkit"] == report["new_version"]
            and software["cuda_available"] is True,
            "The frozen 0.7 CUDA installation does not match the release source")

    source = json.loads(original)
    certified = source["verification"]
    old_software = certified["software"]
    require(source["standalone_verified"] is True
            and old_software["package_tree_sha256"] == report["old_package_tree_sha256"],
            "The source bundle lacks the expected 0.6 certification")
    for name in ("python", "platform", "torch_cuda", "cudnn", "cuda_available"):
        require(software[name] == old_software[name], f"Runtime environment changed: {name}")
    old_dependencies = {key: value for key, value in old_software["versions"].items()
                        if key != "fudan-neuroimaging-toolkit"}
    new_dependencies = {key: value for key, value in software["versions"].items()
                        if key != "fudan-neuroimaging-toolkit"}
    require(new_dependencies == old_dependencies, "Runtime dependency versions changed")

    derived = deepcopy(source)
    derived["verification"] = {
        "schema_version": 2,
        "certification_kind": "source_equivalent_release",
        "profile_id": certified["profile_id"],
        "derived_utc": datetime.now(timezone.utc).isoformat(),
        "software": software,
        "source_certification": {
            "bundle_manifest_sha256": sha256(source_path),
            "verification": certified,
        },
        "release_equivalence": {
            "attestation_sha256": sha256(args.attestation),
            "attestation": report,
            "new_code_manifest_sha256": sha256(args.new_code_manifest),
            "boundary": "No 0.7 reconstruction or numerical comparison was run; 0.6 certification is inherited only for the attested equivalent recon-all path.",
        },
    }
    return copy_path, original, derived


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("certified-bundle-manifest", "bundle-copy", "old-code-manifest",
                 "old-package-root", "new-package-root", "attestation", "new-code-manifest",
                 "new-pyproject"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        path, original, manifest = derive(args)
        if not args.check_only:
            require(path.read_bytes() == original, "The release bundle changed during derivation")
            with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=".manifest-derived-",
                                             suffix=".json", delete=False) as stream:
                temporary = Path(stream.name)
                json.dump(manifest, stream, indent=2, allow_nan=False)
                stream.write("\n")
            try:
                os.chmod(temporary, stat.S_IMODE(path.stat().st_mode))
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
        print(json.dumps({"eligible": True, "written": not args.check_only,
                          "manifest": str(path),
                          "source_manifest_sha256": sha256(args.certified_bundle_manifest),
                          "attestation_sha256": sha256(args.attestation)}))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps({"eligible": False, "error": str(error)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
