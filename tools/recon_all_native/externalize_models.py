#!/usr/bin/env python3
"""Turn a copy of a certified recon-all bundle into an unverified external-model candidate."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tempfile

from preflight_bundle import MODEL_FILES

MODEL_COUNT = len(MODEL_FILES)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_model(path, row, root):
    require(not path.is_symlink() and path.resolve().is_relative_to(root)
            and stat.S_ISREG(path.stat().st_mode), f"Model is not a local regular file: {path}")
    require(path.stat().st_size == row["bytes"] and sha256(path) == row["sha256"],
            f"Model size or SHA-256 differs from certified manifest: {path}")


def prepare(certified_manifest, bundle_copy, models_dir):
    source = certified_manifest.resolve(strict=True)
    bundle = bundle_copy.resolve(strict=True)
    target = bundle / "manifest.json"
    external = models_dir.resolve(strict=True)
    require(source != target and source.parent != bundle,
            "The certified bundle itself cannot be converted")
    require(external.is_dir() and not external.is_relative_to(bundle)
            and not external.is_relative_to(source.parent),
            "Models directory must be outside both bundles")
    original = source.read_bytes()
    require(target.read_bytes() == original,
            "Bundle copy manifest differs from the certified source")
    manifest = json.loads(original)
    if manifest.get("fs_home"):
        require(not external.is_relative_to(Path(manifest["fs_home"]).resolve()),
                "Models directory cannot be inside the source FreeSurfer installation")
    require(manifest.get("standalone_verified") is True
            and isinstance(manifest.get("verification"), dict),
            "Source bundle is not certified")
    require(isinstance(manifest.get("files"), list)
            and isinstance(manifest.get("required_resources"), list),
            "Certified manifest lacks file or resource inventory")

    models = []
    for row in manifest["files"]:
        name = row["path"]
        if not name.startswith("models/"):
            continue
        parts = PurePosixPath(name).parts
        require(len(parts) == 2 and parts[0] == "models"
                and name == f"models/{parts[1]}" and parts[1] not in (".", ".."),
                f"Unsafe model path: {name}")
        require(type(row.get("bytes")) is int and row["bytes"] >= 0
                and isinstance(row.get("sha256"), str)
                and re.fullmatch(r"[0-9a-f]{64}", row["sha256"])
                and row.get("staged") is True,
                f"Model row lacks certified size, SHA-256 or staging: {name}")
        models.append({key: row[key] for key in ("path", "bytes", "sha256")})
    names = {row["path"] for row in models}
    require(len(models) == len(names) == MODEL_COUNT
            and names == {f"models/{name}" for name in MODEL_FILES},
            f"Expected exactly {MODEL_COUNT} fixed certified model files")
    model_dir = bundle / "models"
    require(model_dir.is_dir() and not model_dir.is_symlink(),
            "Bundle copy lacks a regular models directory")
    require({p.name for p in model_dir.iterdir()} == {Path(name).name for name in names},
            "Bundle models directory contains missing or untracked files")
    for row in models:
        filename = Path(row["path"]).name
        verify_model(bundle / row["path"], row, bundle)
        verify_model(external / filename, row, external)

    required = manifest["required_resources"]
    require(all(not name.startswith("models/") or name in names for name in required),
            "Required resources contain a model absent from the certified inventory")
    source_verification = manifest.pop("verification")
    manifest["files"] = [row for row in manifest["files"] if row["path"] not in names]
    manifest["required_resources"] = [name for name in required if name not in names]
    manifest["external_models"] = sorted(models, key=lambda row: row["path"])
    manifest["externalization"] = {
        "schema_version": 1,
        "converted_utc": datetime.now(timezone.utc).isoformat(),
        "source_certified_bundle_manifest_sha256": hashlib.sha256(original).hexdigest(),
        "models_directory": str(external),
        "requires_recertification": True,
        "source_verification": source_verification,
        "boundary": "Candidate only: external-model path closure, clean reconstruction and numerical parity remain unverified.",
    }
    manifest["standalone_verified"] = False
    manifest["candidate_bundle"] = True
    return target, original, manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--certified-manifest", type=Path, required=True)
    parser.add_argument("--bundle-copy", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        target, original, manifest = prepare(
            args.certified_manifest, args.bundle_copy, args.models_dir)
        if not args.check_only:
            require(target.read_bytes() == original,
                    "Bundle copy manifest changed during validation")
            with tempfile.NamedTemporaryFile("w", dir=target.parent,
                                             prefix=".manifest-external-", suffix=".json",
                                             delete=False) as stream:
                temporary = Path(stream.name)
                json.dump(manifest, stream, indent=2, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
            for row in manifest["external_models"]:
                (target.parent / row["path"]).unlink()
        print(json.dumps({"eligible": True, "written": not args.check_only,
                          "manifest": str(target), "external_models": MODEL_COUNT,
                          "model_payload_bytes": sum(row["bytes"] for row in manifest["external_models"]),
                          "source_manifest_sha256": manifest["externalization"]["source_certified_bundle_manifest_sha256"]}))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps({"eligible": False, "error": str(error)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
