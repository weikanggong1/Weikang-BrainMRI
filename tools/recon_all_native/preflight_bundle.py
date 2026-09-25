#!/usr/bin/env python3
"""Check a staged bundle's hashes and relocated ELF linkage without source FS.

This check does not execute a reconstruction or establish resource/runtime
license closure. It never reads from the reference installation.
"""

import argparse
import json
from pathlib import Path
import re
import subprocess
import platform

from audit_bundle import SYSTEM_LIBRARIES, sha256
from script_closure import scan_scripts


MODEL_FILES = (
    "synthstrip.1.pt", "synthmorph.affine.2.h5", "synthmorph.deform.3.h5",
    "synthseg_2.0.h5", "synthseg_segmentation_labels_2.0.npy",
    "synthseg_segmentation_names_2.0.npy", "synthseg_topological_classes_2.0.npy",
    "entowm.fsm31.t1.nstd00-30.nstd21-108.h5", "entowm.ctab",
    "mca-dura.both-lh.nstd21.fhs.h5", "mca-dura.ctab",
    "vsinus.no-sp.m.all.nstd10-070.h5", "sclimbic.volstats.csv",
)


def under(path, root):
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def check_external_models(rows, models_dir):
    if not isinstance(rows, list):
        return [{"error": "external_models must be a list"}], 0
    paths = [row.get("path") if isinstance(row, dict) else None for row in rows]
    expected = {f"models/{name}" for name in MODEL_FILES}
    if (len(paths) != len(expected) or
            not all(isinstance(path, str) for path in paths) or set(paths) != expected):
        return [{"error": "Invalid external model inventory: expected 13 unique fixed model names"}], 0
    if models_dir is None:
        return [{"error": "External models require --models-dir"}], 0
    try:
        root = models_dir.resolve(strict=True)
        if not root.is_dir():
            raise NotADirectoryError(root)
    except (OSError, RuntimeError):
        return [{"error": "Invalid --models-dir", "path": str(models_dir)}], 0

    errors = []
    checked = 0
    for row in rows:
        relative = row["path"]  # The fixed inventory above allows only flat models/name paths.
        if type(row.get("bytes")) is not int or row["bytes"] < 0 or not (
                isinstance(row.get("sha256"), str) and
                re.fullmatch(r"[0-9a-f]{64}", row["sha256"])):
            errors.append({"file": relative, "error": "Invalid external model size or SHA-256"})
            continue
        try:
            candidate = root / relative.removeprefix("models/")
            if candidate.is_symlink():
                raise ValueError("symlink")
            path = candidate.resolve(strict=True)
            if not under(path, root) or not path.is_file():
                raise ValueError("escape or non-file")
            size = path.stat().st_size
            digest = sha256(path)
        except (OSError, RuntimeError, ValueError):
            errors.append({"file": relative, "error": "Missing, symlinked or escaped external model path"})
            continue
        checked += 1
        if size != row["bytes"]:
            errors.append({"file": relative, "error": "External model size mismatch",
                           "expected": row["bytes"], "actual": size})
        if digest != row["sha256"]:
            errors.append({"file": relative, "error": "External model hash mismatch"})
    return errors, checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--package-python", type=Path,
                        help="Python interpreter used by the installed fudan-neuroimaging-toolkit package")
    parser.add_argument("--models-dir", type=Path,
                        help="Directory containing files listed as models/name in external_models")
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    manifest = json.loads((bundle / "manifest.json").read_text())
    errors = []
    model_errors, checked_models = (check_external_models(manifest["external_models"], args.models_dir)
                                    if "external_models" in manifest else ([], 0))
    errors.extend(model_errors)
    if "external_models" in manifest and args.models_dir is not None and manifest.get("fs_home"):
        if args.models_dir.resolve().is_relative_to(Path(manifest["fs_home"]).resolve()):
            errors.append({"error": "External models cannot be inside source FreeSurfer"})
    if manifest.get("target_system") and manifest["target_system"] != platform.system():
        errors.append({"error": "Unsupported target system", "required": manifest["target_system"],
                       "actual": platform.system()})
    linkage = []
    checked = 0
    environment = {"PATH": "/usr/bin:/bin", "LC_ALL": "C",
                   "LD_LIBRARY_PATH": str(bundle / "lib"),
                   "FREESURFER_HOME": str(bundle)}
    for row in manifest["files"]:
        if not row.get("staged"):
            continue
        path = bundle / row["path"]
        if not under(path, bundle) or not path.is_file():
            errors.append({"file": row["path"], "error": "Missing or escaped bundle path"})
            continue
        if sha256(path) != row["sha256"]:
            errors.append({"file": row["path"], "error": "Hash mismatch"})
        checked += 1
        if row["kind"] != "elf":
            continue
        result = subprocess.run(["/usr/bin/ldd", str(path)], env=environment,
                                text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        linkage.append({"file": row["path"], "exit_code": result.returncode,
                        "ldd": result.stdout.splitlines()})
        if result.returncode and "not a dynamic executable" not in result.stdout:
            errors.append({"file": row["path"], "error": "ldd failed"})
        for line in result.stdout.splitlines():
            if "not found" in line:
                errors.append({"file": row["path"], "error": line.strip()})
            match = re.search(r"(?:=>\s+)?(/\S+)\s+\(", line)
            if not match:
                continue
            library = Path(match.group(1))
            name = line.strip().split()[0] if "=>" in line else library.name
            system = SYSTEM_LIBRARIES.match(name) and any(
                under(library, Path(prefix)) for prefix in ("/lib", "/lib64", "/usr/lib", "/usr/lib64"))
            if not under(library, bundle) and not system:
                errors.append({"file": row["path"], "error": "External library dependency",
                               "library": str(library)})
    for row in manifest["libraries"]:
        if row["system_runtime"]:
            continue
        path = bundle / "lib" / Path(row["resolved_source"]).name
        if not path.is_file() or sha256(path) != row["sha256"]:
            errors.append({"file": str(path), "error": "Bundled library missing or changed"})
    closure = scan_scripts(bundle, manifest["files"], python_executable=args.package_python,
                           required_resources=manifest.get("required_resources", []),
                           script_dispatch=manifest.get("script_dispatch", []),
                           runtime_profile=manifest.get("runtime_profile"),
                           inactive_command_audit=manifest.get("inactive_command_audit"))
    for command in manifest.get("required_commands", []):
        if not (bundle / "bin" / command).is_file():
            closure["errors"].append({"command": command, "error": "Required command missing"})
            closure["script_resource_checks_passed"] = False
    result = {"hash_and_linkage_passed": not errors, "checked_staged_files": checked,
              "checked_external_models": checked_models,
              "errors": errors, "elf_linkage": linkage, "standalone_verified": False,
              "script_resource_closure": closure,
              "script_resource_checks_passed": closure["script_resource_checks_passed"],
              "remaining_gates": ["Actual native execution and runtime license handling",
                                  "File and subprocess trace for complete T1 reconstruction",
                                  "Complete model, atlas, template and script closure",
                                  "Scientific comparison with the reference"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in
                      ("hash_and_linkage_passed", "script_resource_checks_passed", "checked_staged_files",
                       "checked_external_models", "errors", "standalone_verified")}))
    if errors or not closure["script_resource_checks_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
