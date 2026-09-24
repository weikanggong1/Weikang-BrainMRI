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


def under(path, root):
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--package-python", type=Path,
                        help="Python interpreter used by the installed freesurfer-torch package")
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    manifest = json.loads((bundle / "manifest.json").read_text())
    errors = []
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
                      ("hash_and_linkage_passed", "script_resource_checks_passed", "checked_staged_files", "errors", "standalone_verified")}))
    if errors or not closure["script_resource_checks_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
