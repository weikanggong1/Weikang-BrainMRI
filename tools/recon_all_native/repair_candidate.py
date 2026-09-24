#!/usr/bin/env python3
"""Complete audited runtime dispatch in an existing, still-unverified candidate."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from audit_bundle import sha256
from runtime_dispatch import AUXILIARY_ASSETS, classify_linux_startup, complete_runtime_dispatch, file_row, replace_auxiliary
from script_closure import mandatory_version_commands, scan_scripts
from single_t1_scope import configure_scope


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--fs-home", type=Path, required=True)
    parser.add_argument("--package-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--replace-mcadura", action="store_true")
    parser.add_argument("--replace-vsinus", action="store_true")
    parser.add_argument("--extra-file", action="append", default=[],
                        help="reviewed relative resource path to add to the candidate")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--single-t1-reference-config", type=Path,
                        help="Attach the reviewed resolved FS8.2 config and restrict this candidate to the fixed single-T1 profile")
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("standalone_verified") or not manifest.get("candidate_bundle"):
        parser.error("Only an explicitly unverified candidate may be repaired")
    stamp = args.fs_home / "build-stamp.txt"
    if not stamp.is_file() or sha256(stamp) != sha256(bundle / "build-stamp.txt"):
        parser.error("Upstream build-stamp differs from the candidate; do not mix versions")
    for row in manifest["files"]:
        path = bundle / row["path"]
        if not path.resolve().is_relative_to(bundle) or not path.is_file() or sha256(path) != row["sha256"]:
            parser.error(f"Existing candidate integrity check failed: {row['path']}")
    recon = bundle / "upstream_scripts/bin/recon-all"
    if not recon.is_file():
        recon = bundle / "bin/recon-all"
    required = ("fs_temp_file", "fs_temp_dir", "fs-check-os", "reconbatchjobs",
                *mandatory_version_commands(recon.read_text()))
    additions = ["bin/" + name for name in required if not (bundle / "bin" / name).is_file()]
    for name in args.extra_file:
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            parser.error(f"Extra file must be inside the installation: {name}")
        if not (bundle / relative).is_file():
            additions.append(name)
    auxiliary = [mode for mode in AUXILIARY_ASSETS if getattr(args, "replace_" + mode)]
    for mode in auxiliary:
        for name in [f"bin/mri_{mode}_seg", *AUXILIARY_ASSETS[mode]]:
            if not (bundle / name).is_file():
                additions.append(name)
    for name in additions:
        if not (args.fs_home / name).is_file():
            parser.error(f"Required upstream helper is missing: {name}")
    if additions:
        with tempfile.TemporaryDirectory(prefix="fs-runtime-repair-", dir=bundle.parent) as temporary:
            temporary = Path(temporary)
            staged = temporary / "stage"
            report = bundle / "metadata" / ("runtime-repair-plan.json" if args.plan_only
                                            else "runtime-repair-source-inventory.json")
            command = [sys.executable, str(Path(__file__).with_name("audit_bundle.py")),
                       "--fs-home", str(args.fs_home), "--output", str(report)]
            if not args.plan_only:
                command += ["--stage", str(staged), "--include-scripts"]
            for name in additions:
                command += ["--file", name]
            subprocess.run(command, check=True)
            extra = json.loads(report.read_text())
            if args.plan_only:
                payload = sum(row["bytes"] for row in extra["files"])
                payload += sum(row["bytes"] for row in extra["libraries"] if not row["system_runtime"]
                               and not (bundle / "lib" / Path(row["resolved_source"]).name).exists())
                print(json.dumps({"additions": additions, "additional_gib": round(payload / 2 ** 30, 4),
                                  "source_files": len(extra["files"]), "candidate_changed": False}))
                return
            known = {row["resolved_source"]: row for row in manifest["libraries"]}
            for row in extra["libraries"]:
                prior = known.get(row["resolved_source"])
                if prior and prior["sha256"] != row["sha256"]:
                    raise RuntimeError(f"Upstream dependency changed: {row['resolved_source']}")
                if row["system_runtime"]:
                    continue
                target = bundle / "lib" / Path(row["resolved_source"]).name
                if target.exists() and sha256(target) != row["sha256"]:
                    raise RuntimeError(f"Conflicting library basename: {target.name}")
                for name in row["names"]:
                    alias = target.parent / name
                    if alias.exists() and alias.resolve() != target.resolve():
                        raise RuntimeError(f"Conflicting library alias: {name}")
            for row in extra["files"]:
                target = bundle / row["path"]
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(staged / row["path"], target)
                manifest["files"].append(row)
            for row in extra["libraries"]:
                prior = known.get(row["resolved_source"])
                if prior:
                    prior["names"] = sorted(set(prior["names"]) | set(row["names"]))
                    prior["users"] = sorted(set(prior["users"]) | set(row["users"]))
                else:
                    manifest["libraries"].append(row)
                    known[row["resolved_source"]] = row
                if row["system_runtime"]:
                    continue
                target = bundle / "lib" / Path(row["resolved_source"]).name
                if not target.exists():
                    shutil.copy2(staged / "lib" / target.name, target)
                for name in row["names"]:
                    alias = target.parent / name
                    if alias != target and not alias.exists():
                        alias.symlink_to(target.name)
                if not any(item["path"] == str(target.relative_to(bundle)) for item in manifest["files"]):
                    manifest["files"].append(file_row(bundle, target, "library"))
    if args.plan_only:
        print(json.dumps({"additions": [], "additional_gib": 0, "candidate_changed": False}))
        return
    backup = bundle / "metadata/manifest.before-runtime-repair.json"
    if not backup.exists():
        shutil.copy2(manifest_path, backup)
    complete_runtime_dispatch(bundle, manifest)
    classify_linux_startup(bundle, manifest)
    replace_auxiliary(bundle, manifest, auxiliary)
    manifest["required_commands"] = sorted(set(manifest.get("required_commands", [])) | set(required))
    manifest["required_resources"] = sorted(set(manifest.get("required_resources", [])) | set(args.extra_file))
    if args.single_t1_reference_config:
        configure_scope(bundle, manifest, args.single_t1_reference_config)
    manifest["files"].sort(key=lambda row: row["path"])
    closure = scan_scripts(bundle, manifest["files"], python_executable=args.package_python,
                           required_resources=manifest.get("required_resources", []),
                           script_dispatch=manifest.get("script_dispatch", []),
                           runtime_profile=manifest.get("runtime_profile"),
                           inactive_command_audit=manifest.get("inactive_command_audit"))
    manifest["script_resource_checks_passed"] = closure["script_resource_checks_passed"]
    manifest["standalone_verified"] = False
    (bundle / "metadata/script_resource_closure.json").write_text(json.dumps(closure, indent=2) + "\n")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"added_helpers": additions, "files": len(manifest["files"]),
                      "closure_issues": len(closure["errors"]),
                      "script_resource_checks_passed": closure["script_resource_checks_passed"],
                      "standalone_verified": False}))


if __name__ == "__main__":
    main()
