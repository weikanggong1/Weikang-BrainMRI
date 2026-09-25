#!/usr/bin/env python3
"""Create an unverified runtime candidate from a trusted FS8.2 installation.

Official prebuilt provenance is retained. Three named neural commands are
replaced by explicit fudan-neuroimaging-toolkit launchers. Personal licenses are excluded.
"""

import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

from audit_bundle import is_personal_license, sha256
from script_closure import FS_RESOURCE, mandatory_version_commands, scan_scripts
from runtime_dispatch import (AUXILIARY_ASSETS, add_interpreter_dispatch, classify_linux_startup,
                              elf_dependencies, replace_auxiliary, replace_script)
from single_t1_scope import configure_scope


NEURAL_TOOLS = ("mri_synthstrip", "mri_synthseg", "mri_synthmorph")
REQUIRED_RESOURCES = (
    "build-stamp.txt", "FreeSurferColorLUT.txt", "etc/recon-config.yaml",
    "etc/global-expert-options.v8.txt", "models/synthstrip.1.pt",
    "models/synthseg_2.0.h5", "models/synthseg_segmentation_labels_2.0.npy",
    "models/synthseg_segmentation_names_2.0.npy",
    "models/synthseg_topological_classes_2.0.npy", "models/synthmorph.affine.2.h5",
    "models/synthmorph.deform.3.h5", "average/RB_all_withskull_2020_01_02.gca",
    *(f"lib/bem/ic{level}.tri" for level in range(8)),
    "lib/bem/inner_skull.dat", "lib/bem/outer_skin.dat", "lib/bem/outer_skull.dat",
)
STARTUP_FILES = (
    "sources.csh", "FreeSurferEnv.csh", "bin/recon-all", "bin/fs-check-version",
    "bin/fsr-getxopts", "bin/rca-config", "bin/rca-config2csh", "bin/fspython",
    "bin/fs_time", "bin/UpdateNeeded", "bin/fsPrintHelp", "bin/fscalc",
    "bin/getfullpath", "bin/fname2stem", "bin/csvprint",
    "bin/fs_temp_file", "bin/fs_temp_dir", "bin/fs-check-os", "bin/reconbatchjobs",
    "python/scripts/rca-config", "python/scripts/rca-config2csh",
)


def safe_relative(value):
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Expected a relative path inside the install: {value}")
    return str(path)


def expand_literal_script_files(root, selected):
    """Follow literal package paths, including bin-to-python/script wrappers."""
    selected = set(selected)
    pending = list(selected)
    scanned = set()
    outside = []
    while pending:
        relative = pending.pop()
        if relative in scanned or relative in {f"bin/{tool}" for tool in NEURAL_TOOLS}:
            continue
        scanned.add(relative)
        source = root / relative
        if not source.is_file() or is_personal_license(source):
            continue
        with source.open("rb") as stream:
            if stream.read(2) != b"#!" and source.suffix not in {".py", ".sh", ".csh"}:
                continue
        text = source.read_text(errors="replace")
        for line_number, line in enumerate(text.splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            candidates = [match[1] for match in FS_RESOURCE.finditer(line)]
            candidates += [match[1] for match in re.finditer(
                re.escape(str(root)) + r"/([^\s\"'`;<>()[\],]+)", line)]
            for candidate in candidates:
                candidate = candidate.rstrip(":").split("#")[0]
                if any(symbol in candidate for symbol in ("$", "*", "?")):
                    continue
                # A literal such as $FREESURFER_HOME/../fspython describes an
                # external dependency, not a path that this bundle can follow.
                resolved = (root / candidate.lstrip("/")).resolve()
                try:
                    candidate = str(resolved.relative_to(root.resolve()))
                except ValueError:
                    outside.append({"script": relative, "line": line_number,
                                    "reference": candidate, "resolved_source": str(resolved)})
                    continue
                if (root / candidate).is_file() and candidate not in selected:
                    selected.add(candidate)
                    pending.append(candidate)
    return selected, outside


def launcher(tool):
    return ("#!/bin/sh\n"
            ': "${FS_TORCH_PYTHON:?set FS_TORCH_PYTHON to the package Python}"\n'
            f'exec "$FS_TORCH_PYTHON" -m fnit.recon_all.gpu_tools {tool} "$@"\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fs-home", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--files-from", type=Path)
    parser.add_argument("--extra-file", action="append", default=[])
    parser.add_argument("--include-tree", action="append", default=[],
                        help="Explicit reviewed resource/package subtree, relative to fs-home")
    parser.add_argument("--license-notice", type=Path,
                        default=Path(__file__).resolve().parents[2] / "licenses/FreeSurfer.txt",
                        help="MGH software license agreement text, never a personal runtime key")
    parser.add_argument("--package-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--tcsh", type=Path, help="Trusted tcsh executable to bundle with its dynamic dependencies")
    parser.add_argument("--tcsh-notice", type=Path, help="Copyright/license notice for that exact tcsh source release")
    parser.add_argument("--plan-only", action="store_true",
                        help="Write OUTPUT.plan.json with exact source inventory and payload sizes; do not copy a bundle")
    parser.add_argument("--replace-entowm", action="store_true",
                        help="Opt in to the separately validated sclimbic.py EntoWM CLI replacement")
    parser.add_argument("--replace-mcadura", action="store_true")
    parser.add_argument("--replace-vsinus", action="store_true")
    parser.add_argument("--single-t1-reference-config", type=Path,
                        help="Attach the reviewed resolved FS8.2 config and restrict this candidate to the fixed single-T1 profile")
    args = parser.parse_args()
    root = args.fs_home.resolve()
    output = args.output.resolve()
    if output.exists():
        parser.error("output must be a new directory")
    if not (root / "bin/recon-all").is_file():
        parser.error("fs-home lacks bin/recon-all")
    if args.replace_entowm and not (root / "bin/mri_entowm_seg").is_file():
        parser.error("EntoWM replacement requires an upstream bin/mri_entowm_seg for provenance")
    if not args.license_notice.is_file():
        parser.error("MGH license agreement text is required")
    notice = args.license_notice.read_text()
    if "PART B." not in notice or "Software License" not in notice:
        parser.error("license-notice must be the software license agreement, not a runtime key")
    if args.tcsh and not args.tcsh.is_file():
        parser.error("tcsh executable is missing")
    tcsh_version = None
    if args.tcsh:
        version = subprocess.run([str(args.tcsh), "--version"], text=True, capture_output=True, timeout=10)
        tcsh_version = version.stdout.strip()
        if version.returncode or not tcsh_version.startswith("tcsh "):
            parser.error("--tcsh must identify itself as tcsh with --version")
    if args.tcsh_notice and not args.tcsh_notice.is_file():
        parser.error("tcsh notice file is missing")
    if args.tcsh_notice and (is_personal_license(args.tcsh_notice)
                            or "Copyright" not in args.tcsh_notice.read_text()):
        parser.error("tcsh-notice must contain its copyright notice, never a runtime license key")
    output.parent.mkdir(parents=True, exist_ok=True)
    tools_dir = Path(__file__).resolve().parent
    required_resources = list(REQUIRED_RESOURCES)
    version_commands = mandatory_version_commands((root / "bin/recon-all").read_text())
    neural_tools = list(NEURAL_TOOLS)
    auxiliary = [mode for mode in AUXILIARY_ASSETS if getattr(args, "replace_" + mode)]
    for mode in auxiliary:
        required_resources += AUXILIARY_ASSETS[mode]
        neural_tools.append(f"mri_{mode}_seg")
    if args.replace_entowm:
        required_resources += ["models/entowm.fsm31.t1.nstd00-30.nstd21-108.h5", "models/entowm.ctab"]
        neural_tools.append("mri_entowm_seg")
    with tempfile.TemporaryDirectory(prefix="fs-native-package-", dir=output.parent) as temporary:
        temporary = Path(temporary)
        subprocess.run([sys.executable, str(tools_dir / "extract_reference.py"),
                        "--log", str(args.log), "--fs-home", str(root),
                        "--output-dir", str(temporary)], check=True)
        selected = set((temporary / "reference_files.txt").read_text().splitlines())
        selected.update(required_resources)
        selected.update(STARTUP_FILES)
        selected.update(f"bin/{tool}" for tool in neural_tools)
        selected.update(f"bin/{tool}" for tool in version_commands)
        selected.update(safe_relative(value) for value in args.extra_file)
        if args.files_from:
            selected.update(safe_relative(value) for value in args.files_from.read_text().splitlines()
                            if value and not value.startswith("#"))
        for directory in args.include_tree:
            tree = root / safe_relative(directory)
            if not tree.is_dir():
                parser.error(f"Explicit include-tree is missing: {directory}")
            selected.update(str(path.relative_to(root)) for path in tree.rglob("*") if path.is_file())
        selected.update("python/scripts/" + Path(relative).name for relative in list(selected)
                        if relative.startswith("bin/") and Path(relative).name not in NEURAL_TOOLS
                        and (root / "python/scripts" / Path(relative).name).is_file())
        selected, outside_references = expand_literal_script_files(root, selected)
        selected_file = temporary / "selected.txt"
        selected_file.write_text("\n".join(sorted(selected)) + "\n")
        inventory = temporary / "source_inventory.json"
        audit = [sys.executable, str(tools_dir / "audit_bundle.py"),
                 "--fs-home", str(root), "--log", str(args.log),
                 "--files-from", str(selected_file), "--output", str(inventory)]
        if not args.plan_only:
            audit += ["--stage", str(output), "--include-scripts"]
        subprocess.run(audit, check=True)
        manifest = json.loads(inventory.read_text())
        manifest["literal_resource_references_outside_fs_home"] = outside_references
        if args.plan_only:
            libraries = {row["resolved_source"]: row for row in manifest["libraries"]}
            interpreter_bytes = 0
            if args.tcsh:
                libraries.update({row["resolved_source"]: row for row in elf_dependencies(args.tcsh)})
                interpreter_bytes = args.tcsh.stat().st_size
            payload = sum(row["bytes"] for row in manifest["files"])
            payload += sum(row["bytes"] for row in libraries.values() if not row["system_runtime"])
            payload += interpreter_bytes + len(notice.encode())
            manifest["planned_file_payload_bytes_before_generated_metadata"] = payload
            manifest["planned_file_payload_gib"] = round(payload / 2 ** 30, 4)
            manifest["planned_tcsh"] = str(args.tcsh) if args.tcsh else None
            manifest["tcsh_version"] = tcsh_version
            manifest["planned_neural_replacements"] = neural_tools
            manifest["standalone_verified"] = False
            plan_path = Path(str(output) + ".plan.json")
            plan_path.write_text(json.dumps(manifest, indent=2) + "\n")
            print(json.dumps({"plan": str(plan_path), "source_files": len(manifest["files"]),
                              "payload_gib": manifest["planned_file_payload_gib"],
                              "bundle_created": False, "standalone_verified": False}))
            return
        metadata = output / "metadata"
        metadata.mkdir(exist_ok=True)
        for name in ("source_inventory.json", "reference_requirements.json", "selected.txt"):
            shutil.copy2(temporary / name, metadata / name)
    manifest["files"] = [row for row in manifest["files"] if row.get("staged")]
    indexed = {row["path"]: row for row in manifest["files"]}
    replacements = []
    for tool in NEURAL_TOOLS:
        relative = f"bin/{tool}"
        prior = indexed.get(relative)
        original = {key: prior[key] for key in ("sha256", "bytes", "resolved_source", "kind")} if prior else None
        path = output / relative
        path.write_text(launcher(tool))
        path.chmod(0o755)
        row = {"path": relative, "sha256": sha256(path), "bytes": path.stat().st_size,
               "kind": "generated_script", "staged": True,
               "implementation": "fnit.recon_all.gpu_tools"}
        if prior:
            manifest["files"].remove(prior)
        manifest["files"].append(row)
        replacements.append({"path": relative, "upstream": original,
                             "replacement_sha256": row["sha256"],
                             "reason": "Explicit PyTorch GPU dispatch replacement"})
    if args.replace_entowm:
        relative = "bin/mri_entowm_seg"
        original = next(row for row in manifest["files"] if row["path"] == relative)
        upstream = {key: original[key] for key in ("sha256", "bytes", "resolved_source", "kind")}
        script = ("#!/bin/sh\n"
                  ': "${FS_TORCH_PYTHON:?set FS_TORCH_PYTHON to the package Python}"\n'
                  ': "${FREESURFER_HOME:?set FREESURFER_HOME to the package runtime}"\n'
                  'exec "$FS_TORCH_PYTHON" -m fnit.recon_all.sclimbic '
                  '--assets "$FREESURFER_HOME/models" --device "${FS_TORCH_DEVICE:-cuda:0}" "$@"\n')
        replacement = replace_script(output, manifest, relative, script,
                                     "Opt-in PyTorch EntoWM subject-mode dispatch")
        replacements.append({"path": relative, "upstream": upstream,
                             "upstream_saved_path": replacement["upstream_path"],
                             "replacement_sha256": replacement["replacement_sha256"],
                             "reason": replacement["reason"]})
    license_path = output / "licenses/FreeSurfer.txt"
    license_path.parent.mkdir(exist_ok=True)
    license_path.write_text(
        'All or portions of this licensed product (such portions are the "Software") '
        'have been obtained under license from The General Hospital Corporation "MGH" '
        'and are subject to the following terms and conditions:\n\n' + notice)
    manifest["files"].append({"path": "licenses/FreeSurfer.txt", "kind": "notice",
                              "sha256": sha256(license_path), "bytes": license_path.stat().st_size,
                              "staged": True})
    add_interpreter_dispatch(output, manifest, args.tcsh, args.tcsh_notice)
    classify_linux_startup(output, manifest)
    manifest["neural_replacements"] = replacements
    replace_auxiliary(output, manifest, auxiliary)
    for row in manifest["libraries"]:
        if not row["system_runtime"]:
            path = output / "lib" / Path(row["resolved_source"]).name
            manifest["files"].append({"path": str(path.relative_to(output)), "kind": "library",
                                      "sha256": sha256(path), "bytes": path.stat().st_size,
                                      "staged": True})
    manifest.update(standalone_verified=False, candidate_bundle=True,
                    neural_replacements=replacements,
                    required_resources=required_resources,
                    required_commands=["recon-all", "fs_temp_file", "fs_temp_dir", "fs-check-os", "reconbatchjobs",
                                       *neural_tools, *version_commands],
                    runtime_license_bundled=False, redistribution_review_complete=False)
    manifest["tcsh_version"] = tcsh_version
    if args.single_t1_reference_config:
        configure_scope(output, manifest, args.single_t1_reference_config)
    manifest["files"].sort(key=lambda row: row["path"])
    closure = scan_scripts(output, manifest["files"], python_executable=args.package_python,
                           required_resources=manifest["required_resources"],
                           script_dispatch=manifest.get("script_dispatch", []),
                           runtime_profile=manifest.get("runtime_profile"),
                           inactive_command_audit=manifest.get("inactive_command_audit"))
    (metadata / "script_resource_closure.json").write_text(json.dumps(closure, indent=2) + "\n")
    manifest["script_resource_checks_passed"] = closure["script_resource_checks_passed"]
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"candidate": str(output), "files": len(manifest["files"]),
                      "neural_replacements": neural_tools,
                      "script_resource_checks_passed": closure["script_resource_checks_passed"],
                      "closure_issues": len(closure["errors"]), "standalone_verified": False}))


if __name__ == "__main__":
    main()
