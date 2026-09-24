#!/usr/bin/env python3
"""Inventory a trusted FreeSurfer installation from observed reference inputs.

This tool does not execute recon-all, copy personal license files, or claim that
an inventory proves runtime closure. Optional staging contains official prebuilt
binaries, not programs rebuilt from source.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess


SYSTEM_LIBRARIES = re.compile(
    r"^(ld-linux[^/]*|lib(?:c|m|dl|rt|pthread|util|anl|BrokenLocale)\.so\.[0-9.]+|"
    r"libgcc_s\.so\.[0-9.]+|libstdc\+\+\.so\.[0-9.]+)$"
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_personal_license(path):
    configured = os.environ.get("FS_LICENSE")
    return (path.name in {".license", "license.txt"}
            or path.resolve().name in {".license", "license.txt"}
            or bool(configured and path.resolve() == Path(configured).resolve()))


def is_elf(path):
    with path.open("rb") as stream:
        return stream.read(4) == b"\x7fELF"


def command_output(command):
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, check=False)
    return result.returncode, result.stdout


def inventory(fs_home, logs, traces, explicit):
    observed = {}
    excluded = []

    def add(path, evidence):
        path = Path(os.path.abspath(path))
        try:
            relative = path.relative_to(fs_home)
        except ValueError:
            return
        if not path.is_file():
            return
        # Runtime license keys are personal files, not distributable resources.
        if is_personal_license(path):
            excluded.append(str(relative))
            return
        observed.setdefault(str(relative), set()).add(evidence)

    for relative in explicit:
        add(fs_home / relative, "explicit")
    path_pattern = re.compile(re.escape(str(fs_home)) + r"/[^\s\"'<>;,()\[\]{}]+")
    for log in logs:
        for number, line in enumerate(log.read_text(errors="replace").splitlines(), 1):
            evidence = f"log:{log.name}:{number}"
            for match in path_pattern.finditer(line):
                add(match.group().rstrip(":").removesuffix("...").split("#")[0], evidence)
            try:
                words = shlex.split(line)
            except ValueError:
                continue
            if words:
                if words[0] in {"fs_time", "time"}:
                    words = words[1:]
                if words:
                    add(fs_home / "bin" / Path(words[0]).name, evidence)
    for trace in traces:
        for number, line in enumerate(trace.read_text(errors="replace").splitlines(), 1):
            if re.search(r"= -1\b", line):
                continue
            for match in path_pattern.finditer(line):
                add(match.group(), f"trace:{trace.name}:{number}")
    return observed, sorted(set(excluded))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fs-home", type=Path, required=True)
    parser.add_argument("--log", type=Path, action="append", default=[])
    parser.add_argument("--trace", type=Path, action="append", default=[])
    parser.add_argument("--file", action="append", default=[],
                        help="Explicit resource or binary, relative to fs-home")
    parser.add_argument("--files-from", type=Path,
                        help="Newline-delimited relative paths, for offline log extraction")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", type=Path,
                        help="Stage observed ELF programs, assets and non-system libraries")
    parser.add_argument("--include-scripts", action="store_true",
                        help="Copy scripts as candidates; their interpreter/import closure remains unverified")
    args = parser.parse_args()
    fs_home = args.fs_home.resolve()
    if not (fs_home / "bin").is_dir():
        parser.error("fs-home must contain bin/")
    for utility in ("ldd", "readelf"):
        if not shutil.which(utility):
            parser.error(f"Audit utility missing: {utility}")
    if args.stage and args.stage.exists():
        parser.error("stage directory must not already exist")

    explicit = list(args.file)
    if args.files_from:
        explicit.extend(line for line in args.files_from.read_text().splitlines()
                        if line and not line.startswith("#"))
    observed, excluded = inventory(fs_home, args.log, args.trace, explicit)
    files = []
    libraries = {}
    missing = []
    scripts = []
    for relative, evidence in sorted(observed.items()):
        path = fs_home / relative
        elf = is_elf(path)
        is_program = relative.startswith(("bin/", "python/scripts/"))
        row = {"path": relative, "resolved_source": str(path.resolve()),
               "sha256": sha256(path), "bytes": path.stat().st_size,
               "kind": "elf" if elf else ("script" if is_program else "asset"),
               "evidence": sorted(evidence)}
        if is_program and not elf:
            scripts.append(relative)
            row["staged"] = False
        if elf:
            _, dynamic = command_output(["readelf", "-d", str(path)])
            row["elf_dynamic"] = [line.strip() for line in dynamic.splitlines()
                                  if any(key in line for key in ("NEEDED", "RPATH", "RUNPATH"))]
            code, linkage = command_output(["ldd", str(path)])
            row["ldd_exit_code"] = code
            row["ldd"] = linkage.splitlines()
            for line in linkage.splitlines():
                if "not found" in line:
                    missing.append({"binary": relative, "dependency": line.strip()})
                match = re.search(r"(?:=>\s+)?(/\S+)\s+\(", line)
                if not match:
                    continue
                library = Path(match.group(1))
                name = line.strip().split()[0] if "=>" in line else library.name
                key = str(library.resolve())
                if key not in libraries:
                    libraries[key] = {"source": str(library), "resolved_source": key,
                                      "sha256": sha256(library), "bytes": library.stat().st_size,
                                      "system_runtime": bool(SYSTEM_LIBRARIES.match(name)),
                                      "names": [], "users": []}
                libraries[key]["names"].append(name)
                libraries[key]["users"].append(relative)
        files.append(row)
    for row in libraries.values():
        row["names"] = sorted(set(row["names"]))
        row["users"] = sorted(set(row["users"]))

    report = {
        "schema_version": 1, "origin": "official_prebuilt_not_rebuilt",
        "fs_home": str(fs_home), "standalone_verified": False,
        "files": files, "libraries": list(libraries.values()),
        "explicit_candidates_not_regular_files": [relative for relative in explicit
                                                   if not (fs_home / relative).is_file()],
        "missing_dynamic_libraries": missing, "excluded_personal_license_files": excluded,
        "observed_scripts": scripts,
        "observed_scripts_not_staged": [] if args.stage and args.include_scripts else scripts,
        "limitations": [
            "Logs and supplied traces establish observed files, not every possible branch.",
            "Relative file accesses in traces need cwd reconstruction and are not resolved here.",
            "Python and shell scripts require porting or a separately audited interpreter closure.",
            "Runtime license requirements remain in official executables; no key is bundled.",
            "Third-party redistribution terms and atlas/model licenses require review.",
            "RPATH and absolute resource paths need relocation and clean-environment validation.",
            "The system runtime allowlist is an inventory classification, not an ABI compatibility guarantee.",
        ],
    }
    if args.stage:
        if missing:
            parser.error("Cannot stage while dynamic dependencies are unresolved")
        args.stage.mkdir(parents=True)
        for row in files:
            if row["kind"] == "script" and not args.include_scripts:
                continue
            destination = args.stage / row["path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(fs_home / row["path"], destination, follow_symlinks=True)
            if sha256(destination) != row["sha256"]:
                raise RuntimeError(f"Staged hash mismatch: {destination}")
            row["staged"] = True
        library_dir = args.stage / "lib"
        library_dir.mkdir(exist_ok=True)
        for row in libraries.values():
            if row["system_runtime"]:
                continue
            source = Path(row["resolved_source"])
            destination = library_dir / source.name
            if destination.exists() and sha256(destination) != row["sha256"]:
                raise RuntimeError(f"Conflicting library basenames: {source.name}")
            shutil.copy2(source, destination)
            for name in row["names"]:
                alias = library_dir / Path(name).name
                if alias != destination:
                    if alias.is_symlink() and alias.resolve() != destination.resolve():
                        raise RuntimeError(f"Conflicting SONAME: {name}")
                    if not alias.exists():
                        alias.symlink_to(destination.name)
        report["stage"] = str(args.stage.resolve())
        (args.stage / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"files": len(files), "elf_files": sum(r["kind"] == "elf" for r in files),
                      "libraries": len(libraries), "missing_libraries": len(missing),
                      "scripts_not_staged": len(scripts), "standalone_verified": False,
                      "report": str(args.output)}))


if __name__ == "__main__":
    main()
