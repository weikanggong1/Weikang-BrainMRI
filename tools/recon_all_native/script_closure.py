#!/usr/bin/env python3
"""Check staged scripts, literal resources and declared Python dependencies.

Static checks are conservative: shell branches and constructed paths cannot be
proved complete without a trace of the actual reconstruction. No script is run.
"""

import ast
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

from single_t1_scope import validate_scope_audit


STANDARD_UTILITIES = frozenset(
    "awk basename bc cat chmod cmp cp cut date dirname du echo env expr false find "
    "grep head hostname id ln ls mkdir mktemp mv printf pwd readlink rm sed seq "
    "sleep sort tail tee test touch tr true uname uniq wc which whoami".split()
)
FS_HELPERS = frozenset(
    "recon-all UpdateNeeded fs_time fsPrintHelp fscalc fspython fs_python "
    "fname2stem getfullpath isargflag printargs csvprint log_append "
    "fs-check-version fsr-getxopts rca-config rca-config2csh reconbatchjobs tcsh csh".split()
)
FS_COMMAND = re.compile(r"^(?:mri_|mris_|mrisp_|mris2|Ants|rca-|fs[-_]|seg2|lta_)")
FS_RESOURCE = re.compile(
    r"\$\{?(?:FREESURFER_HOME_FSPYTHON|FREESURFER_HOME|FREESURFER)\}?/([^\s\"'`;<>()[\],]+)"
)


def python_imports(text):
    modules = set()
    try:
        tree = ast.parse(text)
    except SyntaxError as error:
        return [], f"Python AST parse failed at line {error.lineno}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules.add(node.module.split(".")[0])
    return sorted(modules), None


def label_data(path, first):
    """Recognize the data header written by FreeSurfer LabelWriteInto."""
    if path.suffix != ".label" or not re.match(r"^#!ascii label(?:\s|,)", first):
        return False
    with path.open() as stream:
        next(stream)
        count = next(stream, "").strip()
        if not count.isdigit():
            return False
        points = 0
        for line in stream:
            values = line.split()
            if len(values) != 5:
                return False
            try:
                int(values[0])
                [float(value) for value in values[1:]]
            except ValueError:
                return False
            points += 1
    return points == int(count)


def executable_text(text, relative):
    # Upstream scripts print their trailing help with awk; this text follows
    # an unconditional exit and is not shell code. Require both source markers.
    if "\nBEGINHELP\n" in text:
        code, help_text = text.split("\nBEGINHELP\n", 1)
        statements = [line.strip() for line in code.splitlines()
                      if line.strip() and not line.lstrip().startswith("#")]
        if (statements and re.fullmatch(r"exit\s+\d+;?", statements[-1])
                and "cat $0 | awk" in code):
            return code, {"kind": "trailing_upstream_help", "lines": len(help_text.splitlines()),
                          "evidence": "BEGINHELP follows unconditional exit; help reader uses cat $0 | awk"}
    return text, None


def mandatory_version_commands(text):
    """recon-all requires every program in allinfocmds to pass -all-info."""
    match = re.search(r"\bset\s+allinfocmds\s*=\s*\((.*?)\n\)", text, re.S)
    if not match or "foreach cmd ($allinfocmds)" not in text or "$cmd -all-info" not in text:
        return []
    body = " ".join(line.split("#", 1)[0] for line in match[1].splitlines())
    return sorted(set(body.replace("\\", " ").split()))


def check_python_modules(modules, python_executable):
    # The selected interpreter is the package interpreter, not FreeSurfer's
    # original private Python. Match the dispatcher's isolated sys.path; staged
    # upstream Python directories are not automatically importable at runtime.
    code = """import importlib.util,json,sys
result = {}
for name in json.loads(sys.argv[1]):
    try:
        spec = importlib.util.find_spec(name)
        result[name] = {"available": spec is not None,
                        "origin": None if spec is None else spec.origin}
    except (ImportError, ValueError, AttributeError) as error:
        result[name] = {"available": False, "error": str(error)}
print(json.dumps(result))
"""
    result = subprocess.run(
        [str(python_executable), "-I", "-c", code, json.dumps(sorted(modules))],
        text=True, capture_output=True, check=False)
    if result.returncode:
        return {}, "Package Python module check failed: " + result.stderr.strip()
    return json.loads(result.stdout), None


def scan_scripts(bundle, files, *, python_executable=None, required_resources=(), script_dispatch=(),
                 runtime_profile=None, inactive_command_audit=None):
    bundle = Path(bundle).resolve()
    runtime_path = f"{bundle / 'bin'}:/usr/bin:/bin"
    python_executable = python_executable or sys.executable
    rows = []
    inactive_pairs, errors = validate_scope_audit(bundle, runtime_profile, inactive_command_audit)
    inactive_commands = []
    all_modules = set()
    data_files = []
    required = sorted(set(required_resources))
    dispatch = {row["executed_path"]: row["interpreter"] for row in script_dispatch}
    for resource in required:
        if not (bundle / resource).is_file():
            errors.append({"resource": resource, "error": "Required resource missing"})
    for entry in files:
        if entry.get("executed") is False:
            continue
        relative = entry["path"]
        path = bundle / relative
        if not path.is_file():
            continue
        with path.open("rb") as stream:
            first = stream.readline(1024).decode(errors="replace").rstrip()
        if label_data(path, first):
            data_files.append({"path": relative, "format": "FreeSurfer ASCII label",
                               "evidence": "LabelWriteInto header, count and five-column numeric rows"})
            continue
        if not first.startswith("#!") and path.suffix not in {".py", ".csh", ".sh"}:
            continue
        text = path.read_text(errors="replace")
        text, noncode = executable_text(text, relative)
        words = shlex.split(first[2:]) if first.startswith("#!") else []
        interpreter = words[0] if words else None
        if interpreter and Path(interpreter).name == "env" and len(words) > 1:
            interpreter = words[1]
        name = Path(interpreter).name if interpreter else ""
        interpreter_status = "not_declared"
        if relative in dispatch:
            configured = dispatch[relative]
            interpreter_status = "package_dispatch" if configured == "FS_TORCH_PYTHON" or (bundle / configured).is_file() else "missing"
        elif relative in {"sources.csh", "FreeSurferEnv.csh"} and (bundle / "libexec/tcsh").is_file():
            interpreter_status = "sourced_with_bundled_tcsh"
        elif name in {"sh", "bash"}:
            interpreter_status = "host_base_shell" if shutil.which(name, path=runtime_path) else "missing"
        elif name.startswith("python"):
            interpreter_status = "requires_explicit_package_python_dispatch"
        elif interpreter:
            interpreter_status = "unbundled_interpreter"
        if interpreter_status in {"missing", "unbundled_interpreter", "requires_explicit_package_python_dispatch"}:
            errors.append({"script": relative, "interpreter": interpreter,
                           "error": interpreter_status})
        is_python = name.startswith("python") or path.suffix == ".py"
        modules, parse_error = python_imports(text) if is_python else ([], None)
        if "-m freesurfer_torch." in text:
            modules.append("freesurfer_torch")
        all_modules.update(modules)
        if parse_error:
            errors.append({"script": relative, "error": parse_error})
        version_commands = set(mandatory_version_commands(text))
        commands = set(version_commands)
        resources = set()
        dynamic = set()
        for line in text.splitlines():
            if line.lstrip().startswith("#"):
                continue
            for match in FS_RESOURCE.finditer(line):
                value = match[1].rstrip(":")
                if any(marker in value for marker in ("$", "*", "?")):
                    dynamic.add(value)
                else:
                    resources.add(value)
            if is_python:
                continue
            candidates = re.findall(r"(?:^|[`;|])\s*([A-Za-z_][\w.+-]*)", line)
            candidates += re.findall(r"\bset\s+cmd\s*=\s*\(\s*([A-Za-z_][\w.+-]*)", line)
            candidates += re.findall(r"\bset\s+cmd\s*=\s*\(\s*which\s+([A-Za-z_][\w.+-]*)", line)
            for candidate in candidates:
                if candidate in FS_HELPERS or FS_COMMAND.match(candidate):
                    commands.add(candidate)
                elif candidate in STANDARD_UTILITIES and not shutil.which(candidate, path=runtime_path):
                    errors.append({"script": relative, "command": candidate,
                                   "error": "Required host base utility unavailable"})
        missing_commands = sorted(command for command in commands if not (bundle / "bin" / command).is_file())
        outside_resources = sorted(resource for resource in resources
                                   if not (bundle / resource).resolve().is_relative_to(bundle))
        missing_resources = sorted(resource for resource in resources
                                   if resource not in outside_resources and not (bundle / resource).exists())
        for command in missing_commands:
            if (relative, command) in inactive_pairs and command not in version_commands:
                inactive_commands.append({"script": relative, "command": command,
                                          "profile_id": runtime_profile["id"],
                                          "evidence": "manifest.inactive_command_audit.rules"})
                continue
            errors.append({"script": relative, "command": command,
                           "error": "Literal command candidate not bundled; branch review required"})
        for resource in missing_resources:
            errors.append({"script": relative, "resource": resource,
                           "error": "Literal resource candidate missing; branch review required"})
        for resource in outside_resources:
            errors.append({"script": relative, "resource": resource,
                           "error": "Literal resource resolves outside the package"})
        rows.append({"path": relative, "shebang": first, "interpreter": interpreter,
                     "interpreter_status": interpreter_status, "python_modules": sorted(set(modules)),
                     "command_candidates": sorted(commands), "literal_resources": sorted(resources),
                     "nonexecutable_text": noncode,
                     "constructed_resources_requiring_trace": sorted(dynamic)})
    modules, module_error = check_python_modules(all_modules, python_executable)
    if module_error:
        errors.append({"error": module_error})
    for name, status in modules.items():
        if not status["available"]:
            errors.append({"python_module": name, "error": "Unavailable in selected package Python"})
    return {"script_resource_checks_passed": not errors, "scripts": rows,
            "inactive_commands": inactive_commands,
            "recognized_data_files": data_files,
            "required_resources": required, "python_executable": str(python_executable),
            "python_modules": modules, "errors": errors, "closure_complete": False,
            "limitations": ["Shell branch reachability is not inferred from source text.",
                            "Constructed file paths, imports and subprocesses require an execution trace.",
                            "A present Python module has not been checked for ABI or version compatibility."]}
