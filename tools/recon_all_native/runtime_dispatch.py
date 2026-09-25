"""Package-owned interpreter dispatch with explicit upstream provenance."""

from pathlib import Path
import re
import shlex
import shutil
import subprocess

from audit_bundle import SYSTEM_LIBRARIES, sha256


AUXILIARY_ASSETS = {
    "mcadura": ["models/mca-dura.both-lh.nstd21.fhs.h5",
                "average/mca-dura.prior.warp.mni152.1.0mm.lh.nii.gz",
                "average/mca-dura.prior.warp.mni152.1.0mm.rh.nii.gz"],
    "vsinus": ["models/vsinus.no-sp.m.all.nstd10-070.h5",
               "average/vsinus.no-sp.prior.mni152.1.0mm.mgz"],
}


def elf_dependencies(source):
    result = subprocess.run(["ldd", str(source)], capture_output=True, text=True, check=False)
    if result.returncode or "not found" in result.stdout:
        raise RuntimeError(f"Unresolved runtime interpreter dependencies: {result.stdout}{result.stderr}")
    dependencies = {}
    for line in result.stdout.splitlines():
        match = re.search(r"(?:=>\s+)?(/\S+)\s+\(", line)
        if not match:
            continue
        path = Path(match.group(1))
        name = line.strip().split()[0] if "=>" in line else path.name
        key = str(path.resolve())
        row = dependencies.setdefault(key, {
            "source": str(path), "resolved_source": key, "sha256": sha256(path),
            "bytes": path.stat().st_size, "names": [], "users": ["libexec/tcsh"],
            "system_runtime": bool(SYSTEM_LIBRARIES.match(name))})
        row["names"].append(name)
    return list(dependencies.values())


def file_row(bundle, path, kind, **extra):
    return {"path": str(path.relative_to(bundle)), "sha256": sha256(path),
            "bytes": path.stat().st_size, "kind": kind, "staged": True, **extra}


def replace_script(bundle, manifest, relative, content, reason, *, preserve_for_execution=False):
    previous = next(row for row in manifest["files"] if row["path"] == relative)
    original_hash = previous["sha256"]
    source = bundle / relative
    backup = bundle / "upstream_scripts" / relative
    backup.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(source, backup)
    previous.update(path=str(backup.relative_to(bundle)), executed=preserve_for_execution)
    source.write_text(content)
    source.chmod(0o755)
    row = file_row(bundle, source, "generated_script", implementation=reason)
    manifest["files"].append(row)
    return {"path": relative, "upstream_path": str(backup.relative_to(bundle)),
            "upstream_sha256": original_hash, "replacement_sha256": row["sha256"],
            "reason": reason}


def add_interpreter_dispatch(bundle, manifest, tcsh=None, tcsh_notice=None):
    dispatch = []
    if tcsh:
        source = Path(tcsh).resolve()
        destination = bundle / "libexec/tcsh"
        destination.parent.mkdir(exist_ok=True)
        shutil.copy2(source, destination)
        manifest["files"].append(file_row(bundle, destination, "elf",
                                          origin="system_prebuilt_not_rebuilt", resolved_source=str(source)))
        known = {row["resolved_source"] for row in manifest["libraries"]}
        for row in elf_dependencies(source):
            if row["resolved_source"] not in known:
                manifest["libraries"].append(row)
                known.add(row["resolved_source"])
            if row["system_runtime"]:
                continue
            library = Path(row["resolved_source"])
            target = bundle / "lib" / library.name
            if target.exists() and sha256(target) != row["sha256"]:
                raise RuntimeError(f"Conflicting runtime library: {target.name}")
            shutil.copy2(library, target)
            for name in row["names"]:
                alias = target.parent / name
                if alias != target and not alias.exists():
                    alias.symlink_to(target.name)
        if tcsh_notice:
            notice = bundle / "licenses/tcsh.txt"
            notice.parent.mkdir(exist_ok=True)
            shutil.copy2(tcsh_notice, notice)
            manifest["files"].append(file_row(bundle, notice, "notice"))
        manifest["tcsh_notice_present"] = bool(tcsh_notice)
        for row in list(manifest["files"]):
            if not row["path"].startswith("bin/"):
                continue
            path = bundle / row["path"]
            with path.open("rb") as stream:
                header = stream.readline(512).decode(errors="replace").strip()
            if not header.startswith("#!"):
                continue
            words = shlex.split(header[2:])
            if not words or Path(words[0]).name != "tcsh":
                continue
            flags = " ".join(shlex.quote(word) for word in words[1:])
            relative = row["path"]
            script = ("#!/bin/sh\n"
                      ': "${FREESURFER_HOME:?set FREESURFER_HOME to the package runtime}"\n'
                      f'exec "$FREESURFER_HOME/libexec/tcsh" {flags} '
                      f'"$FREESURFER_HOME/upstream_scripts/{relative}" "$@"\n')
            item = replace_script(bundle, manifest, relative, script, "Bundled tcsh dispatch",
                                  preserve_for_execution=True)
            item.update(interpreter="libexec/tcsh", executed_path=item["upstream_path"])
            dispatch.append(item)
    for name in ("rca-config", "rca-config2csh"):
        relative = f"bin/{name}"
        python_source = f"python/scripts/{name}"
        if not (bundle / python_source).is_file() or not (bundle / relative).is_file():
            continue
        script = ("#!/bin/sh\n"
                  ': "${FS_TORCH_PYTHON:?set FS_TORCH_PYTHON to the package Python}"\n'
                  ': "${FREESURFER_HOME:?set FREESURFER_HOME to the package runtime}"\n'
                  f'exec "$FS_TORCH_PYTHON" -I "$FREESURFER_HOME/{python_source}" "$@"\n')
        item = replace_script(bundle, manifest, relative, script, "Package Python dispatch for official configuration code")
        item.update(interpreter="FS_TORCH_PYTHON", executed_path=python_source)
        dispatch.append(item)
    manifest["script_dispatch"] = dispatch
    complete_runtime_dispatch(bundle, manifest)


def complete_runtime_dispatch(bundle, manifest):
    """Relocate the remaining native shell/Python entry points, preserving code."""
    dispatch = manifest.setdefault("script_dispatch", [])
    has_tcsh = (bundle / "libexec/tcsh").is_file()
    for row in list(manifest["files"]):
        relative = row["path"]
        if not relative.startswith("bin/") or row.get("kind") == "generated_script":
            continue
        path = bundle / relative
        with path.open("rb") as stream:
            header = stream.readline(512).decode(errors="replace").strip()
        if not header.startswith("#!"):
            continue
        words = shlex.split(header[2:])
        if words and Path(words[0]).name == "env":
            words = words[1:]
        if not words:
            continue
        name = Path(words[0]).name
        interpreter = None
        if name in {"csh", "tcsh"} and has_tcsh:
            interpreter = "libexec/tcsh"
            command = '"$FREESURFER_HOME/libexec/tcsh"'
        elif name.startswith("python"):
            interpreter = "FS_TORCH_PYTHON"
            command = '"$FS_TORCH_PYTHON" -I'
        if interpreter:
            flags = " ".join(shlex.quote(word) for word in words[1:])
            script = ("#!/bin/sh\n"
                      ': "${FREESURFER_HOME:?set FREESURFER_HOME to the package runtime}"\n'
                      f'exec {command} {flags} "$FREESURFER_HOME/upstream_scripts/{relative}" "$@"\n')
            item = replace_script(bundle, manifest, relative, script,
                                  "Package interpreter dispatch", preserve_for_execution=True)
            item.update(interpreter=interpreter, executed_path=item["upstream_path"])
            dispatch.append(item)
    if has_tcsh:
        for name in ("tcsh", "csh"):
            path = bundle / "bin" / name
            if path.exists():
                continue
            path.write_text('#!/bin/sh\nexec "$FREESURFER_HOME/libexec/tcsh" "$@"\n')
            path.chmod(0o755)
            manifest["files"].append(file_row(bundle, path, "generated_script",
                                              implementation="Bundled tcsh command entry point"))
    row = next((row for row in manifest["files"] if row["path"] == "bin/fspython"), None)
    if row and row.get("kind") != "generated_script":
        script = ('#!/bin/sh\n'
                  ': "${FS_TORCH_PYTHON:?set FS_TORCH_PYTHON to the package Python}"\n'
                  'exec "$FS_TORCH_PYTHON" -I "$@"\n')
        item = replace_script(bundle, manifest, "bin/fspython", script,
                              "Package Python replaces the original private interpreter lookup")
        manifest.setdefault("runtime_replacements", []).append(item)
    # Installed Python command wrappers dispatch through fspython. Retain the
    # real Python source in the import/ABI scan; changing interpreters does not
    # make missing TensorFlow or other modules disappear.
    for row in manifest["files"]:
        relative = row["path"]
        if not relative.startswith("python/scripts/"):
            continue
        wrapper = bundle / "bin" / Path(relative).name
        if wrapper.is_file() and relative in wrapper.read_text(errors="replace") and "bin/fspython" in wrapper.read_text(errors="replace"):
            if not any(item["executed_path"] == relative for item in dispatch):
                dispatch.append({"path": str(wrapper.relative_to(bundle)), "executed_path": relative,
                                 "interpreter": "FS_TORCH_PYTHON",
                                 "reason": "Installed wrapper dispatches via package-owned bin/fspython"})
    manifest["target_system"] = "Linux"


def classify_linux_startup(bundle, manifest):
    """Record the exact Darwin-only setup branch; retain its files and hashes."""
    source = bundle / "sources.csh"
    if not source.is_file():
        return
    statements = [line.strip() for line in source.read_text().splitlines()
                  if line.strip() and not line.lstrip().startswith("#")]
    if statements != ['if ("`uname -s`" == "Darwin") then',
                      'source $FREESURFER_HOME/SetUpFreeSurfer.csh', 'endif']:
        return
    targets = {"SetUpFreeSurfer.csh", "FreeSurferEnv.csh"}
    allowed = {("sources.csh", "SetUpFreeSurfer.csh"),
               ("SetUpFreeSurfer.csh", "FreeSurferEnv.csh")}
    for row in manifest["files"]:
        if row.get("executed") is False:
            continue
        path = bundle / row["path"]
        if row.get("kind") not in {"script", "generated_script"} and path.name not in targets | {"sources.csh"}:
            continue
        for line in path.read_text(errors="replace").splitlines():
            if line.lstrip().startswith("#"):
                continue
            if re.match(r"^\s*echo(?:\s|$)", line) and "`" not in line and "$(" not in line:
                continue
            for target in targets:
                if re.search(r"\bsource\s+[^;]*" + re.escape(target), line) and (row["path"], target) not in allowed:
                    return
    for row in manifest["files"]:
        if row["path"] in targets:
            row.update(executed=False, execution_condition="Darwin only",
                       execution_evidence={"file": "sources.csh", "sha256": sha256(source),
                                           "condition": 'uname -s == Darwin', "target_system": "Linux"})


def replace_auxiliary(bundle, manifest, modes):
    """Opt in to separately validated subject/morphometry-specific GPU CLIs."""
    for mode in modes:
        relative = f"bin/mri_{mode}_seg"
        if any(item["path"] == relative for item in manifest.get("neural_replacements", [])):
            continue
        for resource in AUXILIARY_ASSETS[mode]:
            if not (bundle / resource).is_file():
                raise RuntimeError(f"Auxiliary replacement resource missing: {resource}")
        row = next(row for row in manifest["files"] if row["path"] == relative)
        # An older candidate already wraps this script through bundled tcsh.
        # Retain that launcher separately and archive the original source once.
        prior = next((item for item in manifest.get("script_dispatch", []) if item["path"] == relative), None)
        if prior:
            original = next(row for row in manifest["files"] if row["path"] == prior["upstream_path"])
            original["executed"] = False
            archive = bundle / "upstream_scripts/generated_dispatch" / Path(relative).name
            archive.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(bundle / relative, archive)
            row.update(path=str(archive.relative_to(bundle)), executed=False)
            manifest["script_dispatch"].remove(prior)
        else:
            original = row
            archive = bundle / "upstream_scripts" / relative
            archive.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(bundle / relative, archive)
            row.update(path=str(archive.relative_to(bundle)), executed=False)
        path = bundle / relative
        path.write_text('#!/bin/sh\n'
                        ': "${FS_TORCH_PYTHON:?set FS_TORCH_PYTHON to the package Python}"\n'
                        ': "${FREESURFER_HOME:?set FREESURFER_HOME to the package runtime}"\n'
                        'exec "$FS_TORCH_PYTHON" -m fnit.recon_all.aux_seg '
                        f'{mode} --assets "$FREESURFER_HOME" --device "${{FS_TORCH_DEVICE:-cuda:0}}" "$@"\n')
        path.chmod(0o755)
        generated = file_row(bundle, path, "generated_script", implementation="fnit.recon_all.aux_seg")
        manifest["files"].append(generated)
        manifest.setdefault("neural_replacements", []).append({
            "path": relative, "upstream_saved_path": original["path"],
            "upstream": {key: original[key] for key in ("sha256", "bytes", "resolved_source", "kind") if key in original},
            "replacement_sha256": generated["sha256"], "reason": f"Opt-in PyTorch {mode} dispatch"})
        manifest["required_resources"] = sorted(set(manifest.get("required_resources", [])) | set(AUXILIARY_ASSETS[mode]))
        manifest["required_commands"] = sorted(set(manifest.get("required_commands", [])) | {Path(relative).name})
    replacements = {item["path"] for item in manifest.get("neural_replacements", [])}
    required = {f"bin/mri_{mode}_seg" for mode in ("entowm", "mcadura", "vsinus")}
    if not required.issubset(replacements):
        return
    helper_paths = {"bin/mri_sclimbic_seg", "python/scripts/mri_sclimbic_seg"}
    for row in manifest["files"]:
        if row.get("executed") is False or row["path"] in helper_paths or row["kind"] not in {"script", "generated_script"}:
            continue
        text = (bundle / row["path"]).read_text(errors="replace")
        if any("mri_sclimbic_seg" in line for line in text.splitlines() if not line.lstrip().startswith("#")):
            return
    for row in manifest["files"]:
        if row["path"] in helper_paths:
            row.update(executed=False, execution_condition="All three upstream SCLimbic callers replaced",
                       execution_evidence={"replaced_commands": sorted(required),
                                           "active_bundle_script_references": [],
                                           "method": "Static literal-reference check; full runtime trace remains required"})
