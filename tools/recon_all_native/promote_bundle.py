#!/usr/bin/env python3
"""Promote one scoped candidate after linked runtime and numerical evidence passes.

The trace check concerns directly visible FreeSurfer/FSL/TensorFlow accesses;
it is not a general reconstruction of all relative paths or file descriptors.
"""

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile

from audit_bundle import sha256
from check_comparison_gate import check_report
from single_t1_scope import PROFILE, validate_scope_audit


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_report(path):
    report = json.loads(path.read_text())
    require(isinstance(report, dict), f"Report must be an object: {path}")
    return report


def required_checks():
    names = set()
    for hemi in ("lh", "rh"):
        for name in ("white", "pial", "white.preaparc", "inflated", "thickness", "area",
                     "area.pial", "area.mid", "volume", "curv", "curv.pial", "sulc",
                     "white.H", "white.K", "inflated.H", "inflated.K"):
            names.add(f"surf/{hemi}.{name}")
        for name in ("aparc.annot", "aparc.DKTatlas.annot", "aparc.a2009s.annot", "cortex.label"):
            names.add(f"label/{hemi}.{name}")
        for name in ("aparc.stats", "aparc.pial.stats", "aparc.DKTatlas.stats", "aparc.a2009s.stats"):
            names.add(f"stats/{hemi}.{name}")
    return names | {"mri/aseg.mgz", "mri/aparc+aseg.mgz", "stats/aseg.stats", "stats/synthseg.vol.csv"}


def trace_audit(path, bundle, source_home, license_file, candidate, image):
    """Pair split open/exec records and reject visible external scientific code/assets."""
    calls = {"open", "openat", "openat2", "creat", "execve", "execveat"}
    pending, violations = {}, []
    successful, bundle_execs, run_execs = 0, 0, 0
    marker = re.compile(r"(?:^|/)(?:freesurfer|fsl|tensorflow)(?:[-_.]v?\d[^/]*)?(?:/|$)|(?:^|/)libtensorflow[^/]*", re.I)
    with path.open(errors="strict") as stream:
        for number, raw in enumerate(stream, 1):
            match = re.match(r"^(?:\[pid\s+(\d+)\]\s+|(\d+)\s+)?(.*)$", raw.rstrip())
            pid, line = match[1] or match[2] or "main", match[3]
            resumed = re.match(r"<\.\.\. (\w+) resumed>(.*)", line)
            if resumed:
                if resumed[1] not in calls:
                    continue
                require(pid in pending, f"Unpaired trace continuation at line {number}")
                start, prefix, name = pending.pop(pid)
                require(name == resumed[1], f"Mismatched trace continuation at line {number}")
                line, number = prefix + resumed[2], start
            call = re.match(r"(\w+)\(", line)
            if not call or call[1] not in calls:
                continue
            name = call[1]
            if "<unfinished ...>" in line:
                require(pid not in pending, f"Overlapping trace calls at line {number}")
                pending[pid] = (number, line.split("<unfinished ...>")[0], name)
                continue
            result = re.search(r"\)\s+=\s+(-?\d+)\b", line)
            require(result is not None, f"Unparsed open/exec result at trace line {number}")
            if int(result[1]) < 0:
                continue
            require(not name.startswith("exec") or int(result[1]) == 0, f"Invalid successful exec result at line {number}")
            quoted = re.search(r'"(?:\\.|[^"\\])*"', line)
            require(quoted is not None and not line[quoted.end():].lstrip().startswith("..."),
                    f"Missing/truncated path at trace line {number}")
            accessed = ast.literal_eval(quoted[0])
            successful += 1
            absolute = Path(accessed).is_absolute()
            target = Path(accessed).resolve() if absolute else None
            if target is not None and target.is_relative_to(bundle):
                bundle_execs += int(name.startswith("exec"))
                if name.startswith("exec") and target == bundle / "bin/recon-all":
                    arguments = [ast.literal_eval(value) for value in re.findall(r'"(?:\\.|[^"\\])*"', line)]
                    pairs = set(zip(arguments, arguments[1:]))
                    run_execs += int({("-s", candidate.name), ("-sd", str(candidate.parent)),
                                     ("-i", str(image))}.issubset(pairs))
                continue
            if target == license_file:
                continue
            if marker.search(accessed) or (target is not None and target.is_relative_to(source_home)):
                violations.append({"line": number, "pid": pid, "call": name, "path": accessed})
    require(not pending, "Trace has unfinished open/exec records")
    require(successful > 0 and bundle_execs > 0 and run_execs > 0,
            "Trace lacks bundle recon-all execution matching this subject and T1")
    require(not violations, f"External FreeSurfer/FSL/TensorFlow access: {violations[:10]}")
    return {"passed": True, "successful_open_exec_records": successful,
            "bundle_exec_records": bundle_execs, "matching_run_exec_records": run_execs,
            "external_scientific_accesses": violations,
            "license_exception": str(license_file),
            "boundary": "Visible path arguments only; relative cwd/dirfd targets and arbitrary other external resources are not reconstructed"}


def validate_launch(path, cwd, args, run):
    tokens = shlex.split(path.read_text())
    require(tokens[:2] == ["env", "-i"], "Trace launch must explicitly start with env -i")
    index = next((i for i, token in enumerate(tokens) if Path(token).name == "strace"), -1)
    require(index > 1, "Trace launch lacks strace")
    require(all("=" in item and item.split("=", 1)[0] in {"HOME", "LANG", "LC_ALL", "PATH", "TMPDIR", "CUDA_VISIBLE_DEVICES"}
                for item in tokens[2:index]), "Unexpected clean launch environment assignment")
    entry = next((i for i in range(index + 1, len(tokens)) if Path(tokens[i]).name == "fs-torch-recon-all"), -1)
    require(entry > index, "Trace launch lacks the package recon-all entry")
    trace_args = tokens[index + 1:entry]
    require("-f" in trace_args and "trace=file,process" in trace_args, "Trace must follow children and file/process calls")
    require("-o" in trace_args and (cwd / trace_args[trace_args.index("-o") + 1]).resolve() == args.trace.resolve(),
            "Trace launch output does not match supplied trace")
    options = tokens[entry + 1:]
    for flag, expected in (("-i", args.input.resolve()), ("-sd", Path(run["subject_dir"]).parent.resolve()),
                           ("--bundle", args.bundle.resolve()), ("--license", args.license_file.resolve())):
        require(flag in options and (cwd / options[options.index(flag) + 1]).resolve() == expected,
                f"Trace launch has different {flag}")
    for flag, expected in (("-s", run["subject"]), ("--threads", "4"), ("--device", run["device"])):
        require(flag in options and options[options.index(flag) + 1] == expected, f"Trace launch has different {flag}")
    return {"cwd": str(cwd), "command": path.read_text().strip(),
            "provenance": "Operator-supplied launch record, not a digital signature"}


def collect_software(python):
    code = '''import hashlib,importlib.metadata,json,platform,sys
from pathlib import Path
import freesurfer_torch,torch
root=Path(freesurfer_torch.__file__).resolve().parent
files={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob("*.py"))}
print(json.dumps({"python_executable":sys.executable,"python":platform.python_version(),
"platform":platform.platform(),"versions":{n:importlib.metadata.version(n) for n in
["freesurfer-torch","torch","numpy","scipy","surfa","nibabel","h5py","PyYAML"]},
"torch_cuda":torch.version.cuda,"cudnn":torch.backends.cudnn.version(),
"cuda_available":torch.cuda.is_available(),"package_root":str(root),"package_files_sha256":files}))'''
    completed = subprocess.run([str(python.absolute()), "-I", "-c", code], text=True, capture_output=True, check=True)
    result = json.loads(completed.stdout)
    require(result.get("cuda_available") is True and bool(result.get("torch_cuda")), "Selected Python does not provide CUDA")
    result["package_tree_sha256"] = hashlib.sha256(json.dumps(result["package_files_sha256"], sort_keys=True).encode()).hexdigest()
    return result


def freeze_code(argv):
    parser = argparse.ArgumentParser(description="Capture package code, versions and bundle manifest before the clean run")
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--package-python", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    software = collect_software(args.package_python)
    record = {"schema_version": 1, "captured_utc": datetime.now(timezone.utc).isoformat(),
              "bundle_manifest_sha256": sha256(args.bundle / "manifest.json"), "software": software}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"code_manifest": str(args.output), "package_tree_sha256": software["package_tree_sha256"]}))
    return 0


def validate(args, manifest, fresh_preflight, software):
    bundle = args.bundle.resolve()
    require(manifest.get("standalone_verified") is False and manifest.get("candidate_bundle") is True,
            "Only an explicitly unverified candidate may be promoted")
    _, scope_errors = validate_scope_audit(bundle, manifest.get("runtime_profile"), manifest.get("inactive_command_audit"))
    require(not scope_errors and manifest.get("runtime_profile") is not None, f"Invalid scoped bundle: {scope_errors}")
    rows = manifest.get("files", [])
    require(rows and len({row["path"] for row in rows}) == len(rows), "Missing/duplicate bundle inventory")
    for row in rows:
        path = bundle / row["path"]
        require(path.resolve().is_relative_to(bundle) and path.is_file() and sha256(path) == row["sha256"],
                f"Bundle file missing or changed: {row['path']}")
    tracked = {row["path"] for row in rows}
    aliases = {bundle / "lib" / name: bundle / "lib" / Path(row["resolved_source"]).name
               for row in manifest.get("libraries", []) if not row["system_runtime"] for name in row["names"]}
    for parent, directories, files in os.walk(bundle, followlinks=False):
        for name in directories + files:
            path = Path(parent) / name
            relative = str(path.relative_to(bundle))
            if path.is_symlink():
                target = aliases.get(path)
                require(target is not None and path.resolve() == target and target.is_file()
                        and str(target.relative_to(bundle)) in tracked, f"Untracked bundle symlink: {relative}")
            elif path.is_file() and relative != "manifest.json" and not relative.startswith("metadata/"):
                require(relative in tracked, f"Untracked bundle file: {relative}")
    reports = {name: load_report(getattr(args, name)) for name in ("preflight", "run", "comparison", "aggregate")}
    for preflight in (reports["preflight"], fresh_preflight):
        require(preflight.get("hash_and_linkage_passed") is True and preflight.get("script_resource_checks_passed") is True
                and preflight.get("errors") == [] and preflight.get("script_resource_closure", {}).get("errors") == []
                and preflight.get("checked_staged_files") == sum(row.get("staged") is True for row in rows),
                "Static preflight is incomplete, failed or for a different inventory")
    run, comparison, aggregate = (reports[name] for name in ("run", "comparison", "aggregate"))
    require(type(run.get("return_code")) is int and run["return_code"] == 0 and run.get("missing_outputs") == [],
            "Clean reconstruction did not complete successfully")
    require(Path(run["bundle"]).resolve() == bundle and run.get("input_sha256") == sha256(args.input), "Run bundle/input differs")
    require(run.get("threads") == 4 and run.get("itk_threads") == 1 and run.get("parallel_hemispheres") is True
            and str(run.get("device", "")).startswith("cuda"), "Run is outside the fixed CUDA profile")
    require(math.isfinite(run["elapsed_seconds"]) and run["elapsed_seconds"] > 0, "Invalid run elapsed time")
    require(datetime.fromisoformat(run["started_utc"]).tzinfo is not None, "Run start time needs a timezone")
    frozen = load_report(args.code_manifest)
    captured = datetime.fromisoformat(frozen["captured_utc"])
    require(captured.tzinfo is not None and captured <= datetime.fromisoformat(run["started_utc"]),
            "Code snapshot was not captured before the run")
    require(frozen.get("bundle_manifest_sha256") == sha256(bundle / "manifest.json"),
            "Bundle manifest differs from the pre-run snapshot")
    require(frozen.get("software") == software and bool(software.get("package_files_sha256")),
            "Package code/runtime versions differ from the pre-run snapshot")
    candidate = Path(run["subject_dir"]).resolve()
    require(candidate.name == run["subject"], "Run subject name and directory differ")
    expected_config = manifest["inactive_command_audit"]["resolved_config"]["sha256"]
    require(run.get("effective_config_matches_profile") is True and run.get("effective_config_sha256") == expected_config
            and sha256(candidate / "scripts/recon-config.yaml") == expected_config, "Effective configuration does not match")
    require(Path(comparison["candidate"]).resolve() == candidate and Path(comparison["reference"]).resolve() != candidate,
            "Comparator candidate differs from the run, or is the reference itself")
    checks = comparison.get("checks", {})
    require(set(checks) == required_checks() and len(checks) == 52 and comparison.get("passed") is True
            and comparison.get("failed_checks") == [] and all(row.get("status") == "passed" for row in checks.values()),
            "All 52 required comparator checks must pass")
    require(comparison.get("tolerances") == load_report(args.tolerances), "Comparator uses different numerical tolerances")
    for name in ["mri/aseg.mgz", "mri/aparc+aseg.mgz", *[f"label/{hemi}.{atlas}.annot"
                 for hemi in ("lh", "rh") for atlas in ("aparc", "aparc.DKTatlas", "aparc.a2009s")]]:
        require(0.995 <= checks[name].get("min_dice", -1) <= 1, f"Relaxed or missing Dice threshold: {name}")
    outputs = required_checks() | {"mri/orig.mgz", "mri/ribbon.mgz", "mri/wmparc.mgz", "surf/lh.sphere.reg", "surf/rh.sphere.reg"}
    output_hashes = {}
    for name in sorted(outputs):
        path = candidate / name
        require(path.is_file() and path.resolve().is_relative_to(candidate), f"Required output missing or external: {name}")
        require(path.stat().st_mtime_ns <= args.comparison.stat().st_mtime_ns, f"Output changed after comparator report: {name}")
        output_hashes[name] = sha256(path)
    recalculated = check_report(comparison)
    require(recalculated["passed"] is True and len(recalculated["checks"]) == 19,
            "Recomputed aggregate gates failed")
    require(aggregate.get("comparison_sha256") == sha256(args.comparison)
            and all(aggregate.get(key) == value for key, value in recalculated.items()), "Aggregate report is stale or inconsistent")
    launch = validate_launch(args.trace_command_file, args.trace_cwd.resolve(), args, run)
    trace = trace_audit(args.trace, bundle, Path(manifest["fs_home"]).resolve(),
                        args.license_file.resolve(), candidate, args.input.resolve())
    evidence = {name: {"path": str(getattr(args, name).resolve()), "sha256": sha256(getattr(args, name))}
                for name in ("preflight", "run", "comparison", "aggregate", "trace", "trace_command_file", "tolerances", "code_manifest")}
    return {"schema_version": 1, "verified_utc": datetime.now(timezone.utc).isoformat(),
            "profile_id": PROFILE["id"], "evidence": evidence, "trace_audit": trace, "trace_launch": launch,
            "software": software, "candidate_outputs_sha256": output_hashes,
            "numerical_criteria": "Provisional FS8.2 numerical/aggregate profile; not bitwise identity",
            "redistribution_review_complete": manifest.get("redistribution_review_complete", False)}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "freeze-code":
        return freeze_code(argv[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "preflight", "run", "comparison", "aggregate", "trace", "trace-command-file",
                 "trace-cwd", "package-python", "input", "license-file", "code-manifest"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--tolerances", type=Path,
                        default=Path(__file__).resolve().parents[2] / "tests/recon_all/tolerances_numeric.json")
    parser.add_argument("--check-only", action="store_true", help="Check evidence without changing the manifest")
    args = parser.parse_args(argv)
    try:
        manifest_path = args.bundle.resolve() / "manifest.json"
        original = manifest_path.read_bytes()
        manifest = json.loads(original)
        software = collect_software(args.package_python)
        with tempfile.TemporaryDirectory(prefix="fs-promotion-") as temporary:
            fresh_path = Path(temporary) / "preflight.json"
            subprocess.run([str(args.package_python.absolute()), str(Path(__file__).with_name("preflight_bundle.py")),
                            "--bundle", str(args.bundle.resolve()), "--package-python", str(args.package_python.absolute()),
                            "--output", str(fresh_path)], check=True)
            fresh = load_report(fresh_path)
            verification = validate(args, manifest, fresh, software)
            verification["original_manifest_sha256"] = hashlib.sha256(original).hexdigest()
            verification["promotion_preflight_sha256"] = sha256(fresh_path)
            require(manifest_path.read_bytes() == original, "Bundle manifest changed during promotion")
            if not args.check_only:
                metadata = args.bundle / "metadata"
                metadata.mkdir(exist_ok=True)
                (metadata / "manifest.before-promotion.json").write_bytes(original)
                (metadata / "preflight.promotion.json").write_bytes(fresh_path.read_bytes())
                manifest.update(standalone_verified=True, verification=verification)
                temporary_manifest = manifest_path.with_suffix(".promoting.tmp")
                temporary_manifest.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
                os.replace(temporary_manifest, manifest_path)
        print(json.dumps({"eligible": True, "standalone_verified": not args.check_only,
                          "manifest": str(manifest_path)}))
        return 0
    except (OSError, ValueError, KeyError, TypeError, IndexError, subprocess.SubprocessError) as error:
        print(json.dumps({"eligible": False, "error": str(error)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
