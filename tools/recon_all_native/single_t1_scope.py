"""Pinned source evidence for one fresh, unedited FS8.2 T1 reconstruction.

This is a finite branch review, not a general shell reachability analyser.
The runner must enforce PROFILE; direct use of bundle commands is outside it.
"""

import copy
import hashlib
import json
from pathlib import Path
import shutil


SOURCE_COMMIT = "d932c45b7941662ea380a05efef580568b98d41a"
PROFILE = {
    "id": "fs820-single-t1-v8-all-v1", "input_count": 1,
    "fresh_subject": True, "empty_subjects_dir": True,
    "extra_flags_allowed": False,
    "argv_suffix": ["-all", "-parallel", "-openmp", "4", "-itkthreads", "1"],
    "threads": 4, "itk_threads": 1, "target_system": "Linux",
}
REFERENCE_CONFIG = "etc/scoped-reference-recon-config.yaml"
REFERENCE_CONFIG_SHA256 = "892990ac5603bae47e22fbdac414c080178d5d53e47f4ede48906fbf56de03de"
CONFIG_VALUES = {"UseSynthSeg": True, "DoSynthSR": False, "UseStopMaskSCM": False}

# The installed recon-all and reviewed scripts match pinned source after CMake substitutes
# FS_VERSION=8.2.0 and BUILD_STAMP=freesurfer-linux-centos7_x86_64-8.2.0-20260314-d932c45.
# The first entry additionally fixes the generated dispatcher, which passes argv unchanged.
PINNED_FILES = {
    "bin/recon-all": "5ae865f81f10483a19aa17e42d426763c99c2f42d0e2891eca13ecb6bcd5a094",
    "build-stamp.txt": "8a1600fb20972b7518249f8f714a6fed51b1d1f21e17430f60f85c0eea48e277",
    "sources.csh": "53bb6fb7aa91914612cfb8e2370a3b8fcc7d5af8bdd50da0e1b2cfe2baf8765c",
    "etc/global-expert-options.v8.txt": "e93998087b2122cb20c0b2265659e62d04e99e1409ca0013de3b66ddd62def58",
    "etc/recon-config.yaml": "029ffc80d3d614d3236eab302e736e2f708cd7c537a170321455b75ef96818c5",
    "python/scripts/rca-config": "f3bb97d2f4b4d36265c0b06b29b11a176530637c1e034ce20ea049572a63484f",
    "python/scripts/rca-config2csh": "40c051f0e5fb70de4dba79434d83784b2501e3a9f91777fce75f9780088bc8ff",
    "upstream_scripts/bin/fsr-getxopts": "dd64ec180d2b3245ad006ce9ccfda80c32c1cb0a69e57e78f8e16d33b86c6415",
    "upstream_scripts/bin/recon-all": "55410a0dc005237f99a1a35fb5d41c4e1994b3ae189ca9232a59b87034f86415",
    "upstream_scripts/bin/fscalc": "250bfce67a46c73cfb3f121a655b6f597ae5841447ab50d32fcf373063182122",
    "upstream_scripts/bin/mri_motion_correct.fsl": "99ffd15633ce6ef54cb64b68b90972b0bbe628c91d4c0b0a5c22fe1744142bad",
    "upstream_scripts/bin/label-cortex": "a9391a5892cc9ffe7ed093e7ca6acd740a678111496d85ce942c08170d1767bb",
    "upstream_scripts/bin/defect2seg": "5e7d576dad34325633bb88cad5c6c55a31ae0e22ebb02a4753fb44a9318b3f39",
    "upstream_scripts/bin/fs-synthmorph-reg": "3195a0d42a6f9413bdaedb66a6d33bed7994cebc3f986d5db1da311aef81698e",
    "upstream_scripts/bin/tkmeditfv": "c2f12e394990af1ed008f21c5f95561050a3d2328f064aa5ad8325d909f06ef8",
    "upstream_scripts/bin/tkregisterfv": "da84771b9d6097fe35d28822b1fb02b5ca1fc7e8b750aa474e48b9824cf808c4",
}
GUI_COMMANDS = ("tkmeditfv", "tkregisterfv")


def gui_launcher(name):
    return ("#!/bin/sh\n"
            f"echo '{name}: GUI commands are unavailable in {PROFILE['id']}' >&2\n"
            "exit 64\n")


def evidence(script, lines, finding):
    return {"source": "scripts/" + script, "lines": lines, "finding": finding,
            "url": f"https://github.com/freesurfer/freesurfer/blob/{SOURCE_COMMIT}/scripts/{script}#L{lines[0]}"}


def rule(command, finding, ranges, *, script="recon-all", config=None, extra=()):
    return {"script": "upstream_scripts/bin/" + script, "command": command,
            "applicable_profile": PROFILE["id"], "condition": finding,
            "resolved_config_values": config or {},
            "source_evidence": [evidence(script, span, finding) for span in ranges] + list(extra)}


RULES = [
    rule("mri_add_new_tp", "longitudinal=0 for the fixed cross-sectional CLI", [[244, 244], [1195, 1206]]),
    rule("mri_compile_edits", "DoShowEdits=0; fixed -all does not enable it", [[80, 80], [1100, 1186], [7198, 7249]]),
    rule("mri_deface", "DoDeface=0; fixed -all does not enable it", [[328, 328], [1720, 1734], [7198, 7249]]),
    rule("mri_seg_diff", "SynthSeg skips CA-label edits; fresh subject has no aseg.manedit.mgz", [[2904, 2959], [3087, 3108]], config={"UseSynthSeg": True}),
    rule("mri_stopmask", "UseStopMaskSCM is false in the resolved pinned configuration", [[3896, 3923]], config={"UseStopMaskSCM": False}),
    rule("mri_synthsr", "DoSynthSR is false in the resolved pinned configuration", [[1513, 1534]], config={"DoSynthSR": False}),
    rule("mris_apply_reg", "UseHighMyelin=0; fixed -all does not enable it", [[191, 191], [4377, 4392], [7198, 7249]]),
    rule("mris_compute_lgi", "DoLocalGyriIndex=0; fixed -all does not enable it", [[378, 378], [5696, 5720], [7198, 7249]]),
    rule("mris_preproc", "DoQdecCache=0; fixed -all does not enable it", [[181, 181], [5727, 5848], [7198, 7249]]),
    rule("mris_reposition_surface", "Fresh unedited T1 subject has no repos.*.json; no T2/FLAIR input", [[4463, 4483], [4525, 4546], [4718, 4735]]),
    rule("mris_spherical_average", "DoLabelExvivoEC=0; fixed -all does not enable it", [[380, 380], [5642, 5689], [7198, 7249]]),
    rule("rca-base-init", "longitudinal=0 and DoCreateBaseSubj=0 for the fixed CLI", [[244, 253], [1195, 1225], [1339, 1355]]),
    rule("rca-long-tp-init", "longitudinal=0 for the fixed cross-sectional CLI", [[244, 244], [1195, 1206], [1227, 1232]]),
    rule("seg2recon", "Direct call is inside literal if(0 && ...) in the pinned source", [[1650, 1669]]),
    rule("mri_average", "Only -version reaches this helper: one input bypasses motion correction", [[45, 51]], script="mri_motion_correct.fsl", extra=[
        evidence("recon-all", [777, 777], "Startup calls mri_motion_correct.fsl -version"),
        evidence("recon-all", [1437, 1494], "One run is copied; processing requires more than one run")]),
    rule("mri_volsynth", "The reconstruction's fscalc caller supplies an existing nxmask volume, not a constant", [[53, 84]], script="fscalc", extra=[
        evidence("label-cortex", [164, 199], "mri_binarize creates nxmask; nonzero status exits before fscalc nxmask and nzmask")]),
]


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expected_audit():
    return {"schema_version": 1, "profile_id": PROFILE["id"], "source_commit": SOURCE_COMMIT,
            "files": [{"path": path, "sha256": digest} for path, digest in PINNED_FILES.items()],
            "resolved_config": {"path": REFERENCE_CONFIG, "sha256": REFERENCE_CONFIG_SHA256,
                                "values": copy.deepcopy(CONFIG_VALUES)},
            "rules": copy.deepcopy(RULES),
            "disabled_gui": [{"command": name, "upstream_path": "upstream_scripts/bin/" + name,
                              "upstream_sha256": PINNED_FILES["upstream_scripts/bin/" + name],
                              "path": "bin/" + name,
                              "sha256": hashlib.sha256(gui_launcher(name).encode()).hexdigest(),
                              "exit_code": 64,
                              "reason": "Fixed-profile GUI entry rejects execution; archived upstream GUI code is not dispatched"}
                             for name in GUI_COMMANDS],
            "gui_reference_evidence": [
                evidence("defect2seg", [155, 155], "tkmeditfv is printed by echo, not executed"),
                {"source": "upstream_scripts/bin/fs-synthmorph-reg", "lines": [393, 393],
                 "finding": "tkregisterfv is printed by echo as an optional QC command"},
                {"source": "upstream_scripts/bin/fs-synthmorph-reg", "lines": [475, 475],
                 "finding": "tkmeditfv is printed by echo as an optional QC command"}],
            "conditions": ["Use only the fixed package runner; no direct arbitrary recon-all CLI",
                           "One T1 in a newly created subject and empty subjects directory",
                           "No manual edits or concurrent changes to subject files during execution",
                           "Clean runner environment; no FS_V8_XOPTS, expert files or extra flags",
                           "Exact pinned V8 expert options, base YAML and configuration resolver"],
            "limitations": ["GUI entry points fail explicitly with exit 64", "No rule exempts a required version-inventory command",
                            "Static branch evidence does not establish full ELF or dynamic-command closure"]}


def validate_scope_audit(bundle, runtime_profile, audit):
    """Return only reviewed (script, command) pairs; any mismatch fails closed."""
    if audit is None and runtime_profile is None:
        return set(), []
    if (json.dumps(runtime_profile, sort_keys=True) != json.dumps(PROFILE, sort_keys=True)
            or json.dumps(audit, sort_keys=True) != json.dumps(expected_audit(), sort_keys=True)):
        return set(), [{"error": "Inactive-command audit or runtime profile differs from the reviewed scope"}]
    bundle = Path(bundle).resolve()
    errors = []
    for row in [*audit["files"], audit["resolved_config"], *audit["disabled_gui"]]:
        path = bundle / row["path"]
        if not path.resolve().is_relative_to(bundle) or not path.is_file() or _sha256(path) != row["sha256"]:
            errors.append({"file": row["path"], "error": "Scoped branch evidence missing or changed"})
    if errors:
        return set(), errors
    return {(row["script"], row["command"]) for row in RULES}, []


def _disable_gui(bundle, manifest):
    from runtime_dispatch import file_row

    for name in GUI_COMMANDS:
        relative = "bin/" + name
        source = "upstream_scripts/" + relative
        upstream = next(row for row in manifest["files"] if row["path"] == source)
        path = bundle / relative
        content = gui_launcher(name)
        if path.read_text() != content:
            old = next(row for row in manifest["files"] if row["path"] == relative)
            archive = bundle / "upstream_scripts/scoped_gui_dispatch" / relative
            if archive.exists():
                raise ValueError(f"Unexpected existing GUI dispatcher archive: {archive}")
            archive.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(path, archive)
            old.update(path=str(archive.relative_to(bundle)), executed=False,
                       execution_condition="Replaced with explicit scoped GUI rejection")
            path.write_text(content)
            path.chmod(0o755)
            manifest["files"].append(file_row(bundle, path, "generated_script",
                                              implementation="Scoped GUI rejection (exit 64)"))
        upstream.update(executed=False, execution_condition="GUI launcher unconditionally exits 64",
                        execution_evidence={"launcher_path": relative,
                                            "launcher_sha256": _sha256(path),
                                            "profile_id": PROFILE["id"]})
        manifest["script_dispatch"] = [row for row in manifest.get("script_dispatch", [])
                                       if row["path"] != relative]


def configure_scope(bundle, manifest, resolved_config):
    """Attach reviewed reference evidence to an unverified candidate."""
    bundle = Path(bundle).resolve()
    resolved_config = Path(resolved_config)
    if _sha256(resolved_config) != REFERENCE_CONFIG_SHA256:
        raise ValueError("Resolved configuration differs from the reviewed FS8.2 reference")
    audit = expected_audit()
    for row in audit["files"]:
        path = bundle / row["path"]
        if not path.resolve().is_relative_to(bundle) or not path.is_file() or _sha256(path) != row["sha256"]:
            raise ValueError(f"Pinned branch source/configuration differs: {row['path']}")
    _disable_gui(bundle, manifest)
    destination = bundle / REFERENCE_CONFIG
    destination.parent.mkdir(parents=True, exist_ok=True)
    if resolved_config.resolve() != destination.resolve():
        shutil.copyfile(resolved_config, destination)
    row = {"path": REFERENCE_CONFIG, "sha256": REFERENCE_CONFIG_SHA256,
           "bytes": destination.stat().st_size, "kind": "scope_evidence", "staged": True}
    manifest["files"] = [prior for prior in manifest["files"] if prior["path"] != REFERENCE_CONFIG] + [row]
    manifest["required_resources"] = sorted(set(manifest.get("required_resources", [])) | {REFERENCE_CONFIG})
    manifest["runtime_profile"] = copy.deepcopy(PROFILE)
    manifest["inactive_command_audit"] = audit
    manifest["standalone_verified"] = False
