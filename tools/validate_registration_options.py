#!/usr/bin/env python3
"""Validate initialized registration, header-only output, and saved transforms.

Requires the artifacts produced by validate_synthmorph.py in validation/full192.
Only two affine pairs are inferred; resampling checks reuse their transforms.
"""
import argparse
import contextlib
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

import numpy as np
import surfa as sf
import torch

from fnit import SynthMorph, apply_transform
from validate_synthmorph import errors, geometry, write_json


def run(command, log, env, check=True):
    start = time.perf_counter()
    with open(log, "w") as stream:
        result = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
    details = {"command": command, "seconds": time.perf_counter()-start,
               "log": str(log), "returncode": result.returncode}
    if check and result.returncode:
        raise RuntimeError(f"Reference exited with code {result.returncode}; see {log}")
    return details


def patched_reference(root, env):
    """Copy the installed reference, changing exactly two composition inputs."""
    freesurfer = Path(env["FREESURFER_HOME"])
    source = freesurfer / "python/packages/synthmorph"
    destination = root / "work/reference_dtype_fix"
    package = destination / "synthmorph"
    shutil.copytree(source, package, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
    original = (source / "registration.py").read_text()
    anchor = "    fw = vxm.utils.compose((net_to_mov, fw, fix_to_net),"
    if original.count(anchor) != 1:
        raise ValueError("Installed registration.py does not match the audited composition site")
    inserted = "    net_to_mov = np.asarray(net_to_mov)\n    net_to_fix = np.asarray(net_to_fix)\n"
    patched = original.replace(anchor, inserted + anchor)
    (package / "registration.py").write_text(patched)
    patched_env = env.copy()
    extra_path = patched_env.get("FS_LOCAL_PYTHONPATH", "")
    patched_env["FS_LOCAL_PYTHONPATH"] = str(destination) + (os.pathsep + extra_path if extra_path else "")
    fspython = freesurfer / "bin/fspython"
    probe = subprocess.run([str(fspython), "-c",
                            "import importlib.util; print(importlib.util.find_spec('synthmorph.registration').origin)"],
                           env=patched_env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    origin = probe.stdout.strip().splitlines()[-1]
    if Path(origin).resolve() != (package / "registration.py").resolve():
        raise RuntimeError(f"FS_LOCAL_PYTHONPATH did not select the copied reference: {origin}")
    manifest = {"kind": "patched reference; not the unmodified original command",
                "source": str(source / "registration.py"), "copy": str(package / "registration.py"),
                "source_sha256": hashlib.sha256(original.encode()).hexdigest(),
                "copy_sha256": hashlib.sha256(patched.encode()).hexdigest(),
                "inserted_code": inserted.strip(), "insertion_site": "immediately before native-space compose",
                "reason": "-i leaves floating TensorFlow matrices as float64, while model fields/matrices are float32; NumPy inputs restore VoxelMorph compose's own float32 casting",
                "verified_import_origin": origin,
                "FS_LOCAL_PYTHONPATH": patched_env["FS_LOCAL_PYTHONPATH"],
                "installed_files_modified": False}
    write_json(destination / "manifest.json", manifest)
    return [str(fspython), str(freesurfer / "python/scripts/mri_synthmorph")], patched_env, manifest


def image_comparison(reference_path, candidate_path, bitwise=False):
    reference, candidate = [sf.load_volume(str(path)) for path in (reference_path, candidate_path)]
    data = errors(reference.data, candidate.data)
    geom = geometry(reference.geom, candidate.geom)
    exact = bool(np.array_equal(reference.data, candidate.data))
    dtype_equal = reference.dtype == candidate.dtype
    geometry_tolerance = 1e-5 if bitwise else 1e-3
    geometry_pass = (geom["reference_shape"] == geom["candidate_shape"]
                     and geom["vox2ras_max_abs"] <= geometry_tolerance)
    nrmse = data.get("normalised_rmse")
    data_pass = (exact and dtype_equal) if bitwise else (nrmse is not None and nrmse <= 1e-4)
    return {"data": data, "geometry": geom, "bitwise_equal": exact,
            "reference_dtype": str(reference.dtype), "candidate_dtype": str(candidate.dtype),
            "passed": bool(geometry_pass and data_pass)}


def affine_comparison(reference_path, candidate_path):
    reference, candidate = [sf.load_affine(str(path)).convert(space="world") for path in (reference_path, candidate_path)]
    data = errors(reference.matrix, candidate.matrix)
    source, target = geometry(reference.source, candidate.source), geometry(reference.target, candidate.target)
    return {"world_matrix": data, "source_geometry": source, "target_geometry": target,
            "passed": bool(data.get("max_abs", float("inf")) <= 1e-3 and source["equal"] and target["equal"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Project directory containing weights/ and validation/full192/")
    parser.add_argument("--outdir", type=Path, default=Path("validation/options"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    root = args.root.resolve()
    outdir = args.outdir if args.outdir.is_absolute() else root / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    if (outdir / "report.json").exists() and not (outdir / "report.before_dtype_patch.json").exists():
        shutil.copy2(outdir / "report.json", outdir / "report.before_dtype_patch.json")
    baseline = root / "validation" / "full192"
    moving, fixed = baseline / "case/moving.nii.gz", baseline / "case/fixed.nii.gz"
    initial = baseline / "affine/reference/forward.lta"
    warp_paths = [baseline / mode / "reference/forward.nii.gz" for mode in ("joint", "deform")]
    warp = next((path for path in warp_paths if path.is_file()), None)
    for path in (moving, fixed, initial):
        if not path.is_file():
            parser.error(f"Missing baseline artifact: {path}")
    if warp is None:
        parser.error("Missing baseline saved joint/deform forward warp")
    executable = shutil.which("mri_synthmorph")
    if not executable or not os.environ.get("FREESURFER_HOME"):
        parser.error("Load the original FreeSurfer module before running this script")
    env = os.environ.copy()
    env["NVIDIA_TF32_OVERRIDE"] = "0"
    gpu_flags = ["-g"] if args.device.startswith("cuda") else []
    if ":" in args.device and args.device.startswith("cuda"):
        env["CUDA_VISIBLE_DEVICES"] = args.device.split(":", 1)[1]
    if not gpu_flags:
        env["CUDA_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    report = {"scope": "Implementation equivalence on a public-template pair; no population accuracy claim",
              "inputs": {"moving": str(moving), "fixed": str(fixed), "init": str(initial), "warp": str(warp)},
              "acceptance": {"world_affine_max_abs": 1e-3, "world_translation_unit": "mm",
                             "linear_matrix_entries": "dimensionless; same 1e-3 tolerance",
                             "registration_image_normalised_rmse": 1e-4,
                             "image_normalisation": "reference root-mean-square",
                             "registration_header_vox2ras_max_abs": 1e-3,
                             "shared_transform_apply_data": "bitwise equality and identical dtype",
                             "shared_transform_apply_geometry_max_abs": 1e-5},
              "reference_environment": {"NVIDIA_TF32_OVERRIDE": "0"}, "cases": {}}
    patched_cli, patched_env, patch_manifest = patched_reference(root, env)
    report["reference_dtype_workaround"] = patch_manifest
    report["original_reference_success_is_separate"] = True
    report["passed_means"] = "Candidate checks passed against the explicitly designated reference; unmodified original failures remain separate"

    def record(name, entry):
        report["cases"][name] = entry
        report["passed"] = all(value.get("passed", False) for value in report["cases"].values())
        report["original_reference_failures"] = [key for key, value in report["cases"].items()
                                                   if value.get("original_reference", {}).get("returncode", 0)]
        write_json(outdir / "report.json", report)
        print(f"{name}: {'PASS' if entry.get('passed') else 'FAIL'}", flush=True)

    # Reuse one initialized model for exactly two network inferences.
    with open(outdir / "candidate_model_load.log", "w") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        start = time.perf_counter()
        model = SynthMorph(weights=root / "weights", device=args.device, model="affine", extent=192)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        report["candidate_model_load_seconds"] = time.perf_counter()-start
    for mid_space in (False, True):
        name = "affine_init_midspace" if mid_space else "affine_init"
        folder = outdir / name
        folder.mkdir(exist_ok=True)
        original_ref, ref, cand = folder / "original_reference", folder / "patched_reference", folder / "candidate"
        original_ref.mkdir(exist_ok=True)
        ref.mkdir(exist_ok=True)
        cand.mkdir(exist_ok=True)
        entry = {"init": str(initial), "mid_space": mid_space, "header_only": True}
        try:
            command = [executable, "register", "-m", "affine", "-e", "192", "-j", "4", "-i", str(initial),
                       "-H", "-o", str(original_ref / "header.nii.gz"), "-O", str(original_ref / "inverse_header.nii.gz"),
                       "-t", str(original_ref / "forward.lta"), "-T", str(original_ref / "inverse.lta"), *gpu_flags]
            if mid_space:
                command.append("-M")
            command.extend((str(moving), str(fixed)))
            entry["original_reference"] = run(command, original_ref / "run.log", env, check=False)
            entry["original_reference"]["succeeded"] = entry["original_reference"]["returncode"] == 0
            if entry["original_reference"]["returncode"]:
                original_error = (original_ref / "run.log").read_text(errors="replace")
                entry["original_reference"]["error_tail"] = original_error[-4000:]
            fixed_command = [*patched_cli, *[value.replace(str(original_ref), str(ref)) for value in command[1:]]]
            entry["patched_reference"] = run(fixed_command, ref / "run.log", patched_env)
            entry["comparison_reference"] = "task-local official source copy with two composition dtype conversions"
            with open(cand / "run.log", "w") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                start = time.perf_counter()
                result = model(moving, fixed, init=initial, mid_space=mid_space, header_only=True)
                if args.device.startswith("cuda"):
                    torch.cuda.synchronize()
                entry["candidate_pair_seconds"] = time.perf_counter()-start
                result.moved.save(cand / "header.nii.gz")
                result.fixed_moved.save(cand / "inverse_header.nii.gz")
                result.transform.save(cand / "forward.lta")
                result.inverse.save(cand / "inverse.lta")
            checks = {"forward_affine": affine_comparison(ref / "forward.lta", cand / "forward.lta"),
                      "inverse_affine": affine_comparison(ref / "inverse.lta", cand / "inverse.lta"),
                      "header_image": image_comparison(ref / "header.nii.gz", cand / "header.nii.gz"),
                      "inverse_header_image": image_comparison(ref / "inverse_header.nii.gz", cand / "inverse_header.nii.gz")}
            # Evaluate ordinary resampling from those transforms without repeating inference.
            command = [executable, "apply", str(ref / "forward.lta"), str(moving), str(ref / "moved.nii.gz")]
            entry["reference_resample"] = run(command, ref / "apply.log", env)
            apply_transform(moving, cand / "forward.lta").save(cand / "moved.nii.gz")
            checks["resampled_image"] = image_comparison(ref / "moved.nii.gz", cand / "moved.nii.gz")
            entry["checks"] = checks
            entry["passed"] = all(value["passed"] for value in checks.values())
        except Exception as exc:
            entry.update(passed=False, error=str(exc), traceback=traceback.format_exc())
        record(name, entry)
    del model
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    # Small integer labels with the exact native source geometry, so nearest
    # comparisons also exercise label dtype and fill conversion to uint8.
    source = sf.load_volume(str(moving))
    ijk = np.indices(source.shape)
    labels = ((ijk[0] // 7 + 3*(ijk[1] // 9) + 5*(ijk[2] // 6)) % 23).astype("int16")
    label_path = outdir / "source_labels.nii.gz"
    sf.Volume(labels, geometry=source.geom).save(label_path)
    for kind, trans in (("affine", initial), ("warp", warp)):
        for method in ("linear", "nearest"):
            for dtype in ("float32", "uint8"):
                name = f"apply_{kind}_{method}_{dtype}"
                folder = outdir / name
                folder.mkdir(exist_ok=True)
                reference, candidate = folder / "reference.nii.gz", folder / "candidate.nii.gz"
                entry = {"shared_transform": str(trans), "method": method, "dtype": dtype, "fill": -7}
                try:
                    command = [executable, "apply", "-m", method, "-t", dtype, "-f", "-7",
                               str(trans), str(label_path), str(reference)]
                    entry["reference"] = run(command, folder / "reference.log", env)
                    start = time.perf_counter()
                    apply_transform(label_path, trans, method=method, dtype=dtype, fill=-7).save(candidate)
                    entry["candidate_seconds"] = time.perf_counter()-start
                    entry["comparison"] = image_comparison(reference, candidate, bitwise=True)
                    entry["passed"] = entry["comparison"]["passed"]
                except Exception as exc:
                    entry.update(passed=False, error=str(exc), traceback=traceback.format_exc())
                record(name, entry)
    # Four frames with distinct intensities expose channel-axis mixups and exercise
    # the original CLI's multi-frame apply behavior (registration itself stays 3D).
    frames_path = outdir / "source_frames.nii.gz"
    frames = np.stack((labels, 2*labels+3, 22-labels, labels % 5), axis=-1).astype("float32")
    sf.Volume(frames, geometry=source.geom).save(frames_path)
    for kind, trans in (("affine", initial), ("warp", warp)):
        for method in ("linear", "nearest"):
            name = f"apply_4d_{kind}_{method}"
            folder = outdir / name
            folder.mkdir(exist_ok=True)
            reference, candidate = folder / "reference.nii.gz", folder / "candidate.nii.gz"
            entry = {"shared_transform": str(trans), "method": method, "dtype": "float32", "fill": -7, "frames": 4}
            try:
                command = [executable, "apply", "-m", method, "-t", "float32", "-f", "-7",
                           str(trans), str(frames_path), str(reference)]
                entry["reference"] = run(command, folder / "reference.log", env)
                apply_transform(frames_path, trans, method=method,
                                dtype="float32", fill=-7).save(candidate)
                entry["comparison"] = image_comparison(reference, candidate, bitwise=True)
                entry["passed"] = entry["comparison"]["passed"]
            except Exception as exc:
                entry.update(passed=False, error=str(exc), traceback=traceback.format_exc())
            record(name, entry)
    # Header-only apply must also honor the requested dtype, as the original CLI does.
    folder = outdir / "apply_header_uint8"
    folder.mkdir(exist_ok=True)
    entry = {"shared_transform": str(initial), "header_only": True, "dtype": "uint8"}
    try:
        reference, candidate = folder / "reference.nii.gz", folder / "candidate.nii.gz"
        command = [executable, "apply", "-H", "-t", "uint8", str(initial), str(label_path), str(reference)]
        entry["reference"] = run(command, folder / "reference.log", env)
        apply_transform(label_path, initial, header_only=True, dtype="uint8").save(candidate)
        entry["comparison"] = image_comparison(reference, candidate, bitwise=True)
        entry["passed"] = entry["comparison"]["passed"]
    except Exception as exc:
        entry.update(passed=False, error=str(exc), traceback=traceback.format_exc())
    record("apply_header_uint8", entry)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
