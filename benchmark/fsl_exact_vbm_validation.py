#!/usr/bin/env python3
"""Validate FLIRT, FNIRT, and the shared FastVBM registration chain.

The harness is intentionally separate from package code.  It consumes the ten
private ``caseNN`` directories created by the UKB-style FSL reference run and
writes all subject-level data below ``--work-dir/private``.  The public JSON
and CSV contain aggregate distributions only.

Three layers localise differences:

``matched_affine``
    FSL FAST GM and the official FSL FLIRT matrix are shared.  This isolates
    nonlinear registration, common applywarp, Jacobian, and modulation.
``matched_gm``
    FSL FAST GM is shared and the package's FSL-target FLIRT is used.  This
    tests the complete affine plus nonlinear registration chain.
``end_to_end``
    Raw T1 is processed by package SynthStrip, TorchFAST, FLIRT, and the chosen
    nonlinear backend.

Both nonlinear backends record the same explicit official reference mask in
their common registration context. FNIRT consumes that mask inside its
estimator; SynthMorph has no mask input. The package-reported
``pre_nonlinear_signature`` is retained per case and its cross-backend equality
is an explicit release gate. CUDA timings synchronize both boundaries; output
serialization is measured separately.

Examples
--------
First inspect the inputs without running a model::

    python benchmark/fsl_exact_vbm_validation.py dry-run \
      --study-root work/ukb_vbm_gpu \
      --reference-mask "$FSLDIR/data/standard/MNI152_T1_2mm_brain_mask_dil.nii.gz" \
      --weights weights --work-dir work/ukb_vbm_gpu/fsl_exact_v09

Run all three layers and both nonlinear backends::

    python benchmark/fsl_exact_vbm_validation.py run \
      --study-root work/ukb_vbm_gpu \
      --reference-mask "$FSLDIR/data/standard/MNI152_T1_2mm_brain_mask_dil.nii.gz" \
      --weights weights --work-dir work/ukb_vbm_gpu/fsl_exact_v09 \
      --device cuda:0

Create the private and aggregate-only reports::

    python benchmark/fsl_exact_vbm_validation.py summarize \
      --study-root work/ukb_vbm_gpu \
      --reference-mask "$FSLDIR/data/standard/MNI152_T1_2mm_brain_mask_dil.nii.gz" \
      --weights weights \
      --work-dir work/ukb_vbm_gpu/fsl_exact_v09
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import time
from typing import Any, Iterable

import nibabel as nib
import numpy as np
import surfa as sf
import torch

import freesurfer_torch
from freesurfer_torch.fast_vbm import FastVBM, flirt_to_world_pull
from freesurfer_torch.fast_vbm.registration import register_gm
from freesurfer_torch.fast_vbm.synthmorph_backend import (
    SynthMorphDeformRegistration,
)
from freesurfer_torch.fnirt import TorchFNIRT

try:
    # Final public layout used by the 0.9 release.
    from freesurfer_torch.flirt import TorchFLIRT as ExactTargetFLIRT
except ImportError:
    # Transitional location while the exact-target port is being validated.
    from freesurfer_torch.fast_vbm.fsl_flirt import FSLFLIRT as ExactTargetFLIRT


SCHEMA_VERSION = 1
LAYERS = ("matched_affine", "matched_gm", "end_to_end")
BACKENDS = ("fnirt", "synthmorph")
OUTPUT_FILENAMES = {
    "warped_gm": "T1_GM_to_template_GM.nii.gz",
    "jacobian": "T1_GM_JAC_nl.nii.gz",
    "modulated_gm": "T1_GM_to_template_GM_mod.nii.gz",
}
DICE_THRESHOLDS = {
    "warped_gm": 0.2,
    "jacobian": 1.0,
    "modulated_gm": 0.2,
}
PREPROCESS_STAGES = (
    "robustfov",
    "crop",
    "bet",
    "standard_space_roi",
    "flirt_xyztrans",
    "xfm_inverse",
    "xfm_concat",
    "fnirt_t1",
    "invwarp",
    "mask_to_native",
    "brain",
    "fast",
)
EXPECTED_WEIGHTS = ("synthstrip.1.pt", "synthmorph.deform.3.h5")
PRIVATE_ID = re.compile(r"case[0-9]+")

# Keep these engineering gates fixed during a validation run. Passing them is
# evidence of functional agreement, not mathematical or bitwise identity.
FLIRT_RMSDIFF_MAX_MM = 0.05
FNIRT_FUNCTIONAL_OUTPUT_GATES = {
    "warped_gm": {
        "pearson": (">=", 0.99),
        "mae": ("<=", 0.01),
        "rmse": ("<=", 0.03),
        "dice": (">=", 0.98),
    },
    "jacobian": {
        "pearson": (">=", 0.99),
        "mae": ("<=", 0.02),
        "rmse": ("<=", 0.05),
        "dice": (">=", 0.98),
    },
    "modulated_gm": {
        "pearson": (">=", 0.99),
        "mae": ("<=", 0.02),
        "rmse": ("<=", 0.05),
        "dice": (">=", 0.97),
    },
}


@dataclass(frozen=True)
class CasePaths:
    """Private paths for one de-identified validation slot."""

    case_id: str
    root: Path
    raw_t1: Path
    fsl_gm: Path
    fsl_matrix: Path
    fsl_warped_gm: Path
    fsl_jacobian: Path
    fsl_modulated_gm: Path
    fsl_timing: Path


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def atomic_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(_jsonable(value), indent=2, allow_nan=False) + "\n"
    )
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_source_digest() -> str:
    root = Path(freesurfer_torch.__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def script_digest() -> str:
    return sha256(Path(__file__).resolve())


def array_fingerprint(value: np.ndarray) -> str:
    """Use the same array hash contract as ``pre_nonlinear_signature``."""
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed(device: torch.device, function):
    synchronize(device)
    started = time.perf_counter()
    value = function()
    synchronize(device)
    return value, time.perf_counter() - started


def tf32_state(device: torch.device) -> dict[str, Any]:
    return {
        "cuda_execution": device.type == "cuda",
        "matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn": bool(torch.backends.cudnn.allow_tf32),
        "reduced_precision_tensor_dtype": False,
    }


def execution_environment(device: torch.device) -> dict[str, Any]:
    return {
        "platform": platform.system(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
        ),
    }


def affine_rmsdiff_mm(first: np.ndarray, second: np.ndarray, radius=80.0) -> float:
    """Return the MISCMATHS affine RMS displacement used by FLIRT."""
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if first.shape != (4, 4) or second.shape != (4, 4):
        raise ValueError("affine RMS difference requires two 4x4 matrices")
    difference = first @ np.linalg.inv(second) - np.eye(4, dtype=np.float64)
    linear = difference[:3, :3]
    translation = difference[:3, 3]
    return float(
        np.sqrt(
            translation @ translation
            + (float(radius) ** 2 / 5.0) * np.trace(linear.T @ linear)
        )
    )


def resolve_template(args) -> Path:
    if args.template is not None:
        return Path(args.template)
    return Path(args.study_root) / "assets" / "template_GM_v1.nii.gz"


def resolve_reference_mask(args) -> Path:
    if args.reference_mask is not None:
        return Path(args.reference_mask)
    fsldir = os.environ.get("FSLDIR")
    if fsldir:
        return Path(fsldir) / "data/standard/MNI152_T1_2mm_brain_mask_dil.nii.gz"
    raise ValueError("--reference-mask is required when FSLDIR is not set")


def case_paths(root: Path, case_id: str) -> CasePaths:
    case_root = Path(root) / "subjects" / case_id
    t1 = case_root / "T1"
    reference = t1 / "T1_vbm" / "ukb"
    return CasePaths(
        case_id=case_id,
        root=case_root,
        raw_t1=t1 / "T1_orig.nii.gz",
        fsl_gm=t1 / "T1_fast" / "T1_brain_pve_1.nii.gz",
        fsl_matrix=reference / "T1_GM_to_template_GM.mat",
        fsl_warped_gm=reference / OUTPUT_FILENAMES["warped_gm"],
        fsl_jacobian=reference / OUTPUT_FILENAMES["jacobian"],
        fsl_modulated_gm=reference / OUTPUT_FILENAMES["modulated_gm"],
        fsl_timing=case_root / "timings.private.json",
    )


def discover_cases(study_root: Path, case_count: int) -> list[CasePaths]:
    subjects = Path(study_root) / "subjects"
    identifiers = sorted(
        path.name
        for path in subjects.iterdir()
        if path.is_dir() and PRIVATE_ID.fullmatch(path.name)
    )
    if len(identifiers) < case_count:
        raise ValueError(
            f"found {len(identifiers)} caseNN directories; expected at least {case_count}"
        )
    return [case_paths(study_root, value) for value in identifiers[:case_count]]


def required_case_files(case: CasePaths) -> dict[str, Path]:
    return {
        "raw_t1": case.raw_t1,
        "fsl_fast_gm": case.fsl_gm,
        "fsl_flirt_matrix": case.fsl_matrix,
        "fsl_warped_gm": case.fsl_warped_gm,
        "fsl_jacobian": case.fsl_jacobian,
        "fsl_modulated_gm": case.fsl_modulated_gm,
        "fsl_timing": case.fsl_timing,
    }


def validate_grid(template_path: Path, mask_path: Path) -> tuple[int, str]:
    template = nib.load(str(template_path))
    mask = nib.load(str(mask_path))
    if template.shape[:3] != mask.shape[:3] or not np.allclose(
        template.affine, mask.affine, atol=1e-5, rtol=0
    ):
        raise ValueError("official reference mask must use the template grid")
    mask_data = np.asarray(mask.dataobj)
    if mask_data.ndim == 4 and mask_data.shape[-1] == 1:
        mask_data = mask_data[..., 0]
    if mask_data.ndim != 3 or not np.isfinite(mask_data).all():
        raise ValueError("official reference mask must be one finite 3D image")
    binary = np.asarray(mask_data > 0, dtype=np.uint8)
    if not np.any(binary):
        raise ValueError("official reference mask is empty")
    return int(binary.sum()), array_fingerprint(binary)


def validate_inputs(
    args, *, require_weights: bool, check_device: bool = True
) -> dict[str, Any]:
    study_root = Path(args.study_root)
    template = resolve_template(args)
    reference_mask = resolve_reference_mask(args)
    for path in (study_root, template, reference_mask):
        if not path.exists():
            raise FileNotFoundError(path)
    cases = discover_cases(study_root, args.case_count)
    missing = []
    for case in cases:
        missing.extend(
            str(path)
            for path in required_case_files(case).values()
            if not path.is_file()
        )
    if missing:
        raise FileNotFoundError("missing validation inputs: " + ", ".join(missing))
    weights = Path(args.weights) if getattr(args, "weights", None) else None
    weight_files = {}
    if require_weights:
        if weights is None or not weights.is_dir():
            raise FileNotFoundError(weights or "--weights")
        for filename in EXPECTED_WEIGHTS:
            path = weights / filename
            if not path.is_file():
                raise FileNotFoundError(path)
            weight_files[filename] = {
                "path_private": str(path.resolve()),
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
            }
    mask_voxels, mask_array_hash = validate_grid(template, reference_mask)
    if check_device:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
    return {
        "study_root": study_root,
        "template": template,
        "reference_mask": reference_mask,
        "cases": cases,
        "weights": weights,
        "weight_files": weight_files,
        "mask_voxels": mask_voxels,
        "mask_array_sha256": mask_array_hash,
    }


def candidate_paths(work_dir: Path, layer: str, backend: str, case_id: str):
    directory = Path(work_dir) / "private" / "candidates" / layer / backend / case_id
    return {
        "directory": directory,
        "record": directory / "run.private.json",
        **{name: directory / filename for name, filename in OUTPUT_FILENAMES.items()},
    }


def flirt_paths(work_dir: Path, case_id: str):
    directory = Path(work_dir) / "private" / "flirt" / case_id
    return {
        "directory": directory,
        "record": directory / "run.private.json",
        "matrix": directory / "T1_GM_to_template_GM.mat",
        "moved": directory / "T1_GM_to_template_GM_affine.nii.gz",
    }


def signature(payload: dict[str, Any]) -> str:
    encoded = json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def case_input_hashes(case: CasePaths) -> dict[str, str]:
    return {
        name: sha256(path) for name, path in required_case_files(case).items()
    }


def validation_context(
    args,
    inputs: dict[str, Any],
    *,
    source_digest: str,
    harness_digest: str,
    tf32: dict[str, Any],
    environment: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": SCHEMA_VERSION,
        "script_sha256": harness_digest,
        "package_source_sha256": source_digest,
        "package_version": freesurfer_torch.__version__,
        "template_sha256": sha256(inputs["template"]),
        "official_reference_mask_sha256": sha256(inputs["reference_mask"]),
        "official_reference_mask_array_sha256": inputs["mask_array_sha256"],
        "weights_sha256": {
            name: metadata["sha256"]
            for name, metadata in inputs["weight_files"].items()
        },
        "device": args.device,
        "threads": args.threads,
        "tf32": tf32,
        "execution_environment": environment,
    }


def flirt_provenance(
    context: dict[str, Any], case_hashes: dict[str, str]
) -> dict[str, Any]:
    return {
        **context,
        "case_inputs_sha256": case_hashes,
        "algorithm": "FSLFLIRT exact-target default 12-DOF correlation-ratio path",
    }


def run_manifest_provenance(
    context: dict[str, Any],
    cases: list[CasePaths],
    case_hashes: dict[str, dict[str, str]],
    layers: Iterable[str],
    backends: Iterable[str],
) -> dict[str, Any]:
    return {
        **context,
        "case_inputs_sha256": {
            case.case_id: case_hashes[case.case_id] for case in cases
        },
        "layers": list(layers),
        "backends": list(backends),
    }


def completed_record(
    path: Path, expected_signature: str, outputs: dict[str, Path]
):
    if not Path(path).is_file():
        return None
    record = json.loads(Path(path).read_text())
    if record.get("status") != "success":
        return None
    if record.get("run_signature") != expected_signature:
        raise RuntimeError(
            f"stale successful record has a different signature: {path}; "
            "use --overwrite after reviewing the changed provenance"
        )
    expected_hashes = record.get("output_sha256")
    if not isinstance(expected_hashes, dict) or set(expected_hashes) != set(outputs):
        raise RuntimeError(f"cached output manifest is incomplete: {path}")
    for name, value in outputs.items():
        if not Path(value).is_file():
            return None
        if sha256(value) != expected_hashes[name]:
            raise RuntimeError(
                f"cached output hash mismatch for {value}; use --overwrite"
            )
    return record


def run_flirt(
    args,
    case: CasePaths,
    template: sf.Volume,
    context: dict[str, Any],
    case_hashes: dict[str, str],
):
    paths = flirt_paths(args.work_dir, case.case_id)
    provenance = flirt_provenance(context, case_hashes)
    run_signature = signature(provenance)
    if not args.overwrite:
        cached = completed_record(
            paths["record"],
            run_signature,
            {"matrix": paths["matrix"], "moved": paths["moved"]},
        )
        if cached is not None:
            return cached, True
    paths["directory"].mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for name in ("record", "matrix", "moved"):
            paths[name].unlink(missing_ok=True)
    record = {
        "status": "running",
        "case_id": case.case_id,
        "run_signature": run_signature,
        "provenance_private": {
            **provenance,
            "moving_gm": str(case.fsl_gm.resolve()),
            "template": str(resolve_template(args).resolve()),
            "official_matrix": str(case.fsl_matrix.resolve()),
        },
    }
    atomic_json(paths["record"], record)
    try:
        def compute():
            moving = sf.load_volume(str(case.fsl_gm))
            return ExactTargetFLIRT(device=args.device)(moving, template)

        result, compute_sec = timed(
            torch.device(args.device),
            compute,
        )
        save_started = time.perf_counter()
        np.savetxt(paths["matrix"], np.asarray(result.matrix), fmt="%.12g")
        result.moved.save(paths["moved"])
        save_sec = time.perf_counter() - save_started
        official = np.loadtxt(case.fsl_matrix, dtype=np.float64)
        record.update(
            status="success",
            synchronized_compute_sec=float(compute_sec),
            output_save_sec=float(save_sec),
            matrix_rmsdiff_mm=affine_rmsdiff_mm(result.matrix, official),
            qc=_jsonable(result.qc),
            output_sha256={
                "matrix": sha256(paths["matrix"]),
                "moved": sha256(paths["moved"]),
            },
            finished_at=time.time(),
        )
    except Exception as error:
        record.update(
            status="failed",
            error_private=f"{type(error).__name__}: {error}",
            finished_at=time.time(),
        )
        atomic_json(paths["record"], record)
        raise
    atomic_json(paths["record"], record)
    return record, False


def build_deform_model(backend: str, args):
    if backend == "synthmorph":
        return SynthMorphDeformRegistration(
            weights=args.weights,
            device=args.device,
            extent=256,
            hyper=0.5,
            steps=7,
        )
    return TorchFNIRT(device=args.device)


def initial_pull_from_flirt(matrix, moving: sf.Volume, fixed: sf.Volume):
    return flirt_to_world_pull(
        np.asarray(matrix, dtype=np.float64),
        moving.geom.vox2world.matrix,
        fixed.geom.vox2world.matrix,
        moving.shape[:3],
        fixed.shape[:3],
        moving.geom.voxsize,
        fixed.geom.voxsize,
    )


def save_vbm_outputs(result, paths) -> float:
    started = time.perf_counter()
    result.warped_gm.save(paths["warped_gm"])
    result.jacobian.save(paths["jacobian"])
    result.modulated_gm.save(paths["modulated_gm"])
    return time.perf_counter() - started


def selected_registration_metadata(registration) -> dict[str, Any]:
    qc = registration.qc
    return {
        "pre_nonlinear_signature": qc.get("pre_nonlinear_signature"),
        "reference_mask_source": qc.get("reference_mask_source"),
        "shared_pre_nonlinear_inputs": qc.get("shared_pre_nonlinear_inputs"),
        "reference_mask_role": qc.get("reference_mask_role"),
        "only_backend_specific_stage": qc.get("only_backend_specific_stage"),
        "nonlinear_backend": qc.get("nonlinear_backend"),
        "fnirt_implementation": qc.get("fnirt_implementation"),
        "synthmorph_implementation": qc.get("synthmorph_implementation"),
        "linear": qc.get("linear"),
        "nonlinear_estimator_qc": qc.get("nonlinear_estimator_qc"),
        "native_jacobian_comparison": qc.get("native_jacobian_comparison"),
        "applywarp": qc.get("applywarp"),
    }


def candidate_provenance(
    args,
    case: CasePaths,
    layer: str,
    backend: str,
    context: dict[str, Any],
    case_hashes: dict[str, str],
) -> dict[str, Any]:
    moving_path = case.raw_t1 if layer == "end_to_end" else case.fsl_gm
    affine = "estimated-inside-end-to-end"
    if layer == "matched_affine":
        affine = sha256(case.fsl_matrix)
    elif layer == "matched_gm":
        affine = sha256(flirt_paths(args.work_dir, case.case_id)["matrix"])
    return {
        **context,
        "case_inputs_sha256": case_hashes,
        "layer": layer,
        "backend": backend,
        "moving_input_sha256": sha256(moving_path),
        "affine_input_sha256_or_role": affine,
    }


def run_candidate(
    args,
    case: CasePaths,
    layer: str,
    backend: str,
    template: sf.Volume,
    reference_mask: sf.Volume,
    deform_model,
    pipeline,
    context: dict[str, Any],
    case_hashes: dict[str, str],
):
    paths = candidate_paths(args.work_dir, layer, backend, case.case_id)
    provenance = candidate_provenance(
        args,
        case,
        layer,
        backend,
        context,
        case_hashes,
    )
    run_signature = signature(provenance)
    if not args.overwrite:
        cached = completed_record(
            paths["record"],
            run_signature,
            {name: paths[name] for name in OUTPUT_FILENAMES},
        )
        if cached is not None:
            return cached, True
    paths["directory"].mkdir(parents=True, exist_ok=True)
    for name in (*OUTPUT_FILENAMES, "record"):
        if args.overwrite and name in paths:
            paths[name].unlink(missing_ok=True)
    input_path = case.raw_t1 if layer == "end_to_end" else case.fsl_gm
    record = {
        "status": "running",
        "case_id": case.case_id,
        "layer": layer,
        "backend": backend,
        "run_signature": run_signature,
        "provenance_private": {
            **provenance,
            "input": str(input_path.resolve()),
            "template": str(resolve_template(args).resolve()),
            "reference_mask": str(resolve_reference_mask(args).resolve()),
        },
        "outputs_private": {
            name: str(paths[name].resolve()) for name in OUTPUT_FILENAMES
        },
    }
    atomic_json(paths["record"], record)
    device = torch.device(args.device)
    try:
        if layer == "end_to_end":
            result, compute_sec = timed(
                device,
                lambda: pipeline(
                    case.raw_t1,
                    template,
                    reference_mask=reference_mask,
                ),
            )
            registration = result.registration
            substage_timing = result.timing_sec
        else:
            def compute():
                moving = sf.load_volume(str(case.fsl_gm))
                if layer == "matched_affine":
                    matrix = np.loadtxt(case.fsl_matrix, dtype=np.float64)
                else:
                    matrix = np.loadtxt(
                        flirt_paths(args.work_dir, case.case_id)["matrix"],
                        dtype=np.float64,
                    )
                initial_pull = initial_pull_from_flirt(matrix, moving, template)
                return register_gm(
                    moving,
                    template,
                    device=args.device,
                    initial_pull=initial_pull,
                    initial_pull_convention="fixed-to-moving-world-ras",
                    reference_mask=reference_mask,
                    synthmorph_weights=args.weights,
                    registration_backend=backend,
                    deform_model=deform_model,
                )

            registration, compute_sec = timed(
                device,
                compute,
            )
            result = registration
            substage_timing = None
        save_sec = save_vbm_outputs(result, paths)
        metadata = selected_registration_metadata(registration)
        if metadata["pre_nonlinear_signature"] is None:
            raise RuntimeError("registration did not report pre_nonlinear_signature")
        if metadata["reference_mask_source"] != "explicit":
            raise RuntimeError(
                "registration did not record the explicit reference mask"
            )
        record.update(
            status="success",
            synchronized_compute_sec=float(compute_sec),
            output_save_sec=float(save_sec),
            total_compute_and_save_sec=float(compute_sec + save_sec),
            pipeline_substage_timing_sec=_jsonable(substage_timing),
            registration=_jsonable(metadata),
            output_sha256={
                name: sha256(paths[name]) for name in OUTPUT_FILENAMES
            },
            finished_at=time.time(),
        )
    except Exception as error:
        record.update(
            status="failed",
            error_private=f"{type(error).__name__}: {error}",
            finished_at=time.time(),
        )
        atomic_json(paths["record"], record)
        raise
    atomic_json(paths["record"], record)
    return record, False


def dry_run(args) -> int:
    inputs = validate_inputs(args, require_weights=True)
    report = {
        "status": "ready",
        "case_count": len(inputs["cases"]),
        "case_ids_private": [case.case_id for case in inputs["cases"]],
        "study_root_private": str(inputs["study_root"].resolve()),
        "template_private": str(inputs["template"].resolve()),
        "reference_mask_private": str(inputs["reference_mask"].resolve()),
        "reference_mask_voxels": inputs["mask_voxels"],
        "reference_mask_sha256": sha256(inputs["reference_mask"]),
        "script_sha256": script_digest(),
        "package_source_sha256": package_source_digest(),
        "weights_private": inputs["weight_files"],
        "case_inputs_sha256_private": {
            case.case_id: case_input_hashes(case) for case in inputs["cases"]
        },
        "layers": list(args.layers),
        "backends": list(args.backends),
        "device": args.device,
        "threads": args.threads,
        "required_files_per_case": len(required_case_files(inputs["cases"][0])),
        "planned_candidate_runs": (
            len(inputs["cases"]) * len(args.layers) * len(args.backends)
        ),
        "planned_standalone_flirt_runs": len(inputs["cases"]),
    }
    if args.output_json is not None:
        atomic_json(args.output_json, report)
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


def run(args) -> int:
    inputs = validate_inputs(args, require_weights=True)
    device = torch.device(args.device)
    torch.set_num_threads(args.threads)
    template = sf.load_volume(str(inputs["template"]))
    reference_mask = sf.load_volume(str(inputs["reference_mask"]))
    source_digest = package_source_digest()
    harness_digest = script_digest()
    setup = []
    deform_models = {}
    pipelines = {}
    for backend in args.backends:
        deform_models[backend], elapsed = timed(
            device, lambda selected=backend: build_deform_model(selected, args)
        )
        setup.append(
            {"backend": backend, "component": "matched-layer nonlinear model", "sec": elapsed}
        )
        if "end_to_end" in args.layers:
            pipelines[backend], elapsed = timed(
                device,
                lambda selected=backend: FastVBM(
                    device=args.device,
                    threads=args.threads,
                    synthstrip_weights=args.weights,
                    synthmorph_weights=args.weights,
                    bias_correction=True,
                    registration_backend=selected,
                ),
            )
            setup.append(
                {"backend": backend, "component": "end-to-end pipeline", "sec": elapsed}
            )
            # Both layers use the same already-loaded nonlinear implementation.
            # The estimator is stateless between subjects.
            pipelines[backend].deform_model = deform_models[backend]
    actual_tf32 = tf32_state(device)
    if device.type == "cuda" and not (
        actual_tf32["matmul"] and actual_tf32["cudnn"]
    ):
        raise RuntimeError("CUDA validation requires TF32 matmul and cuDNN enabled")
    environment = execution_environment(device)
    context = validation_context(
        args,
        inputs,
        source_digest=source_digest,
        harness_digest=harness_digest,
        tf32=actual_tf32,
        environment=environment,
    )
    hashes_by_case = {
        case.case_id: case_input_hashes(case) for case in inputs["cases"]
    }
    cache_hits = {"flirt": 0, "candidate": 0}
    started = time.perf_counter()
    for case in inputs["cases"]:
        flirt, cached = run_flirt(
            args, case, template, context, hashes_by_case[case.case_id]
        )
        cache_hits["flirt"] += int(cached)
        print(
            f"flirt {case.case_id}: {flirt['matrix_rmsdiff_mm']:.6g} mm, "
            f"{flirt['synchronized_compute_sec']:.3f}s",
            flush=True,
        )
        for layer in args.layers:
            for backend in args.backends:
                record, cached = run_candidate(
                    args,
                    case,
                    layer,
                    backend,
                    template,
                    reference_mask,
                    deform_models[backend],
                    pipelines.get(backend),
                    context,
                    hashes_by_case[case.case_id],
                )
                cache_hits["candidate"] += int(cached)
                print(
                    f"{layer} {backend} {case.case_id}: "
                    f"{record['synchronized_compute_sec']:.3f}s compute, "
                    f"{record['output_save_sec']:.3f}s save",
                    flush=True,
                )
        if device.type == "cuda":
            torch.cuda.empty_cache()
    synchronize(device)
    invocation_wall_sec = time.perf_counter() - started
    planned_records = len(inputs["cases"]) * (
        1 + len(args.layers) * len(args.backends)
    )
    total_cache_hits = cache_hits["flirt"] + cache_hits["candidate"]
    all_records_fresh = total_cache_hits == 0
    manifest_provenance = run_manifest_provenance(
        context,
        inputs["cases"],
        hashes_by_case,
        args.layers,
        args.backends,
    )
    manifest = {
        "status": "success",
        "schema": SCHEMA_VERSION,
        "package_version": freesurfer_torch.__version__,
        "package_source_sha256": source_digest,
        "script_sha256": harness_digest,
        "run_signature": signature(manifest_provenance),
        "provenance_private": manifest_provenance,
        "case_count": len(inputs["cases"]),
        "case_ids_private": [case.case_id for case in inputs["cases"]],
        "layers": list(args.layers),
        "backends": list(args.backends),
        "device": args.device,
        "threads": args.threads,
        "tf32": actual_tf32,
        "execution_environment": environment,
        "constructor_setup_sec": setup,
        "planned_record_count": planned_records,
        "cache_hits": {**cache_hits, "total": total_cache_hits},
        "fresh_record_count": planned_records - total_cache_hits,
        "all_records_fresh": all_records_fresh,
        "invocation_wall_sec": invocation_wall_sec,
        "batch_wall_sec": invocation_wall_sec if all_records_fresh else None,
        "timing_contract": (
            "CUDA is synchronized immediately before and after each compute call; "
            "NIfTI and matrix writes are measured separately; batch_wall_sec is "
            "reported only when every record was computed in this invocation"
        ),
    }
    atomic_json(Path(args.work_dir) / "private" / "run.private.json", manifest)
    return 0


def load_image(path: Path, template_image=None) -> tuple[nib.spatialimages.SpatialImage, np.ndarray]:
    image = nib.load(str(path))
    data = np.asarray(image.dataobj, dtype=np.float32)
    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]
    if data.ndim != 3 or not np.isfinite(data).all():
        raise ValueError(f"{path} must contain one finite 3D image")
    if template_image is not None and (
        image.shape[:3] != template_image.shape[:3]
        or not np.allclose(image.affine, template_image.affine, atol=1e-5, rtol=0)
    ):
        raise ValueError(f"template-grid mismatch: {path}")
    return image, data


def pearson(reference: np.ndarray, candidate: np.ndarray) -> float:
    first = np.asarray(reference, dtype=np.float64)
    second = np.asarray(candidate, dtype=np.float64)
    first = first - first.mean()
    second = second - second.mean()
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if denominator == 0:
        raise ValueError("Pearson correlation is undefined for a constant image")
    return float(first.dot(second) / denominator)


def image_metrics(
    reference: np.ndarray,
    candidate: np.ndarray,
    mask: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    valid = mask & np.isfinite(reference) & np.isfinite(candidate)
    if int(valid.sum()) < 3:
        raise ValueError("fewer than three finite evaluation voxels")
    first = reference[valid].astype(np.float64)
    second = candidate[valid].astype(np.float64)
    difference = second - first
    first_binary = first >= threshold
    second_binary = second >= threshold
    denominator = int(first_binary.sum() + second_binary.sum())
    return {
        "pearson": pearson(first, second),
        "mae": float(np.abs(difference).mean()),
        "rmse": float(np.sqrt(np.square(difference).mean())),
        "dice": (
            float(
                2
                * np.logical_and(first_binary, second_binary).sum()
                / denominator
            )
            if denominator
            else 1.0
        ),
    }


def distribution(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=np.float64)
    if not array.size or not np.isfinite(array).all():
        raise ValueError("distribution values must be finite and nonempty")
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "median": float(np.median(array)),
        "q1": float(np.percentile(array, 25)),
        "q3": float(np.percentile(array, 75)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def aggregate_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("cannot aggregate an empty record list")
    result = {}
    for key in records[0]:
        values = [record[key] for record in records]
        if isinstance(values[0], dict):
            result[key] = aggregate_metrics(values)
        else:
            result[key] = distribution(values)
    return result


def fsl_timing(case: CasePaths) -> dict[str, float]:
    values = json.loads(case.fsl_timing.read_text())
    preprocessing = sum(float(values[name]) for name in PREPROCESS_STAGES)
    registration = float(values["fsl_reg_ukb"]) + float(values["modulate_ukb"])
    return {
        "preprocessing_through_fast_sec": preprocessing,
        "gm_registration_and_modulation_sec": registration,
        "raw_t1_through_modulated_gm_sec": preprocessing + registration,
    }


def candidate_timing(record, flirt_record=None) -> dict[str, float]:
    compute = float(record["synchronized_compute_sec"])
    save = float(record["output_save_sec"])
    result = {
        "registration_or_pipeline_compute_sec": compute,
        "vbm_output_save_sec": save,
        "comparable_compute_sec": compute,
        "all_output_save_sec": save,
    }
    if flirt_record is not None:
        result.update(
            flirt_compute_sec=float(flirt_record["synchronized_compute_sec"]),
            flirt_output_save_sec=float(flirt_record["output_save_sec"]),
        )
        result["comparable_compute_sec"] += result["flirt_compute_sec"]
        result["all_output_save_sec"] += result["flirt_output_save_sec"]
    result["compute_and_save_sec"] = (
        result["comparable_compute_sec"] + result["all_output_save_sec"]
    )
    return result


def compare(operator: str, observed: float, threshold: float) -> bool:
    if operator == "<=":
        return observed <= threshold
    if operator == ">=":
        return observed >= threshold
    if operator == "==":
        return observed == threshold
    raise ValueError(f"unsupported gate operator: {operator}")


def gate_record(
    name: str,
    observed: float | int | bool | None,
    operator: str,
    threshold: float | int | bool,
    *,
    scope: str,
    note: str,
) -> dict[str, Any]:
    if observed is None:
        outcome = "not_evaluated"
    else:
        outcome = "passed" if compare(operator, observed, threshold) else "failed"
    return {
        "name": name,
        "scope": scope,
        "operator": operator,
        "threshold": threshold,
        "observed": observed,
        "outcome": outcome,
        "note": note,
    }


def evaluate_gates(
    args,
    flirt_values: list[float],
    raw_metrics: dict[str, dict[str, list[dict[str, Any]]]],
    paired_signatures: dict[str, list[bool]],
    explicit_mask_flags: list[bool],
    mask_hash_flags: list[bool],
    fnirt_numerical_equivalence_flags: list[bool],
) -> list[dict[str, Any]]:
    gates = [
        gate_record(
            "flirt_matrix_rmsdiff_mm_maximum",
            max(flirt_values) if flirt_values else None,
            "<=",
            FLIRT_RMSDIFF_MAX_MM,
            scope="FLIRT exact-target component",
            note="worst case across the selected real-data cohort",
        ),
        gate_record(
            "official_reference_mask_recorded_for_every_candidate",
            bool(explicit_mask_flags) and all(explicit_mask_flags),
            "==",
            True,
            scope="shared FastVBM chain",
            note=(
                "FNIRT consumes this mask; SynthMorph records it in the common "
                "context because that model has no mask input"
            ),
        ),
        gate_record(
            "official_reference_mask_array_hash_matches_every_candidate",
            bool(mask_hash_flags) and all(mask_hash_flags),
            "==",
            True,
            scope="shared FastVBM chain",
            note="checks the binarized mask included in pre_nonlinear_signature",
        ),
        gate_record(
            "fnirt_implementation_reports_numerical_equivalence",
            (
                all(fnirt_numerical_equivalence_flags)
                if fnirt_numerical_equivalence_flags
                else None
            ),
            "==",
            True,
            scope="FNIRT numerical-equivalence claim",
            note=(
                "remains false until direct coefficient, iout, jout, and "
                "modulated-GM oracle tests pass"
            ),
        ),
    ]
    for layer in args.layers:
        values = paired_signatures.get(layer, [])
        gates.append(
            gate_record(
                f"{layer}_pre_nonlinear_signature_equal_across_backends",
                (all(values) if values else None),
                "==",
                True,
                scope="shared FastVBM chain",
                note=(
                    "same moving GM, template, affine, mask, geometry, and common "
                    "preparation; only nonlinear estimation may differ"
                ),
            )
        )
    for layer in ("matched_affine", "matched_gm"):
        records = raw_metrics.get(layer, {}).get("fnirt", [])
        for output, specifications in FNIRT_FUNCTIONAL_OUTPUT_GATES.items():
            for metric, (operator, threshold) in specifications.items():
                values = [record[output][metric] for record in records]
                observed = None
                if values:
                    observed = min(values) if operator == ">=" else max(values)
                gates.append(
                    gate_record(
                        f"{layer}_fnirt_{output}_{metric}_worst_case",
                        observed,
                        operator,
                        threshold,
                        scope="FNIRT functional scalar-output gate",
                        note=(
                            "FSL FAST GM reference; matched_affine isolates nonlinear "
                            "registration and matched_gm also includes package FLIRT"
                        ),
                    )
                )
    return gates


CSV_FIELDS = (
    "category",
    "layer",
    "backend",
    "output",
    "metric",
    "n",
    "mean",
    "std",
    "median",
    "q1",
    "q3",
    "minimum",
    "maximum",
    "operator",
    "threshold",
    "outcome",
)


def append_distributions(
    rows: list[dict[str, Any]],
    value: dict[str, Any],
    *,
    category: str,
    layer: str,
    backend: str,
    output: str = "",
    prefix: str = "",
) -> None:
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict) and set(item) == {
            "n",
            "mean",
            "std",
            "median",
            "q1",
            "q3",
            "minimum",
            "maximum",
        }:
            rows.append(
                {
                    "category": category,
                    "layer": layer,
                    "backend": backend,
                    "output": output,
                    "metric": name,
                    **item,
                }
            )
        elif isinstance(item, dict):
            next_output = output
            next_prefix = name
            if not output and key in OUTPUT_FILENAMES:
                next_output = key
                next_prefix = prefix
            append_distributions(
                rows,
                item,
                category=category,
                layer=layer,
                backend=backend,
                output=next_output,
                prefix=next_prefix,
            )


def privacy_check(public_json: Path, public_csv: Path, private_roots: Iterable[Path]):
    text = Path(public_json).read_text() + Path(public_csv).read_text()
    forbidden = ["/cwStorage/", "\\cwStorage\\", "case_id", "case_ids"]
    forbidden.extend(str(Path(path).resolve()) for path in private_roots)
    if any(value and value in text for value in forbidden):
        raise RuntimeError("public report contains a private path or identifier field")
    if PRIVATE_ID.search(text):
        raise RuntimeError("public report contains a caseNN identifier")


def validated_run_manifest(args, inputs: dict[str, Any]):
    path = Path(args.work_dir) / "private" / "run.private.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    manifest = json.loads(path.read_text())
    if manifest.get("status") != "success":
        raise RuntimeError(f"incomplete validation run manifest: {path}")
    cases = inputs["cases"]
    expected_case_ids = [case.case_id for case in cases]
    for key, expected in (
        ("case_ids_private", expected_case_ids),
        ("layers", list(args.layers)),
        ("backends", list(args.backends)),
        ("device", args.device),
        ("threads", args.threads),
    ):
        if manifest.get(key) != expected:
            raise RuntimeError(f"run manifest {key} does not match summary request")
    recorded = manifest.get("provenance_private")
    if not isinstance(recorded, dict):
        raise RuntimeError("run manifest is missing provenance_private")
    recorded_tf32 = recorded.get("tf32")
    recorded_environment = recorded.get("execution_environment")
    if not isinstance(recorded_tf32, dict) or not isinstance(
        recorded_environment, dict
    ):
        raise RuntimeError("run manifest is missing execution TF32 or hardware")
    if torch.device(args.device).type == "cuda" and not (
        recorded_tf32.get("matmul") and recorded_tf32.get("cudnn")
    ):
        raise RuntimeError("recorded CUDA run did not enable required TF32 defaults")
    context = validation_context(
        args,
        inputs,
        source_digest=package_source_digest(),
        harness_digest=script_digest(),
        tf32=recorded_tf32,
        environment=recorded_environment,
    )
    hashes_by_case = {case.case_id: case_input_hashes(case) for case in cases}
    expected = run_manifest_provenance(
        context,
        cases,
        hashes_by_case,
        args.layers,
        args.backends,
    )
    if recorded != expected or manifest.get("run_signature") != signature(expected):
        raise RuntimeError("run manifest provenance does not match current inputs")
    if manifest.get("tf32") != recorded_tf32 or manifest.get(
        "execution_environment"
    ) != recorded_environment:
        raise RuntimeError("run manifest execution metadata is internally inconsistent")
    cache_hits = manifest.get("cache_hits")
    if not isinstance(cache_hits, dict):
        raise RuntimeError("run manifest is missing cache accounting")
    try:
        flirt_hits = int(cache_hits["flirt"])
        candidate_hits = int(cache_hits["candidate"])
        total_hits = int(cache_hits["total"])
        planned = int(manifest["planned_record_count"])
        fresh = int(manifest["fresh_record_count"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("run manifest cache accounting is incomplete") from error
    expected_planned = len(cases) * (1 + len(args.layers) * len(args.backends))
    if (
        min(flirt_hits, candidate_hits, total_hits, planned, fresh) < 0
        or total_hits != flirt_hits + candidate_hits
        or planned != expected_planned
        or fresh != planned - total_hits
    ):
        raise RuntimeError("run manifest cache accounting is inconsistent")
    all_fresh = manifest.get("all_records_fresh") is True
    if all_fresh != (total_hits == 0):
        raise RuntimeError("run manifest cache accounting is inconsistent")
    if not all_fresh and manifest.get("batch_wall_sec") is not None:
        raise RuntimeError("resumed runs cannot report a complete cohort batch wall")
    if all_fresh and not isinstance(manifest.get("batch_wall_sec"), (int, float)):
        raise RuntimeError("fresh runs must report the complete cohort batch wall")
    return manifest, context, hashes_by_case


def summarize(args) -> int:
    inputs = validate_inputs(args, require_weights=True, check_device=False)
    run_manifest, context, hashes_by_case = validated_run_manifest(args, inputs)
    template_image, _ = load_image(inputs["template"])
    _, mask_values = load_image(inputs["reference_mask"], template_image)
    evaluation_mask = mask_values > 0
    expected_mask_array_hash = inputs["mask_array_sha256"]

    private_cases = []
    raw_metrics: dict[str, dict[str, list[dict[str, Any]]]] = {}
    raw_timings: dict[str, dict[str, list[dict[str, float]]]] = {}
    raw_fsl_timings = []
    flirt_values = []
    paired_signatures: dict[str, list[bool]] = {layer: [] for layer in args.layers}
    explicit_mask_flags = []
    mask_hash_flags = []
    fnirt_numerical_equivalence_flags = []
    execution_source_digests = set()

    for case in inputs["cases"]:
        reference = {}
        for output, path in (
            ("warped_gm", case.fsl_warped_gm),
            ("jacobian", case.fsl_jacobian),
            ("modulated_gm", case.fsl_modulated_gm),
        ):
            _, reference[output] = load_image(path, template_image)
        fsl_times = fsl_timing(case)
        raw_fsl_timings.append(fsl_times)

        fpaths = flirt_paths(args.work_dir, case.case_id)
        expected_flirt_signature = signature(
            flirt_provenance(context, hashes_by_case[case.case_id])
        )
        flirt_record = completed_record(
            fpaths["record"],
            expected_flirt_signature,
            {"matrix": fpaths["matrix"], "moved": fpaths["moved"]},
        )
        if flirt_record is None:
            raise FileNotFoundError(
                f"no complete current-signature FLIRT result: {fpaths['record']}"
            )
        candidate_matrix = np.loadtxt(fpaths["matrix"], dtype=np.float64)
        official_matrix = np.loadtxt(case.fsl_matrix, dtype=np.float64)
        rmsdiff = affine_rmsdiff_mm(candidate_matrix, official_matrix)
        flirt_values.append(rmsdiff)
        execution_source_digests.add(
            flirt_record["provenance_private"]["package_source_sha256"]
        )

        case_private = {
            "case_id": case.case_id,
            "reference_private": {
                name: {
                    "path": str(path.resolve()),
                    "sha256": sha256(path),
                }
                for name, path in required_case_files(case).items()
            },
            "flirt": {
                "matrix_rmsdiff_mm": rmsdiff,
                "timing": candidate_timing(flirt_record),
                "record_private": str(fpaths["record"].resolve()),
            },
            "candidates": {},
        }
        signatures_for_layer: dict[str, dict[str, str]] = {}
        for layer in args.layers:
            for backend in args.backends:
                paths = candidate_paths(args.work_dir, layer, backend, case.case_id)
                expected_candidate_signature = signature(
                    candidate_provenance(
                        args,
                        case,
                        layer,
                        backend,
                        context,
                        hashes_by_case[case.case_id],
                    )
                )
                record = completed_record(
                    paths["record"],
                    expected_candidate_signature,
                    {name: paths[name] for name in OUTPUT_FILENAMES},
                )
                if record is None:
                    raise FileNotFoundError(
                        "no complete current-signature candidate result: "
                        f"{paths['record']}"
                    )
                execution_source_digests.add(
                    record["provenance_private"]["package_source_sha256"]
                )
                candidate = {}
                for output in OUTPUT_FILENAMES:
                    _, values = load_image(paths[output], template_image)
                    candidate[output] = image_metrics(
                        reference[output],
                        values,
                        evaluation_mask,
                        DICE_THRESHOLDS[output],
                    )
                shared = record["registration"]["pre_nonlinear_signature"]
                signatures_for_layer.setdefault(layer, {})[backend] = shared[
                    "combined_sha256"
                ]
                explicit = record["registration"]["reference_mask_source"] == "explicit"
                correct_mask = (
                    shared.get("reference_mask_sha256") == expected_mask_array_hash
                )
                explicit_mask_flags.append(explicit)
                mask_hash_flags.append(correct_mask)
                if backend == "fnirt":
                    estimator_qc = record["registration"].get(
                        "nonlinear_estimator_qc"
                    ) or {}
                    fnirt_numerical_equivalence_flags.append(
                        bool(
                            estimator_qc.get(
                                "fsl_fnirt_numerically_equivalent", False
                            )
                        )
                    )
                timing = candidate_timing(
                    record, flirt_record if layer == "matched_gm" else None
                )
                raw_metrics.setdefault(layer, {}).setdefault(backend, []).append(candidate)
                raw_timings.setdefault(layer, {}).setdefault(backend, []).append(timing)
                case_private["candidates"][f"{layer}/{backend}"] = {
                    "metrics": candidate,
                    "timing": timing,
                    "pre_nonlinear_signature": shared,
                    "explicit_reference_mask": explicit,
                    "official_reference_mask_hash_matches": correct_mask,
                    "record_private": str(paths["record"].resolve()),
                }
        if set(args.backends) == set(BACKENDS):
            for layer in args.layers:
                values = signatures_for_layer[layer]
                paired_signatures[layer].append(
                    values["fnirt"] == values["synthmorph"]
                )
        private_cases.append(case_private)
        atomic_json(
            Path(args.work_dir)
            / "private"
            / "summaries"
            / case.case_id
            / "summary.private.json",
            case_private,
        )

    if len(execution_source_digests) != 1:
        raise RuntimeError(
            "candidate records were produced by multiple package source snapshots"
        )
    execution_source_digest = next(iter(execution_source_digests))

    aggregate_results = {}
    rows: list[dict[str, Any]] = []
    for layer in args.layers:
        aggregate_results[layer] = {}
        for backend in args.backends:
            accuracy = aggregate_metrics(raw_metrics[layer][backend])
            timing = aggregate_metrics(raw_timings[layer][backend])
            aggregate_results[layer][backend] = {
                "case_count": len(raw_metrics[layer][backend]),
                "accuracy": accuracy,
                "timing": timing,
            }
            append_distributions(
                rows,
                accuracy,
                category="candidate_accuracy",
                layer=layer,
                backend=backend,
            )
            append_distributions(
                rows,
                timing,
                category="candidate_timing",
                layer=layer,
                backend=backend,
            )

    flirt_distribution = distribution(flirt_values)
    append_distributions(
        rows,
        {"matrix_rmsdiff_mm": flirt_distribution},
        category="flirt_accuracy",
        layer="matched_gm",
        backend="pytorch_flirt",
    )
    fsl_timing_distribution = aggregate_metrics(raw_fsl_timings)
    append_distributions(
        rows,
        fsl_timing_distribution,
        category="reference_timing",
        layer="reference",
        backend="fsl_cpu",
    )

    paired_speed = {}
    for layer, reference_key in (
        ("matched_gm", "gm_registration_and_modulation_sec"),
        ("end_to_end", "raw_t1_through_modulated_gm_sec"),
    ):
        if layer not in args.layers:
            continue
        paired_speed[layer] = {}
        reference_values = [value[reference_key] for value in raw_fsl_timings]
        for backend in args.backends:
            compute_values = [
                value["comparable_compute_sec"]
                for value in raw_timings[layer][backend]
            ]
            compute_and_save_values = [
                value["compute_and_save_sec"]
                for value in raw_timings[layer][backend]
            ]
            primary_ratios = [
                reference / candidate
                for reference, candidate in zip(
                    reference_values, compute_and_save_values
                )
            ]
            primary_differences = [
                reference - candidate
                for reference, candidate in zip(
                    reference_values, compute_and_save_values
                )
            ]
            compute_only_ratios = [
                reference / candidate
                for reference, candidate in zip(reference_values, compute_values)
            ]
            compute_only_differences = [
                reference - candidate
                for reference, candidate in zip(reference_values, compute_values)
            ]
            paired_speed[layer][backend] = {
                "fsl_observed_wall_over_candidate_compute_and_save": distribution(
                    primary_ratios
                ),
                "fsl_observed_wall_minus_candidate_compute_and_save_sec": distribution(
                    primary_differences
                ),
                "fsl_observed_wall_over_candidate_compute_only": distribution(
                    compute_only_ratios
                ),
                "fsl_observed_wall_minus_candidate_compute_only_sec": distribution(
                    compute_only_differences
                ),
            }
            append_distributions(
                rows,
                paired_speed[layer][backend],
                category="paired_timing",
                layer=layer,
                backend=backend,
            )

    gates = evaluate_gates(
        args,
        flirt_values,
        raw_metrics,
        paired_signatures,
        explicit_mask_flags,
        mask_hash_flags,
        fnirt_numerical_equivalence_flags,
    )
    for gate in gates:
        rows.append(
            {
                "category": "release_gate",
                "layer": gate["scope"],
                "backend": "",
                "output": "",
                "metric": gate["name"],
                "operator": gate["operator"],
                "threshold": gate["threshold"],
                "minimum": gate["observed"],
                "maximum": gate["observed"],
                "outcome": gate["outcome"],
            }
        )

    hardware = {
        **context["execution_environment"],
        "device": context["device"],
        "tf32": context["tf32"],
        "threads": context["threads"],
    }
    public = {
        "schema": SCHEMA_VERSION,
        "package_version": freesurfer_torch.__version__,
        "feature": "FSL exact-target registration and FastVBM shared-chain validation",
        "cohort": {
            "modality": "real T1w",
            "case_count": len(inputs["cases"]),
            "individual_metrics_published": False,
            "identifiers_published": False,
        },
        "provenance": {
            "reference_pipeline": "UKB v1.5-method VBM reference outputs",
            "reference_software_context": "FSL 6.0.7.4 project environment",
            "template_sha256": sha256(inputs["template"]),
            "official_reference_mask_sha256": sha256(inputs["reference_mask"]),
            "candidate_execution_package_source_sha256": execution_source_digest,
            "summarizer_package_source_sha256": package_source_digest(),
            "source_unchanged_at_summarize": (
                execution_source_digest == package_source_digest()
            ),
            "fsl_source_targets": {
                "flirt": "2111.2",
                "fnirt": "2203.0",
            },
        },
        "evaluation": {
            "candidate_implementations": {
                "affine": "freesurfer_torch.flirt.TorchFLIRT exact-target port",
                "fnirt": "freesurfer_torch.fnirt.TorchFNIRT",
                "synthmorph": (
                    "freesurfer_torch.fast_vbm.SynthMorphDeformRegistration "
                    "using the package PyTorch SynthMorph model"
                ),
                "pipeline": "freesurfer_torch.fast_vbm.FastVBM",
            },
            "layers": {
                "matched_affine": "FSL FAST GM plus official FSL FLIRT matrix",
                "matched_gm": "FSL FAST GM plus package FSL-target FLIRT",
                "end_to_end": "raw T1 plus package SynthStrip, TorchFAST, and FLIRT",
            },
            "common_after_nonlinear_estimation": (
                "FSL-coordinate field conversion, package GPU applywarp, dense "
                "nonlinear-only Jacobian, and modulation"
            ),
            "only_backend_specific_stage": (
                "nonlinear pull-field estimation, including estimator-specific "
                "objective and mask use"
            ),
            "reference_mask_role": (
                "FNIRT estimator input and common recorded context; SynthMorph "
                "has no reference-mask input"
            ),
            "evaluation_mask": "official MNI152_T1_2mm_brain_mask_dil",
            "evaluation_mask_voxels": inputs["mask_voxels"],
            "metrics": ["Pearson", "MAE", "RMSE", "thresholded Dice"],
            "dice_thresholds": DICE_THRESHOLDS,
            "matched_affine_timing_boundary": (
                "candidate nonlinear/common-chain timing is reported, but no isolated "
                "FSL nonlinear-only timing exists in timings.private.json"
            ),
        },
        "execution": {
            "hardware": hardware,
            "timing_contract": (
                "candidate compute calls synchronize CUDA at both boundaries; model "
                "construction is separate and output writes are reported separately. "
                "Primary paired values use candidate compute plus the three selected "
                "VBM output saves"
            ),
            "constructor_setup_sec": run_manifest["constructor_setup_sec"],
            "invocation_wall_sec": float(run_manifest["invocation_wall_sec"]),
            "batch_wall_sec": run_manifest["batch_wall_sec"],
            "all_records_fresh": run_manifest["all_records_fresh"],
            "cache_hits": run_manifest["cache_hits"],
            "first_call_boundary": (
                "end-to-end first calls include lazy SynthStrip loading; nonlinear "
                "models are constructed in the separately reported setup"
            ),
        },
        "flirt": {
            "matrix_contract": "input to reference in FSL scaled-mm coordinates",
            "rmsdiff_radius_mm": 80.0,
            "matrix_rmsdiff_mm": flirt_distribution,
        },
        "reference_fsl_timing": fsl_timing_distribution,
        "reference_fsl_timing_provenance": {
            "source": "existing per-case timings.private.json files",
            "original_hardware_recorded": False,
            "original_thread_count_recorded": False,
            "fsl_version_recorded_per_case": False,
            "controlled_speed_claim_allowed": False,
            "interpretation": (
                "Observed historical wall times only; they are not a controlled "
                "CPU-versus-GPU speed benchmark"
            ),
        },
        "results": aggregate_results,
        "paired_timing": paired_speed,
        "paired_timing_interpretation": (
            "Descriptive ratios against historical FSL wall times with unrecorded "
            "original hardware and thread count and different intermediate-output I/O; "
            "no controlled speed claim"
        ),
        "shared_chain_audit": {
            "pre_nonlinear_signature_equal_count": {
                layer: int(sum(values)) for layer, values in paired_signatures.items()
            },
            "pre_nonlinear_signature_pair_count": {
                layer: len(values) for layer, values in paired_signatures.items()
            },
            "explicit_official_reference_mask_recorded_count": int(
                sum(explicit_mask_flags)
            ),
            "candidate_count": len(explicit_mask_flags),
            "official_mask_array_hash_match_count": int(sum(mask_hash_flags)),
            "fnirt_numerical_equivalence_true_count": int(
                sum(fnirt_numerical_equivalence_flags)
            ),
            "fnirt_candidate_count": len(fnirt_numerical_equivalence_flags),
        },
        "release_gates": {
            "basis": (
                "versioned engineering tolerances; a passed functional gate does "
                "not imply bitwise identity or by itself authorize a numerical-"
                "equivalence claim"
            ),
            "results": gates,
            "release_decision": "not made by this harness",
            "synthmorph_accuracy_role": (
                "descriptive FSL comparison because the nonlinear algorithm is intentionally different"
            ),
            "end_to_end_accuracy_role": (
                "descriptive attribution of upstream brain-extraction and GM-estimation differences"
            ),
        },
        "privacy": (
            "Aggregate-only public artifacts; paths, case labels, source identifiers, "
            "matrices, signatures, and individual measurements are excluded."
        ),
    }
    private = {
        "schema": SCHEMA_VERSION,
        "public_report_companion": str(Path(args.public_json).resolve()),
        "study_root_private": str(Path(args.study_root).resolve()),
        "template_private": str(inputs["template"].resolve()),
        "reference_mask_private": str(inputs["reference_mask"].resolve()),
        "cases": private_cases,
        "gates": gates,
    }
    atomic_json(args.private_json, private)
    atomic_json(args.public_json, public)
    Path(args.public_csv).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.public_csv).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    privacy_check(
        args.public_json,
        args.public_csv,
        (Path(args.study_root), Path(args.work_dir), inputs["template"], inputs["reference_mask"]),
    )
    print(
        json.dumps(
            {
                "status": "summarized",
                "case_count": len(inputs["cases"]),
                "private_json": str(args.private_json),
                "public_json": str(args.public_json),
                "public_csv": str(args.public_csv),
                "release_decision": "not made by this harness",
            }
        )
    )
    return 0


def add_common(parser: argparse.ArgumentParser, *, include_weights: bool) -> None:
    parser.add_argument(
        "--study-root",
        type=Path,
        required=True,
        help="directory containing subjects/caseNN and assets/template_GM_v1.nii.gz",
    )
    parser.add_argument(
        "--template",
        type=Path,
        help="GM template; defaults to STUDY_ROOT/assets/template_GM_v1.nii.gz",
    )
    parser.add_argument(
        "--reference-mask",
        type=Path,
        help="official FSL dilated 2-mm brain mask; defaults below FSLDIR",
    )
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--case-count", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--layers", nargs="+", choices=LAYERS, default=list(LAYERS))
    parser.add_argument(
        "--backends", nargs="+", choices=BACKENDS, default=list(BACKENDS)
    )
    if include_weights:
        parser.add_argument(
            "--weights",
            type=Path,
            required=True,
            help="directory containing official SynthStrip and SynthMorph checkpoints",
        )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    dry_parser = subparsers.add_parser("dry-run", help="validate every required input")
    add_common(dry_parser, include_weights=True)
    dry_parser.add_argument("--output-json", type=Path)
    dry_parser.set_defaults(function=dry_run)

    run_parser = subparsers.add_parser("run", help="write private per-case results")
    add_common(run_parser, include_weights=True)
    run_parser.add_argument("--overwrite", action="store_true")
    run_parser.set_defaults(function=run)

    summary_parser = subparsers.add_parser(
        "summarize", help="write private detail and aggregate-only public reports"
    )
    add_common(summary_parser, include_weights=True)
    summary_parser.add_argument("--private-json", type=Path)
    summary_parser.add_argument("--public-json", type=Path)
    summary_parser.add_argument("--public-csv", type=Path)
    summary_parser.set_defaults(function=summarize)

    args = parser.parse_args(argv)
    if args.case_count < 1 or args.threads < 1:
        parser.error("--case-count and --threads must be positive")
    if len(set(args.layers)) != len(args.layers):
        parser.error("--layers must not contain duplicates")
    if len(set(args.backends)) != len(args.backends):
        parser.error("--backends must not contain duplicates")
    args.study_root = Path(args.study_root)
    args.work_dir = Path(args.work_dir)
    if args.command == "summarize":
        if args.private_json is None:
            args.private_json = args.work_dir / "private" / "summary.private.json"
        if args.public_json is None:
            args.public_json = args.work_dir / "summary.public.json"
        if args.public_csv is None:
            args.public_csv = args.work_dir / "summary.public.csv"
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    return args.function(args)


if __name__ == "__main__":
    raise SystemExit(main())
