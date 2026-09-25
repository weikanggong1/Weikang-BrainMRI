#!/usr/bin/env python3
"""Validate TorchFNIRT against FSL 6.0.7.4 on matched GM inputs.

The tool keeps case-level paths, identifiers, metrics, signatures, and images
below ``WORK_DIR/private``.  ``summarize`` writes a separate aggregate-only
public JSON without paths or case identifiers.

Examples
--------
Inspect the ten inputs and cache signatures without running registration::

    python tools/validate_fnirt_fsl.py dry-run \
      --study-root work/ukb_vbm_gpu \
      --work-dir work/ukb_vbm_gpu/fnirt_fsl_validation \
      --fsl-dir "$FSLDIR" --device cuda:1

Run or resume the private case-level comparison::

    python tools/validate_fnirt_fsl.py run \
      --study-root work/ukb_vbm_gpu \
      --work-dir work/ukb_vbm_gpu/fnirt_fsl_validation \
      --fsl-dir "$FSLDIR" --device cuda:1

Create the private detail report and aggregate-only public report::

    python tools/validate_fnirt_fsl.py summarize \
      --study-root work/ukb_vbm_gpu \
      --work-dir work/ukb_vbm_gpu/fnirt_fsl_validation \
      --fsl-dir "$FSLDIR" --device cuda:1
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Any, Iterable
import uuid

import nibabel as nib
import numpy as np
import torch

import freesurfer_torch
from freesurfer_torch.fnirt.standalone import (
    DEFAULT_REFERENCE_MASK,
    SUPPORTED_CONFIG,
    SUPPORTED_CONFIG_SHA256,
    run_fnirt,
)


SCHEMA_VERSION = 1
EXPECTED_FSL_VERSION = "6.0.7.4"
CASE_PATTERN = re.compile(r"case[0-9]+")
OUTPUT_KEYS = (
    "coefficients",
    "nonlinear_residual",
    "iout",
    "jout",
    "modulated_gm",
)
OUTPUT_PATH_KEYS = {
    "coefficients": "cout",
    "nonlinear_residual": "nonlinear_residual",
    "iout": "iout",
    "jout": "jout",
    "modulated_gm": "modulated_gm",
}
METRIC_KEYS = ("pearson", "mae", "rmse", "maxabs")


@dataclass(frozen=True)
class CaseInputs:
    case_id: str
    gm: Path
    affine: Path


@dataclass(frozen=True)
class FSLTools:
    root: Path
    fnirt: Path
    fnirtfileutils: Path
    config: Path
    version: str


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def atomic_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    )
    try:
        temporary.write_text(
            json.dumps(_jsonable(value), indent=2, allow_nan=False) + "\n"
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


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


def signature(value: dict[str, Any]) -> str:
    encoded = json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def tf32_policy(device: str | torch.device) -> dict[str, bool]:
    """Return the package execution policy without sampling a summary process."""
    enabled = torch.device(device).type == "cuda"
    return {
        "matmul": enabled,
        "cudnn": enabled,
        "reduced_precision_tensor_dtype": False,
    }


def apply_tf32_policy(device: str | torch.device) -> dict[str, bool]:
    policy = tf32_policy(device)
    if torch.device(device).type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    actual = {
        "matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn": bool(torch.backends.cudnn.allow_tf32),
        "reduced_precision_tensor_dtype": False,
    }
    if torch.device(device).type == "cuda" and actual != policy:
        raise RuntimeError(f"failed to apply CUDA TF32 policy: {actual}")
    return policy


def execution_order(case: CaseInputs) -> tuple[str, str]:
    index = int(case.case_id.removeprefix("case"))
    return ("fsl", "torch") if index % 2 else ("torch", "fsl")


def resolve_template(args) -> Path:
    if args.template is not None:
        return Path(args.template)
    return Path(args.study_root) / "assets" / "template_GM_v1.nii.gz"


def resolve_reference_mask(args) -> Path:
    if args.reference_mask is not None:
        return Path(args.reference_mask)
    fsl_dir = args.fsl_dir or os.environ.get("FSLDIR")
    if not fsl_dir:
        raise ValueError("--reference-mask or --fsl-dir/FSLDIR is required")
    return Path(fsl_dir) / "data" / "standard" / DEFAULT_REFERENCE_MASK


def resolve_fsl_tools(args) -> FSLTools:
    root_value = args.fsl_dir or os.environ.get("FSLDIR")
    if not root_value:
        raise ValueError("--fsl-dir or FSLDIR is required")
    root = Path(root_value)
    version_path = root / "etc" / "fslversion"
    fnirt = root / "bin" / "fnirt"
    fnirtfileutils = root / "bin" / "fnirtfileutils"
    config = (
        Path(args.config)
        if args.config is not None
        else root / "etc" / "flirtsch" / SUPPORTED_CONFIG
    )
    for path in (version_path, fnirt, fnirtfileutils, config):
        if not path.is_file():
            raise FileNotFoundError(path)
    version = version_path.read_text().strip()
    if version != EXPECTED_FSL_VERSION:
        raise ValueError(
            f"this validation requires FSL {EXPECTED_FSL_VERSION}; found {version}"
        )
    if sha256(config) != SUPPORTED_CONFIG_SHA256:
        raise ValueError(f"{config} is not the official {SUPPORTED_CONFIG}")
    return FSLTools(root, fnirt, fnirtfileutils, config, version)


def case_inputs(study_root: Path, case_id: str) -> CaseInputs:
    t1 = Path(study_root) / "subjects" / case_id / "T1"
    return CaseInputs(
        case_id=case_id,
        gm=t1 / "T1_fast" / "T1_brain_pve_1.nii.gz",
        affine=t1 / "T1_vbm" / "ukb" / "T1_GM_to_template_GM.mat",
    )


def discover_cases(study_root: Path, case_count: int) -> list[CaseInputs]:
    cases = [
        case_inputs(study_root, f"case{index:02d}")
        for index in range(1, case_count + 1)
    ]
    missing = [
        path
        for case in cases
        for path in (case.gm, case.affine)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "missing matched FNIRT inputs: " + ", ".join(map(str, missing))
        )
    return cases


def _load_3d(path: Path) -> tuple[nib.spatialimages.SpatialImage, np.ndarray]:
    image = nib.load(str(path))
    data = np.asarray(image.dataobj, dtype=np.float32)
    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]
    if data.ndim != 3 or not np.isfinite(data).all():
        raise ValueError(f"{path} must contain one finite 3D image")
    return image, data


def validate_grid(template_path: Path, mask_path: Path) -> dict[str, Any]:
    template, _ = _load_3d(template_path)
    mask_image, mask = _load_3d(mask_path)
    if template.shape[:3] != mask_image.shape[:3] or not np.allclose(
        template.affine, mask_image.affine, atol=1e-5, rtol=0
    ):
        raise ValueError("reference mask must use the template grid")
    if not np.all((mask == 0) | (mask == 1)) or not np.any(mask == 1):
        raise ValueError("reference mask must be a non-empty binary 0/1 image")
    return {
        "shape": list(template.shape[:3]),
        "voxel_sizes": list(template.header.get_zooms()[:3]),
        "mask_voxels": int(np.count_nonzero(mask)),
    }


def validate_case_input(case: CaseInputs) -> None:
    _load_3d(case.gm)
    affine = np.loadtxt(case.affine, dtype=np.float64)
    if affine.shape != (4, 4) or not np.isfinite(affine).all():
        raise ValueError(f"{case.affine} must contain one finite 4x4 matrix")
    if not np.allclose(affine[3], (0, 0, 0, 1), atol=1e-8, rtol=0):
        raise ValueError(f"{case.affine} is not a homogeneous FLIRT matrix")
    if abs(float(np.linalg.det(affine[:3, :3]))) < 1e-10:
        raise ValueError(f"{case.affine} is singular")


def validate_inputs(args, *, check_device: bool) -> dict[str, Any]:
    study_root = Path(args.study_root)
    work_dir = Path(args.work_dir)
    template = resolve_template(args)
    reference_mask = resolve_reference_mask(args)
    tools = resolve_fsl_tools(args)
    for path in (study_root, template, reference_mask):
        if not path.exists():
            raise FileNotFoundError(path)
    cases = discover_cases(study_root, args.case_count)
    grid = validate_grid(template, reference_mask)
    for case in cases:
        validate_case_input(case)
    device = torch.device(args.device)
    if check_device and device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        device_index = torch.cuda.current_device() if device.index is None else device.index
        if not 0 <= device_index < torch.cuda.device_count():
            raise ValueError(f"requested CUDA device does not exist: {args.device}")
    return {
        "study_root": study_root,
        "work_dir": work_dir,
        "template": template,
        "reference_mask": reference_mask,
        "tools": tools,
        "cases": cases,
        "grid": grid,
        "package_source_sha256": package_source_digest(),
        "script_sha256": sha256(Path(__file__)),
        "template_sha256": sha256(template),
        "reference_mask_sha256": sha256(reference_mask),
        "config_sha256": sha256(tools.config),
        "fnirt_sha256": sha256(tools.fnirt),
        "fnirtfileutils_sha256": sha256(tools.fnirtfileutils),
    }


def case_provenance(args, inputs: dict[str, Any], case: CaseInputs) -> dict[str, Any]:
    return {
        "schema": SCHEMA_VERSION,
        "script_sha256": inputs["script_sha256"],
        "package_version": freesurfer_torch.__version__,
        "package_source_sha256": inputs["package_source_sha256"],
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": args.device,
        "tf32": tf32_policy(args.device),
        "threads": args.threads,
        "execution_order": list(execution_order(case)),
        "fsl_version": inputs["tools"].version,
        "fsl_fnirt_sha256": inputs["fnirt_sha256"],
        "fsl_fnirtfileutils_sha256": inputs["fnirtfileutils_sha256"],
        "config_sha256": inputs["config_sha256"],
        "moving_gm_sha256": sha256(case.gm),
        "flirt_matrix_sha256": sha256(case.affine),
        "template_sha256": inputs["template_sha256"],
        "reference_mask_sha256": inputs["reference_mask_sha256"],
        "algorithm": "matched FSL GM and official FLIRT affine; FNIRT GM config",
        "timing_boundary": "FNIRT call including cout/iout/jout writes",
    }


def output_paths(work_dir: Path, case_id: str) -> dict[str, Any]:
    directory = Path(work_dir) / "private" / "cases" / case_id

    def arm(name: str) -> dict[str, Path]:
        root = directory / name
        return {
            "directory": root,
            "cout": root / "coefficients.nii.gz",
            "nonlinear_residual": root / "nonlinear_residual.nii.gz",
            "iout": root / "iout.nii.gz",
            "jout": root / "jout.nii.gz",
            "modulated_gm": root / "modulated_gm.nii.gz",
        }

    return {
        "directory": directory,
        "record": directory / "run.private.json",
        "log": directory / "fsl.private.log",
        "fsl": arm("fsl"),
        "torch": arm("torch"),
    }


def _result_files(paths: dict[str, Any]) -> dict[str, Path]:
    return {
        f"{arm}_{name}": paths[arm][name]
        for arm in ("fsl", "torch")
        for name in ("cout", "nonlinear_residual", "iout", "jout", "modulated_gm")
    }


def completed_record(
    record_path: Path,
    expected_signature: str,
    outputs: dict[str, Path],
) -> dict[str, Any] | None:
    if not record_path.is_file():
        return None
    record = json.loads(record_path.read_text())
    if record.get("status") != "success":
        return None
    if record.get("run_signature") != expected_signature:
        raise RuntimeError(
            f"stale successful cache has a different signature: {record_path}; "
            "use --overwrite after reviewing the changed provenance"
        )
    expected_hashes = record.get("output_sha256", {})
    if set(expected_hashes) != set(outputs):
        raise RuntimeError(f"cached output manifest is incomplete: {record_path}")
    for name, path in outputs.items():
        if not path.is_file():
            return None
        if sha256(path) != expected_hashes[name]:
            raise RuntimeError(
                f"cached output hash mismatch for {path}; use --overwrite"
            )
    return record


def _clear_known_outputs(paths: dict[str, Any]) -> None:
    for path in _result_files(paths).values():
        path.unlink(missing_ok=True)
    paths["log"].unlink(missing_ok=True)


def _fsl_environment(tools: FSLTools, threads: int) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        FSLDIR=str(tools.root),
        FSLOUTPUTTYPE="NIFTI_GZ",
        OMP_NUM_THREADS=str(threads),
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    return environment


def run_command(
    command: list[str],
    *,
    environment: dict[str, str],
    log_path: Path,
) -> float:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log_path.open("a") as log:
        log.write(json.dumps(command) + "\n")
        log.flush()
        subprocess.run(
            command,
            check=True,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    return time.perf_counter() - started


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_torch_fnirt(
    args,
    case: CaseInputs,
    inputs: dict[str, Any],
    outputs: dict[str, Path],
) -> float:
    device = torch.device(args.device)
    synchronize(device)
    started = time.perf_counter()
    run_fnirt(
        case.gm,
        inputs["template"],
        case.affine,
        cout=outputs["cout"],
        iout=outputs["iout"],
        jout=outputs["jout"],
        refmask=inputs["reference_mask"],
        config=inputs["tools"].config,
        device=args.device,
        overwrite=True,
    )
    synchronize(device)
    return time.perf_counter() - started


def run_fsl_fnirt(
    case: CaseInputs,
    inputs: dict[str, Any],
    outputs: dict[str, Path],
    environment: dict[str, str],
    log_path: Path,
) -> float:
    tools = inputs["tools"]
    command = [
        str(tools.fnirt),
        f"--config={tools.config}",
        f"--in={case.gm}",
        f"--ref={inputs['template']}",
        f"--aff={case.affine}",
        f"--refmask={inputs['reference_mask']}",
        f"--cout={outputs['cout']}",
        f"--iout={outputs['iout']}",
        f"--jout={outputs['jout']}",
    ]
    return run_command(command, environment=environment, log_path=log_path)


def expand_coefficients(
    inputs: dict[str, Any],
    coefficients: Path,
    output: Path,
    environment: dict[str, str],
    log_path: Path,
) -> float:
    return run_command(
        [
            str(inputs["tools"].fnirtfileutils),
            f"--in={coefficients}",
            f"--ref={inputs['template']}",
            f"--out={output}",
            "--outformat=field",
        ],
        environment=environment,
        log_path=log_path,
    )


def write_modulated(iout_path: Path, jout_path: Path, output_path: Path) -> None:
    iout_image, iout = _load_3d(iout_path)
    jout_image, jout = _load_3d(jout_path)
    if iout_image.shape[:3] != jout_image.shape[:3] or not np.allclose(
        iout_image.affine, jout_image.affine, atol=1e-5, rtol=0
    ):
        raise ValueError("iout and jout must use the same grid")
    header = iout_image.header.copy()
    header.set_data_dtype(np.float32)
    header.set_slope_inter(1.0, 0.0)
    image = nib.Nifti1Image(
        (iout * jout).astype(np.float32), iout_image.affine, header=header
    )
    qform, qcode = iout_image.get_qform(coded=True)
    sform, scode = iout_image.get_sform(coded=True)
    image.set_qform(qform, int(qcode))
    image.set_sform(sform, int(scode))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(
        f".{output_path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}.nii.gz"
    )
    try:
        nib.save(image, str(temporary))
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)


def load_output(
    path: Path,
    template: nib.spatialimages.SpatialImage,
    *,
    vector: bool,
) -> tuple[nib.spatialimages.SpatialImage, np.ndarray]:
    image = nib.load(str(path))
    data = np.asarray(image.dataobj, dtype=np.float64)
    expected_shape = (*template.shape[:3], 3) if vector else template.shape[:3]
    if data.shape != expected_shape or not np.isfinite(data).all():
        raise ValueError(f"unexpected output shape or non-finite values: {path}")
    if not np.allclose(image.affine, template.affine, atol=1e-5, rtol=0):
        raise ValueError(f"output is not on the template grid: {path}")
    return image, data


def load_coefficients(
    path: Path,
) -> tuple[nib.spatialimages.SpatialImage, np.ndarray]:
    image = nib.load(str(path))
    data = np.asarray(image.dataobj, dtype=np.float64)
    if data.ndim != 4 or data.shape[-1] != 3 or not np.isfinite(data).all():
        raise ValueError(f"unexpected coefficient shape or non-finite values: {path}")
    return image, data


def _same_optional_matrix(first, second, *, atol=1e-5) -> bool:
    if first is None or second is None:
        return first is None and second is None
    return bool(np.allclose(first, second, atol=atol, rtol=0))


def nifti_contract(
    reference: nib.spatialimages.SpatialImage,
    candidate: nib.spatialimages.SpatialImage,
) -> dict[str, bool]:
    reference_qform, reference_qcode = reference.get_qform(coded=True)
    candidate_qform, candidate_qcode = candidate.get_qform(coded=True)
    reference_sform, reference_scode = reference.get_sform(coded=True)
    candidate_sform, candidate_scode = candidate.get_sform(coded=True)
    fields = {
        "shape_equal": tuple(reference.shape) == tuple(candidate.shape),
        "affine_equal": bool(
            np.allclose(reference.affine, candidate.affine, atol=1e-5, rtol=0)
        ),
        "voxel_sizes_equal": bool(
            np.allclose(
                reference.header.get_zooms(),
                candidate.header.get_zooms(),
                atol=1e-6,
                rtol=0,
            )
        ),
        "dtype_equal": (
            reference.header.get_data_dtype() == candidate.header.get_data_dtype()
        ),
        "qform_code_equal": int(reference_qcode) == int(candidate_qcode),
        "qform_matrix_equal": _same_optional_matrix(
            reference_qform, candidate_qform
        ),
        "sform_code_equal": int(reference_scode) == int(candidate_scode),
        "sform_matrix_equal": _same_optional_matrix(
            reference_sform, candidate_sform
        ),
        "intent_code_equal": int(reference.header["intent_code"])
        == int(candidate.header["intent_code"]),
        "intent_parameters_equal": bool(
            np.allclose(
                [
                    reference.header["intent_p1"],
                    reference.header["intent_p2"],
                    reference.header["intent_p3"],
                ],
                [
                    candidate.header["intent_p1"],
                    candidate.header["intent_p2"],
                    candidate.header["intent_p3"],
                ],
                atol=1e-6,
                rtol=0,
            )
        ),
        "intent_name_equal": bytes(reference.header["intent_name"])
        == bytes(candidate.header["intent_name"]),
    }
    fields["all_fields_equal"] = all(fields.values())
    return fields


def comparison_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    first = np.asarray(reference, dtype=np.float64).reshape(-1)
    second = np.asarray(candidate, dtype=np.float64).reshape(-1)
    if first.shape != second.shape or not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("metric arrays must have equal shape and finite values")
    difference = second - first
    first_centered = first - first.mean()
    second_centered = second - second.mean()
    denominator = np.linalg.norm(first_centered) * np.linalg.norm(second_centered)
    pearson = (
        None
        if denominator == 0
        else float(first_centered.dot(second_centered) / denominator)
    )
    return {
        "pearson": pearson,
        "pearson_defined": pearson is not None,
        "mae": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(difference * difference))),
        "maxabs": float(np.max(np.abs(difference))),
        "voxel_values": int(first.size),
    }


def compare_outputs(
    paths: dict[str, Any], template_path: Path
) -> tuple[dict[str, Any], dict[str, bool]]:
    template = nib.load(str(template_path))
    metrics = {}
    contracts = {}
    for name in OUTPUT_KEYS:
        path_key = OUTPUT_PATH_KEYS[name]
        if name == "coefficients":
            reference_image, reference = load_coefficients(paths["fsl"][path_key])
            candidate_image, candidate = load_coefficients(paths["torch"][path_key])
            if reference.shape != candidate.shape:
                raise ValueError("FSL and Torch coefficient arrays have different shapes")
            metrics[name] = comparison_metrics(reference, candidate)
            contracts[name] = nifti_contract(reference_image, candidate_image)
            continue
        vector = name == "nonlinear_residual"
        reference_image, reference = load_output(
            paths["fsl"][path_key], template, vector=vector
        )
        candidate_image, candidate = load_output(
            paths["torch"][path_key], template, vector=vector
        )
        metrics[name] = comparison_metrics(reference, candidate)
        contracts[name] = nifti_contract(reference_image, candidate_image)
    return metrics, contracts


def run_case(args, inputs: dict[str, Any], case: CaseInputs) -> tuple[dict[str, Any], bool]:
    paths = output_paths(args.work_dir, case.case_id)
    outputs = _result_files(paths)
    provenance = case_provenance(args, inputs, case)
    run_signature = signature(provenance)
    if not args.overwrite:
        cached = completed_record(paths["record"], run_signature, outputs)
        if cached is not None:
            return cached, True
    previous = (
        json.loads(paths["record"].read_text())
        if paths["record"].is_file()
        else None
    )
    if previous is not None and not args.overwrite:
        status = previous.get("status")
        if status == "running":
            raise RuntimeError(
                f"case already has a running record: {paths['record']}"
            )
        if status == "success":
            raise RuntimeError(
                f"successful cache is incomplete: {paths['record']}; "
                "use --overwrite after reviewing it"
            )
    existing = [path for path in outputs.values() if path.exists()]
    recoverable = previous is not None and previous.get("status") == "failed"
    if existing and not (args.overwrite or recoverable):
        raise FileExistsError(
            f"untracked or incomplete output exists below {paths['directory']}; "
            "use --overwrite after reviewing it"
        )
    _clear_known_outputs(paths)
    for arm in ("fsl", "torch"):
        paths[arm]["directory"].mkdir(parents=True, exist_ok=True)
    record = {
        "schema": SCHEMA_VERSION,
        "status": "running",
        "case_id": case.case_id,
        "run_signature": run_signature,
        "provenance_private": {
            **provenance,
            "moving_gm": str(case.gm.resolve()),
            "flirt_matrix": str(case.affine.resolve()),
            "template": str(inputs["template"].resolve()),
            "reference_mask": str(inputs["reference_mask"].resolve()),
            "fsl_root": str(inputs["tools"].root.resolve()),
        },
        "outputs_private": {
            name: str(path.resolve()) for name, path in outputs.items()
        },
        "started_at_unix": time.time(),
    }
    atomic_json(paths["record"], record)
    environment = _fsl_environment(inputs["tools"], args.threads)
    try:
        primary_seconds = {}
        for arm in execution_order(case):
            if arm == "fsl":
                primary_seconds[arm] = run_fsl_fnirt(
                    case,
                    inputs,
                    paths["fsl"],
                    environment,
                    paths["log"],
                )
            else:
                primary_seconds[arm] = run_torch_fnirt(
                    args, case, inputs, paths["torch"]
                )
        fsl_expand_seconds = expand_coefficients(
            inputs,
            paths["fsl"]["cout"],
            paths["fsl"]["nonlinear_residual"],
            environment,
            paths["log"],
        )
        write_modulated(
            paths["fsl"]["iout"],
            paths["fsl"]["jout"],
            paths["fsl"]["modulated_gm"],
        )
        torch_expand_seconds = expand_coefficients(
            inputs,
            paths["torch"]["cout"],
            paths["torch"]["nonlinear_residual"],
            environment,
            paths["log"],
        )
        write_modulated(
            paths["torch"]["iout"],
            paths["torch"]["jout"],
            paths["torch"]["modulated_gm"],
        )
        metrics, output_contracts = compare_outputs(paths, inputs["template"])
        output_hashes = {name: sha256(path) for name, path in outputs.items()}
        fsl_seconds = primary_seconds["fsl"]
        torch_seconds = primary_seconds["torch"]
        record.update(
            status="success",
            execution_order=list(execution_order(case)),
            timing_seconds={
                "fsl_cpu_fnirt_wall": float(fsl_seconds),
                "torch_fnirt_synchronized_wall": float(torch_seconds),
                "fsl_cpu_over_torch": float(fsl_seconds / torch_seconds),
                "fsl_residual_expansion_wall": float(fsl_expand_seconds),
                "torch_residual_expansion_wall": float(torch_expand_seconds),
            },
            metrics=metrics,
            output_contracts=output_contracts,
            output_sha256=output_hashes,
            finished_at_unix=time.time(),
        )
    except Exception as error:
        record.update(
            status="failed",
            error_private=f"{type(error).__name__}: {error}",
            finished_at_unix=time.time(),
        )
        atomic_json(paths["record"], record)
        raise
    atomic_json(paths["record"], record)
    return record, False


def cache_status(
    record_path: Path,
    expected_signature: str,
    outputs: dict[str, Path],
) -> str:
    if not record_path.is_file():
        return "missing"
    try:
        return (
            "valid"
            if completed_record(record_path, expected_signature, outputs) is not None
            else "incomplete"
        )
    except RuntimeError:
        return "stale_or_corrupt"


def dry_run(args) -> int:
    inputs = validate_inputs(args, check_device=True)
    cases = []
    for case in inputs["cases"]:
        paths = output_paths(args.work_dir, case.case_id)
        run_signature = signature(case_provenance(args, inputs, case))
        cases.append(
            {
                "case_id": case.case_id,
                "moving_gm_private": str(case.gm.resolve()),
                "flirt_matrix_private": str(case.affine.resolve()),
                "run_signature": run_signature,
                "cache_status": cache_status(
                    paths["record"], run_signature, _result_files(paths)
                ),
            }
        )
    report = {
        "schema": SCHEMA_VERSION,
        "status": "ready",
        "private_report": True,
        "case_count": len(cases),
        "cases": cases,
        "study_root_private": str(inputs["study_root"].resolve()),
        "work_dir_private": str(inputs["work_dir"].resolve()),
        "template_private": str(inputs["template"].resolve()),
        "reference_mask_private": str(inputs["reference_mask"].resolve()),
        "reference_grid": inputs["grid"],
        "fsl_version": inputs["tools"].version,
        "device": args.device,
        "threads": args.threads,
    }
    atomic_json(args.output_json, report)
    print(json.dumps(_jsonable(report), indent=2, allow_nan=False))
    return 0


def run(args) -> int:
    inputs = validate_inputs(args, check_device=True)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    tf32 = apply_tf32_policy(device)
    setup_started = time.perf_counter()
    if device.type == "cuda":
        torch.empty(1, device=device)
        synchronize(device)
    setup_seconds = time.perf_counter() - setup_started
    started = time.perf_counter()
    cache_hits = 0
    for case in inputs["cases"]:
        record, cached = run_case(args, inputs, case)
        cache_hits += int(cached)
        timing = record["timing_seconds"]
        print(
            f"{case.case_id}: FSL {timing['fsl_cpu_fnirt_wall']:.3f}s; "
            f"Torch {timing['torch_fnirt_synchronized_wall']:.3f}s"
            + (" (cache)" if cached else ""),
            flush=True,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
    synchronize(device)
    manifest = {
        "schema": SCHEMA_VERSION,
        "status": "success",
        "private_report": True,
        "case_count": len(inputs["cases"]),
        "case_ids": [case.case_id for case in inputs["cases"]],
        "device": args.device,
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
        ),
        "tf32": tf32,
        "threads": args.threads,
        "device_setup_seconds": setup_seconds,
        "batch_wall_seconds": time.perf_counter() - started,
        "cache_hits": cache_hits,
        "package_source_sha256": inputs["package_source_sha256"],
        "script_sha256": inputs["script_sha256"],
        "timing_contract": (
            "Torch CUDA is synchronized immediately before and after run_fnirt; "
            "both FNIRT timings include cout/iout/jout writes and exclude residual "
            "expansion and modulation"
        ),
        "execution_order_policy": (
            "odd case slots run FSL then Torch; even case slots run Torch then FSL"
        ),
    }
    atomic_json(
        Path(args.work_dir) / "private" / "run.private.json",
        manifest,
    )
    return 0


def distribution(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("distribution requires at least one finite value")
    return {
        "count": int(array.size),
        "minimum": float(array.min()),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "mean": float(array.mean()),
        "q75": float(np.quantile(array, 0.75)),
        "maximum": float(array.max()),
    }


def aggregate_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate = {}
    for output in OUTPUT_KEYS:
        aggregate[output] = {}
        for metric in METRIC_KEYS:
            values = [record[output][metric] for record in records]
            defined = [value for value in values if value is not None]
            summary = (
                distribution(defined)
                if defined
                else {
                    "count": 0,
                    "minimum": None,
                    "q25": None,
                    "median": None,
                    "mean": None,
                    "q75": None,
                    "maximum": None,
                }
            )
            summary["undefined_count"] = len(values) - len(defined)
            aggregate[output][metric] = summary
    return aggregate


def privacy_check(public: dict[str, Any], forbidden: Iterable[Path | str]) -> None:
    text = json.dumps(public, sort_keys=True)
    if CASE_PATTERN.search(text):
        raise ValueError("public report contains a case identifier")
    for value in forbidden:
        token = str(value)
        if token and token in text:
            raise ValueError(f"public report contains a private path: {token}")
    if any(token in text for token in ("/cwStorage/", "/public/home/", "\\cwStorage\\")):
        raise ValueError("public report contains a private server path")
    for forbidden_key in (
        "run_signature",
        "case_id",
        "outputs_private",
        "provenance_private",
    ):
        if f'"{forbidden_key}"' in text:
            raise ValueError(f"public report contains private key {forbidden_key}")


def validated_run_manifest(args, inputs: dict[str, Any]) -> dict[str, Any]:
    path = Path(args.work_dir) / "private" / "run.private.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    manifest = json.loads(path.read_text())
    expected = {
        "status": "success",
        "case_count": len(inputs["cases"]),
        "case_ids": [case.case_id for case in inputs["cases"]],
        "device": args.device,
        "tf32": tf32_policy(args.device),
        "threads": args.threads,
        "package_source_sha256": inputs["package_source_sha256"],
        "script_sha256": inputs["script_sha256"],
    }
    mismatches = {
        key: {"expected": value, "observed": manifest.get(key)}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"run manifest does not match summary inputs: {mismatches}")
    try:
        cache_hits = int(manifest["cache_hits"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("run manifest cache accounting is incomplete") from error
    if not 0 <= cache_hits <= len(inputs["cases"]):
        raise RuntimeError("run manifest cache accounting is inconsistent")
    expected_order_policy = (
        "odd case slots run FSL then Torch; even case slots run Torch then FSL"
    )
    if manifest.get("execution_order_policy") != expected_order_policy:
        raise RuntimeError("run manifest execution-order policy is inconsistent")
    return manifest


def summarize(args) -> int:
    inputs = validate_inputs(args, check_device=False)
    run_manifest = validated_run_manifest(args, inputs)
    private_cases = []
    metric_records = []
    output_contract_records = []
    timing_records = []
    execution_digests = set()
    for case in inputs["cases"]:
        paths = output_paths(args.work_dir, case.case_id)
        provenance = case_provenance(args, inputs, case)
        expected_signature = signature(provenance)
        record = completed_record(
            paths["record"], expected_signature, _result_files(paths)
        )
        if record is None:
            raise FileNotFoundError(
                f"no successful current-signature result for {case.case_id}"
            )
        metrics, output_contracts = compare_outputs(paths, inputs["template"])
        metric_records.append(metrics)
        output_contract_records.append(output_contracts)
        timing_records.append(record["timing_seconds"])
        execution_digests.add(
            record["provenance_private"]["package_source_sha256"]
        )
        case_private = {
            "case_id": case.case_id,
            "run_signature": expected_signature,
            "record_private": str(paths["record"].resolve()),
            "metrics": metrics,
            "output_contracts": output_contracts,
            "timing_seconds": record["timing_seconds"],
            "output_sha256": record["output_sha256"],
        }
        private_cases.append(case_private)
        atomic_json(
            Path(args.work_dir)
            / "private"
            / "summaries"
            / case.case_id
            / "summary.private.json",
            case_private,
        )
    if len(execution_digests) != 1:
        raise RuntimeError("case records contain multiple package source snapshots")
    execution_digest = next(iter(execution_digests))
    timing = {
        key: distribution(record[key] for record in timing_records)
        for key in (
            "fsl_cpu_fnirt_wall",
            "torch_fnirt_synchronized_wall",
            "fsl_cpu_over_torch",
        )
    }
    output_contract_summary = {
        output: {
            key: {
                "true_count": int(
                    sum(record[output][key] for record in output_contract_records)
                ),
                "case_count": len(output_contract_records),
                "all_cases": bool(output_contract_records)
                and all(record[output][key] for record in output_contract_records),
            }
            for key in output_contract_records[0][output]
        }
        for output in OUTPUT_KEYS
    }
    public = {
        "schema": SCHEMA_VERSION,
        "status": "measured",
        "feature": "matched-input TorchFNIRT versus FSL FNIRT validation",
        "cohort": {
            "modality": "real T1w-derived FSL FAST grey-matter probability maps",
            "case_count": len(inputs["cases"]),
            "individual_results_published": False,
            "identifiers_published": False,
        },
        "provenance": {
            "fsl_version": inputs["tools"].version,
            "fsl_fnirt_sha256": inputs["fnirt_sha256"],
            "fsl_fnirtfileutils_sha256": inputs["fnirtfileutils_sha256"],
            "config": SUPPORTED_CONFIG,
            "config_sha256": inputs["config_sha256"],
            "template_sha256": inputs["template_sha256"],
            "official_reference_mask_sha256": inputs[
                "reference_mask_sha256"
            ],
            "candidate_execution_package_source_sha256": execution_digest,
            "summarizer_package_source_sha256": inputs[
                "package_source_sha256"
            ],
            "source_unchanged_at_summarize": (
                execution_digest == inputs["package_source_sha256"]
            ),
        },
        "matched_inputs": {
            "moving": "same FSL FAST GM probability map",
            "reference": "same group GM template",
            "affine": "same official FLIRT input-to-reference scaled-mm matrix",
            "reference_mask": "same official dilated MNI152 2-mm binary mask",
        },
        "outputs": {
            "coefficients": "native cout cubic B-spline control coefficients",
            "nonlinear_residual": (
                "cout expanded by the same FSL fnirtfileutils without --withaff"
            ),
            "iout": "warped input on the reference grid",
            "jout": "nonlinear-only Jacobian determinant",
            "modulated_gm": "float32 iout multiplied by jout with common code",
        },
        "metrics": {
            "scope": (
                "all finite coefficient values on the native control grid; all "
                "finite values for other outputs on the complete reference grid"
            ),
            "names": ["Pearson", "MAE", "RMSE", "maximum absolute error"],
            "results": aggregate_metrics(metric_records),
        },
        "output_contract": output_contract_summary,
        "timing": {
            "device": args.device,
            "threads": args.threads,
            "contract": (
                "FSL CPU wall time waits for process completion; Torch CUDA is "
                "synchronized at both boundaries. Both include cout/iout/jout "
                "writes and exclude fnirtfileutils expansion and modulation."
            ),
            "distributions_seconds_or_ratio": timing,
            "device_name": run_manifest["device_name"],
            "tf32": run_manifest["tf32"],
            "execution_order_policy": run_manifest["execution_order_policy"],
        },
        "interpretation": {
            "equivalence_decision": "not made by this tool",
            "requirement": (
                "Inspect aggregate errors and case-level private records before "
                "making a numerical-equivalence claim."
            ),
        },
        "privacy": (
            "Aggregate-only artifact; paths, identifiers, individual metrics, "
            "matrices, cache signatures, and output filenames are excluded."
        ),
    }
    private = {
        "schema": SCHEMA_VERSION,
        "private_report": True,
        "study_root_private": str(inputs["study_root"].resolve()),
        "template_private": str(inputs["template"].resolve()),
        "reference_mask_private": str(inputs["reference_mask"].resolve()),
        "cases": private_cases,
    }
    privacy_check(
        public,
        (
            inputs["study_root"].resolve(),
            inputs["work_dir"].resolve(),
            inputs["template"].resolve(),
            inputs["reference_mask"].resolve(),
            *(case.case_id for case in inputs["cases"]),
        ),
    )
    atomic_json(args.private_json, private)
    atomic_json(args.public_json, public)
    print(
        json.dumps(
            {
                "status": "summarized",
                "case_count": len(inputs["cases"]),
                "private_json": str(args.private_json),
                "public_json": str(args.public_json),
            },
            allow_nan=False,
        )
    )
    return 0


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--study-root",
        type=Path,
        required=True,
        help="directory containing subjects/case01..case10 and assets/template_GM_v1.nii.gz",
    )
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--template", type=Path)
    parser.add_argument("--reference-mask", type=Path)
    parser.add_argument("--fsl-dir", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--case-count", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=4)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    dry_parser = subparsers.add_parser(
        "dry-run", help="validate matched inputs and write a private cache manifest"
    )
    add_common(dry_parser)
    dry_parser.add_argument("--output-json", type=Path)
    dry_parser.set_defaults(function=dry_run)

    run_parser = subparsers.add_parser(
        "run", help="run or resume private per-case FSL and TorchFNIRT comparisons"
    )
    add_common(run_parser)
    run_parser.add_argument("--overwrite", action="store_true")
    run_parser.set_defaults(function=run)

    summary_parser = subparsers.add_parser(
        "summarize", help="write private details and an aggregate-only public JSON"
    )
    add_common(summary_parser)
    summary_parser.add_argument("--private-json", type=Path)
    summary_parser.add_argument("--public-json", type=Path)
    summary_parser.set_defaults(function=summarize)

    args = parser.parse_args(argv)
    if args.case_count < 1 or args.threads < 1:
        parser.error("--case-count and --threads must be positive")
    args.study_root = Path(args.study_root)
    args.work_dir = Path(args.work_dir)
    if args.command == "dry-run" and args.output_json is None:
        args.output_json = args.work_dir / "private" / "dry_run.private.json"
    if args.command == "summarize":
        if args.private_json is None:
            args.private_json = args.work_dir / "private" / "summary.private.json"
        if args.public_json is None:
            args.public_json = args.work_dir / "summary.public.json"
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    return args.function(args)


if __name__ == "__main__":
    raise SystemExit(main())
