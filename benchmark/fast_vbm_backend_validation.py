#!/usr/bin/env python3
"""Run and summarize the paired FastVBM nonlinear-backend validation.

Private inputs, individual outputs, and per-case measurements stay below
``--work-dir``.  The public JSON and CSV contain aggregate values only.

Three layers separate upstream and registration effects:

* ``end_to_end``: raw T1 -> SynthStrip -> TorchFAST -> PyTorch FLIRT -> nonlinear
* ``matched_gm``: FSL FAST GM -> PyTorch FLIRT -> nonlinear
* ``matched_affine``: FSL FAST GM + the FSL FLIRT matrix -> nonlinear

Every candidate output is compared with the FSL UKB VBM result from the same
scan on the UKB GM-template grid.  CUDA timing uses explicit synchronization at
both boundaries.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import time

import nibabel as nib
import numpy as np
import surfa as sf
import torch

import freesurfer_torch
from freesurfer_torch.fast_vbm import (
    FastVBM,
    PyTorchFNIRTRegistration,
    flirt_to_world_pull,
    register_gm,
)
from freesurfer_torch.fast_vbm.synthmorph_backend import (
    SynthMorphDeformRegistration,
)


LAYERS = ("end_to_end", "matched_gm", "matched_affine")
BACKENDS = ("synthmorph", "fnirt")
OUTPUTS = {
    "warped_gm": "T1_GM_to_template_GM.nii.gz",
    "jacobian": "T1_GM_JAC_nl.nii.gz",
    "modulated_gm": "T1_GM_to_template_GM_mod.nii.gz",
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


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def package_digest():
    root = Path(freesurfer_torch.__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def load_cases(manifest_path, limit):
    manifest = json.loads(Path(manifest_path).read_text())
    cases = manifest.get("cases", [])[:limit]
    if len(cases) != limit:
        raise ValueError(f"manifest has {len(cases)} selected cases; expected {limit}")
    identifiers = [case.get("case_id") for case in cases]
    if (
        len(set(identifiers)) != len(identifiers)
        or any(not re.fullmatch(r"case[0-9]+", value or "") for value in identifiers)
    ):
        raise ValueError("manifest case_id values must be unique caseNN labels")
    for case in cases:
        path = Path(case["path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        stat = path.stat()
        if stat.st_size != case.get("size_bytes") or stat.st_mtime_ns != case.get(
            "mtime_ns"
        ):
            if sha256(path) != case.get("sha256"):
                raise RuntimeError(f"manifest input changed for {case['case_id']}")
    return cases


def fsl_paths(root, case_id):
    t1 = Path(root) / case_id / "T1"
    reference = t1 / "T1_vbm" / "ukb"
    return {
        "gm": t1 / "T1_fast" / "T1_brain_pve_1.nii.gz",
        "matrix": reference / "T1_GM_to_template_GM.mat",
        "warped_gm": reference / OUTPUTS["warped_gm"],
        "jacobian": reference / OUTPUTS["jacobian"],
        "modulated_gm": reference / OUTPUTS["modulated_gm"],
        "timing": Path(root) / case_id / "timings.private.json",
    }


def candidate_paths(work_dir, layer, backend, case_id):
    directory = Path(work_dir) / "private_runs" / layer / backend / case_id
    return {
        "directory": directory,
        "warped_gm": directory / OUTPUTS["warped_gm"],
        "jacobian": directory / OUTPUTS["jacobian"],
        "modulated_gm": directory / OUTPUTS["modulated_gm"],
        "record": directory / "run.private.json",
    }


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed(device, function):
    synchronize(device)
    started = time.perf_counter()
    value = function()
    synchronize(device)
    return value, time.perf_counter() - started


def build_deform(backend, args):
    if backend == "synthmorph":
        return SynthMorphDeformRegistration(
            weights=args.weights,
            device=args.device,
            extent=256,
            hyper=0.5,
            steps=7,
        )
    return PyTorchFNIRTRegistration(device=args.device)


def run_signature(args, case, layer, backend, paths, source_digest, template_hash):
    inputs = {
        "raw_t1_sha256": case["sha256"],
        "template_sha256": template_hash,
        "package_source_sha256": source_digest,
        "layer": layer,
        "backend": backend,
        "device": args.device,
        "threads": args.threads,
    }
    if layer in ("matched_gm", "matched_affine"):
        inputs["fsl_fast_gm_sha256"] = sha256(paths["gm"])
    if layer == "matched_affine":
        inputs["fsl_flirt_matrix_sha256"] = sha256(paths["matrix"])
    return hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest(), inputs


def save_registration(result, paths):
    result.warped_gm.save(paths["warped_gm"])
    result.jacobian.save(paths["jacobian"])
    result.modulated_gm.save(paths["modulated_gm"])


def registration_metadata(result):
    return {
        "fit_score_normalized_correlation": float(result.fit_score),
        "maximum_displacement_mm": float(result.maximum_displacement_mm),
        "pull_world_affine": np.asarray(result.pull_world_affine).tolist(),
        "qc": result.qc,
    }


def run_one(
    args,
    case,
    layer,
    backend,
    template,
    pipeline,
    deform,
    source_digest,
    template_hash,
):
    fsl = fsl_paths(args.fsl_root, case["case_id"])
    for name in ("gm", "matrix", "warped_gm", "jacobian", "modulated_gm"):
        if not fsl[name].is_file():
            raise FileNotFoundError(fsl[name])
    paths = candidate_paths(args.work_dir, layer, backend, case["case_id"])
    signature, provenance = run_signature(
        args, case, layer, backend, fsl, source_digest, template_hash
    )
    if not args.overwrite and paths["record"].is_file():
        previous = json.loads(paths["record"].read_text())
        complete = all(paths[name].is_file() for name in OUTPUTS)
        if (
            previous.get("status") == "success"
            and previous.get("signature") == signature
            and complete
        ):
            print(f"resume {layer} {backend} {case['case_id']}", flush=True)
            return previous
    paths["directory"].mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for name in (*OUTPUTS, "record"):
            paths[name].unlink(missing_ok=True)

    record = {
        "status": "running",
        "case_id": case["case_id"],
        "layer": layer,
        "backend": backend,
        "signature": signature,
        "provenance_private": provenance,
        "input_private": case["path"],
        "outputs_private": {name: str(paths[name]) for name in OUTPUTS},
        "started_at": time.time(),
    }
    atomic_json(paths["record"], record)
    print(f"start {layer} {backend} {case['case_id']}", flush=True)
    try:
        if layer == "end_to_end":
            result, compute_sec = timed(
                torch.device(args.device),
                lambda: pipeline(case["path"], args.template),
            )
            metadata = result.report()
            save_function = lambda: (
                result.warped_gm.save(paths["warped_gm"]),
                result.jacobian.save(paths["jacobian"]),
                result.modulated_gm.save(paths["modulated_gm"]),
            )
        else:
            moving = sf.load_volume(str(fsl["gm"]))
            if layer == "matched_affine":
                matrix = np.loadtxt(fsl["matrix"], dtype=np.float64)
                initial_pull = flirt_to_world_pull(
                    matrix,
                    moving.geom.vox2world.matrix,
                    template.geom.vox2world.matrix,
                    moving.shape[:3],
                    template.shape[:3],
                    moving.geom.voxsize,
                    template.geom.voxsize,
                )
            else:
                initial_pull = None
            result, compute_sec = timed(
                torch.device(args.device),
                lambda: register_gm(
                    moving,
                    template,
                    device=args.device,
                    initial_pull=initial_pull,
                    initial_pull_convention=(
                        "fixed-to-moving-world-ras"
                        if initial_pull is not None
                        else None
                    ),
                    synthmorph_weights=args.weights,
                    registration_backend=backend,
                    deform_model=deform,
                ),
            )
            metadata = registration_metadata(result)
            save_function = lambda: save_registration(result, paths)

        save_started = time.perf_counter()
        save_function()
        save_sec = time.perf_counter() - save_started
        record.update(
            status="success",
            synchronized_compute_sec=compute_sec,
            output_save_sec=save_sec,
            total_compute_and_save_sec=compute_sec + save_sec,
            metadata=metadata,
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
    print(
        f"finish {layer} {backend} {case['case_id']}: {compute_sec:.3f}s compute, "
        f"{save_sec:.3f}s save",
        flush=True,
    )
    return record


def run(args):
    cases = load_cases(args.manifest, args.limit)
    template_path = Path(args.template)
    if not template_path.is_file():
        raise FileNotFoundError(template_path)
    template = sf.load_volume(str(template_path))
    template_data = np.asarray(template.data)
    if template_data.ndim != 3 or not np.isfinite(template_data).all():
        raise ValueError("template must contain one finite 3D image")
    if not np.any(template_data > 0.01):
        raise ValueError("template mask at intensity > 0.01 is empty")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.set_num_threads(args.threads)
    source_digest = package_digest()
    template_hash = sha256(template_path)
    started = time.perf_counter()
    setup = []

    for backend in args.backends:
        pipeline = None
        deform = None
        if "end_to_end" in args.layers:
            pipeline, setup_sec = timed(
                device,
                lambda: FastVBM(
                    device=args.device,
                    threads=args.threads,
                    synthstrip_weights=args.weights,
                    synthmorph_weights=args.weights,
                    bias_correction=True,
                    registration_backend=backend,
                ),
            )
            setup.append(
                {"backend": backend, "component": "end_to_end", "sec": setup_sec}
            )
        if any(layer != "end_to_end" for layer in args.layers):
            deform, setup_sec = timed(device, lambda: build_deform(backend, args))
            setup.append(
                {"backend": backend, "component": "matched_registration", "sec": setup_sec}
            )
        for case in cases:
            for layer in args.layers:
                run_one(
                    args,
                    case,
                    layer,
                    backend,
                    template,
                    pipeline,
                    deform,
                    source_digest,
                    template_hash,
                )
        del pipeline, deform
        if device.type == "cuda":
            torch.cuda.empty_cache()
    synchronize(device)
    run_manifest = {
        "status": "success",
        "package_version": freesurfer_torch.__version__,
        "package_source_sha256": source_digest,
        "template_sha256": template_hash,
        "case_count": len(cases),
        "layers": list(args.layers),
        "backends": list(args.backends),
        "device": args.device,
        "threads": args.threads,
        "constructor_setup": setup,
        "wall_sec": time.perf_counter() - started,
        "timing_definition": (
            "synchronized_compute_sec brackets each API call with CUDA synchronize; "
            "eager constructors and output NIfTI writes are reported separately; "
            "the first end-to-end call includes lazy checkpoint loading"
        ),
    }
    run_name = (
        "run."
        + "-".join(args.backends)
        + "."
        + "-".join(args.layers)
        + ".private.json"
    )
    atomic_json(Path(args.work_dir) / run_name, run_manifest)
    atomic_json(Path(args.work_dir) / "run.private.json", run_manifest)
    return 0


def load_image(path, template_image=None):
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


def pearson(reference, candidate):
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    reference = reference - reference.mean()
    candidate = candidate - candidate.mean()
    denominator = np.linalg.norm(reference) * np.linalg.norm(candidate)
    if denominator == 0:
        return None
    return float(reference.dot(candidate) / denominator)


def image_metrics(reference, candidate, mask, threshold=0.2):
    valid = mask & np.isfinite(reference) & np.isfinite(candidate)
    if valid.sum() < 3:
        raise ValueError("fewer than three finite evaluation voxels")
    first = reference[valid].astype(np.float64)
    second = candidate[valid].astype(np.float64)
    difference = second - first
    first_mask = first >= threshold
    second_mask = second >= threshold
    denominator = int(first_mask.sum() + second_mask.sum())
    return {
        "pearson": pearson(first, second),
        "mae": float(np.abs(difference).mean()),
        "rmse": float(np.sqrt(np.square(difference).mean())),
        "mean_error": float(difference.mean()),
        "dice_at_0.2": (
            float(2 * np.logical_and(first_mask, second_mask).sum() / denominator)
            if denominator
            else 1.0
        ),
    }


def jacobian_metrics(reference, candidate, mask):
    agreement = image_metrics(reference, candidate, mask, threshold=0.2)
    agreement.pop("dice_at_0.2")
    values = candidate[mask].astype(np.float64)
    finite = values[np.isfinite(values)]
    positive = finite[finite > 0]
    if not finite.size:
        raise ValueError("candidate Jacobian has no finite template-mask voxels")
    return {
        "agreement_with_fsl": agreement,
        "candidate_distribution": {
            "finite_fraction": float(finite.size / values.size),
            "minimum": float(finite.min()),
            "p01": float(np.percentile(finite, 1)),
            "median": float(np.median(finite)),
            "mean": float(finite.mean()),
            "p99": float(np.percentile(finite, 99)),
            "maximum": float(finite.max()),
            "nonpositive_fraction": float(np.mean(~np.isfinite(values) | (values <= 0))),
            "below_0.2_fraction": float(np.mean(values < 0.2)),
            "above_5_fraction": float(np.mean(values > 5)),
            "log_sd_positive": (
                float(np.std(np.log(positive))) if positive.size else None
            ),
        },
    }


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    if not values.size or not np.isfinite(values).all():
        raise ValueError("aggregate values must be finite and nonempty")
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "q1": float(np.percentile(values, 25)),
        "q3": float(np.percentile(values, 75)),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
    }


def aggregate_nested(records):
    result = {}
    keys = records[0].keys()
    for key in keys:
        values = [record[key] for record in records]
        if isinstance(values[0], dict):
            result[key] = aggregate_nested(values)
        elif values[0] is not None:
            result[key] = distribution(values)
    return result


def subtract_nested(left, right):
    result = {}
    for key in left:
        if key not in right:
            raise ValueError(f"paired backend record is missing {key}")
        if isinstance(left[key], dict):
            result[key] = subtract_nested(left[key], right[key])
        elif left[key] is not None and right[key] is not None:
            result[key] = float(left[key]) - float(right[key])
    return result


def fsl_stage_times(paths):
    timing = json.loads(paths["timing"].read_text())
    preprocessing = sum(
        float(timing.get(stage, 0))
        for stage in PREPROCESS_STAGES
        if isinstance(timing.get(stage, 0), (int, float))
    )
    registration = float(timing["fsl_reg_ukb"]) + float(timing["modulate_ukb"])
    return preprocessing, registration, preprocessing + registration


def flatten(prefix, value, rows, layer, backend, category):
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict) and set(item) == {
            "n",
            "mean",
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
                    "metric": name,
                    **item,
                }
            )
        elif isinstance(item, dict):
            flatten(name, item, rows, layer, backend, category)


def privacy_check(public_json, public_csv):
    text = Path(public_json).read_text() + Path(public_csv).read_text()
    forbidden = ("/cwStorage/", "\\cwStorage\\", "input_private", "outputs_private")
    if any(value in text for value in forbidden) or re.search(r"\bcase[0-9]+\b", text):
        raise RuntimeError("public report contains a private path or case label")


def public_invocations(paths):
    result = []
    for path in paths:
        record = json.loads(Path(path).read_text())
        if record.get("status") != "success":
            raise RuntimeError(f"incomplete invocation manifest: {path}")
        setup = record.get("constructor_setup", record.get("model_setup", []))
        result.append(
            {
                "case_count": int(record["case_count"]),
                "layers": list(record["layers"]),
                "backends": list(record["backends"]),
                "device": record["device"],
                "threads": int(record["threads"]),
                "wall_sec": float(record["wall_sec"]),
                "constructor_setup": [
                    {
                        "backend": item["backend"],
                        "component": item["component"],
                        "sec": float(item["sec"]),
                    }
                    for item in setup
                ],
                "timing_definition": record["timing_definition"],
            }
        )
    return result


def paired_timing(reference, candidates):
    if len(reference) != len(candidates):
        raise ValueError("paired timing vectors have different lengths")
    compute = [item["synchronized_compute_sec"] for item in candidates]
    compute_and_save = [item["compute_and_save_sec"] for item in candidates]
    return {
        "reference_over_candidate_compute": distribution(
            [left / right for left, right in zip(reference, compute)]
        ),
        "reference_minus_candidate_compute_sec": distribution(
            [left - right for left, right in zip(reference, compute)]
        ),
        "reference_over_candidate_compute_and_save": distribution(
            [left / right for left, right in zip(reference, compute_and_save)]
        ),
        "reference_minus_candidate_compute_and_save_sec": distribution(
            [left - right for left, right in zip(reference, compute_and_save)]
        ),
    }


def summarize(args):
    cases = load_cases(args.manifest, args.limit)
    template_image, template = load_image(args.template)
    mask = np.isfinite(template) & (template > 0.01)
    if not np.any(mask):
        raise ValueError("template mask at intensity > 0.01 is empty")

    private = []
    result = {}
    timing_result = {}
    output_counts = {}
    reference_jacobians = []
    fsl_preprocessing = []
    fsl_registration = []
    fsl_full = []
    for case in cases:
        fsl = fsl_paths(args.fsl_root, case["case_id"])
        _, reference_warped = load_image(fsl["warped_gm"], template_image)
        _, reference_jacobian = load_image(fsl["jacobian"], template_image)
        _, reference_modulated = load_image(fsl["modulated_gm"], template_image)
        reference_jacobians.append(
            jacobian_metrics(reference_jacobian, reference_jacobian, mask)[
                "candidate_distribution"
            ]
        )
        preprocessing, registration, full = fsl_stage_times(fsl)
        fsl_preprocessing.append(preprocessing)
        fsl_registration.append(registration)
        fsl_full.append(full)

        case_private = {"case_id": case["case_id"], "comparisons": {}}
        for layer in args.layers:
            for backend in args.backends:
                paths = candidate_paths(args.work_dir, layer, backend, case["case_id"])
                provenance = "executed_in_this_benchmark"
                record = None
                if paths["record"].is_file():
                    record = json.loads(paths["record"].read_text())
                    if record.get("status") != "success":
                        raise RuntimeError(
                            f"incomplete candidate: {layer}/{backend}/{case['case_id']}"
                        )
                elif (
                    layer == "end_to_end"
                    and backend == "synthmorph"
                    and args.reuse_end_to_end_synthmorph_root is not None
                ):
                    directory = (
                        args.reuse_end_to_end_synthmorph_root / case["case_id"]
                    )
                    paths = {
                        **paths,
                        **{name: directory / filename for name, filename in OUTPUTS.items()},
                    }
                    provenance = "reused_previously_validated_output"
                else:
                    raise FileNotFoundError(
                        f"missing candidate: {layer}/{backend}/{case['case_id']}"
                    )
                _, candidate_warped = load_image(paths["warped_gm"], template_image)
                _, candidate_jacobian = load_image(paths["jacobian"], template_image)
                _, candidate_modulated = load_image(
                    paths["modulated_gm"], template_image
                )
                comparison = {
                    "warped_gm": image_metrics(
                        reference_warped, candidate_warped, mask
                    ),
                    "modulated_gm": image_metrics(
                        reference_modulated, candidate_modulated, mask
                    ),
                    "jacobian": jacobian_metrics(
                        reference_jacobian, candidate_jacobian, mask
                    ),
                }
                timing = None
                if record is not None:
                    timing = {
                        "synchronized_compute_sec": float(
                            record["synchronized_compute_sec"]
                        ),
                        "output_save_sec": float(record["output_save_sec"]),
                        "compute_and_save_sec": float(
                            record["total_compute_and_save_sec"]
                        ),
                    }
                    timing_result.setdefault(layer, {}).setdefault(backend, []).append(
                        timing
                    )
                key = f"{layer}/{backend}"
                case_private["comparisons"][key] = {
                    "output_provenance": provenance,
                    "accuracy": comparison,
                    "timing": timing,
                }
                result.setdefault(layer, {}).setdefault(backend, []).append(comparison)
                counts = output_counts.setdefault(layer, {}).setdefault(
                    backend,
                    {
                        "accuracy_outputs": 0,
                        "executed_outputs": 0,
                        "reused_outputs": 0,
                        "executed_timings": 0,
                    },
                )
                counts["accuracy_outputs"] += 1
                counts["executed_outputs" if record is not None else "reused_outputs"] += 1
                counts["executed_timings"] += int(timing is not None)
        private.append(case_private)

    public_results = {}
    paired_backend_differences = {}
    rows = []
    for layer in args.layers:
        public_results[layer] = {}
        for backend in args.backends:
            records = result[layer][backend]
            accuracy = aggregate_nested(records)
            timings = timing_result.get(layer, {}).get(backend, [])
            aggregated_timing = aggregate_nested(timings) if timings else {}
            public_results[layer][backend] = {
                "counts": output_counts[layer][backend],
                "accuracy": accuracy,
                "timing": aggregated_timing,
            }
            flatten("", accuracy, rows, layer, backend, "candidate_accuracy")
            flatten("", aggregated_timing, rows, layer, backend, "candidate_timing")
        if set(args.backends) == set(BACKENDS):
            paired = [
                subtract_nested(fnirt, synthmorph)
                for fnirt, synthmorph in zip(
                    result[layer]["fnirt"], result[layer]["synthmorph"]
                )
            ]
            paired_backend_differences[layer] = {
                "direction": "fnirt_minus_synthmorph",
                "accuracy": aggregate_nested(paired),
            }
            flatten(
                "",
                paired_backend_differences[layer]["accuracy"],
                rows,
                layer,
                "fnirt_minus_synthmorph",
                "paired_backend_difference",
            )

    fsl_timing = {
        "preprocessing_through_fast_sec": distribution(fsl_preprocessing),
        "gm_registration_and_modulation_sec": distribution(fsl_registration),
        "raw_t1_through_modulated_gm_sec": distribution(fsl_full),
    }
    fsl_jacobian = aggregate_nested(reference_jacobians)
    flatten("", fsl_timing, rows, "reference", "fsl_ukb", "reference_timing")
    flatten("jacobian", fsl_jacobian, rows, "reference", "fsl_ukb", "reference_qc")

    invocation_paths = args.run_manifests
    if not invocation_paths:
        invocation_paths = sorted(Path(args.work_dir).glob("run.*.private.json"))
        if not invocation_paths and (Path(args.work_dir) / "run.private.json").is_file():
            invocation_paths = [Path(args.work_dir) / "run.private.json"]
    invocations = public_invocations(invocation_paths)
    for invocation in invocations:
        layer = "+".join(invocation["layers"])
        backend = "+".join(invocation["backends"])
        wall = distribution([invocation["wall_sec"]])
        rows.append(
            {
                "category": "candidate_invocation",
                "layer": layer,
                "backend": backend,
                "metric": "batch_wall_sec",
                **wall,
            }
        )
        for setup in invocation["constructor_setup"]:
            rows.append(
                {
                    "category": "candidate_invocation",
                    "layer": layer,
                    "backend": setup["backend"],
                    "metric": f"constructor_setup.{setup['component']}.sec",
                    **distribution([setup["sec"]]),
                }
            )

    paired_time = {}
    for layer, reference in (
        ("end_to_end", fsl_full),
        ("matched_gm", fsl_registration),
    ):
        for backend in args.backends:
            candidates = timing_result.get(layer, {}).get(backend, [])
            if candidates:
                key = f"{layer}/{backend}"
                paired_time[key] = paired_timing(reference, candidates)
                flatten(
                    "",
                    paired_time[key],
                    rows,
                    layer,
                    backend,
                    "paired_timing",
                )

    hardware = {
        "platform": platform.system(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": args.device,
        "device_name": (
            torch.cuda.get_device_name(torch.device(args.device))
            if torch.device(args.device).type == "cuda" and torch.cuda.is_available()
            else "CPU"
        ),
        "threads": args.threads,
    }
    public = {
        "schema": 1,
        "package_version": freesurfer_torch.__version__,
        "feature": "FastVBM SynthMorph/FNIRT backend comparison with UKB/FSL reference",
        "cohort": {
            "modality": "real T1w",
            "count": len(cases),
            "identifiers_published": False,
            "individual_metrics_published": False,
        },
        "evaluation": {
            "reference": "FSL 6.0.7.4 UKB-v1.5-method VBM outputs",
            "template_mask": "fixed UKB template intensity > 0.01",
            "template_mask_voxels": int(mask.sum()),
            "image_metrics": ["Pearson", "MAE", "RMSE", "mean error", "Dice@0.2"],
            "coordinate_rule": (
                "compare scalar outputs only after verifying the same template shape "
                "and voxel-to-world affine"
            ),
            "layers": {
                "end_to_end": (
                    "raw T1; package SynthStrip, TorchFAST, PyTorch FLIRT, selected nonlinear backend"
                ),
                "matched_gm": (
                    "same FSL FAST GM input; package PyTorch FLIRT and selected nonlinear backend"
                ),
                "matched_affine": (
                    "same FSL FAST GM and converted FSL FLIRT affine; selected nonlinear backend"
                ),
            },
        },
        "method_boundaries": {
            "flirt": (
                "matrix role and FSL scaled-mm conversion are compatible; optimizer and interpolation are independent"
            ),
            "fnirt": (
                "FNIRT-style cubic B-spline SSD optimizer and nonlinear-only Jacobian; not FSL numerical equivalence"
            ),
            "ukb": (
                "method-level reproduction on non-UKB T1w scans; gradient distortion correction unavailable"
            ),
        },
        "execution": {
            "hardware": hardware,
            "timing_scope": (
                "candidate API calls use explicit CUDA synchronization before and after; "
                "eager setup and NIfTI writes are separate, while the first end-to-end "
                "call includes lazy checkpoint loading; reused outputs have no timing; "
                "shared-node load was not controlled"
            ),
            "run_order": "backend-major, then manifest order, then layer order",
            "invocations": invocations,
            "paired_timing": paired_time,
            "paired_timing_boundary": (
                "end_to_end uses FSL raw-T1-through-modulated-GM time; matched_gm "
                "uses FSL GM-registration-and-modulation time; matched_affine has "
                "no FSL nonlinear-only timing and is excluded"
            ),
        },
        "reference_fsl": {
            "timing": fsl_timing,
            "jacobian": fsl_jacobian,
        },
        "results": public_results,
        "paired_backend_differences": paired_backend_differences,
        "privacy": (
            "Aggregate-only public artifacts; source paths, subject labels, PIDs, and individual values are excluded."
        ),
    }
    atomic_json(args.private_json, private)
    atomic_json(args.public_json, public)
    Path(args.public_csv).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.public_csv).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    privacy_check(args.public_json, args.public_csv)
    print(
        json.dumps(
            {
                "status": "success",
                "cases": len(cases),
                "public_json": str(args.public_json),
                "public_csv": str(args.public_csv),
            }
        ),
        flush=True,
    )
    return 0


def add_common(parser):
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--fsl-root", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--layers", nargs="+", choices=LAYERS, default=list(LAYERS))
    parser.add_argument("--backends", nargs="+", choices=BACKENDS, default=list(BACKENDS))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="write private candidate runs")
    add_common(run_parser)
    run_parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="directory containing official SynthStrip and SynthMorph weights",
    )
    run_parser.add_argument("--overwrite", action="store_true")
    run_parser.set_defaults(function=run)

    summary_parser = subparsers.add_parser(
        "summarize", help="compare completed private runs and write aggregate reports"
    )
    add_common(summary_parser)
    summary_parser.add_argument("--private-json", type=Path, required=True)
    summary_parser.add_argument("--public-json", type=Path, required=True)
    summary_parser.add_argument("--public-csv", type=Path, required=True)
    summary_parser.add_argument(
        "--run-manifests",
        type=Path,
        nargs="*",
        default=(),
        help="private invocation manifests used only to publish sanitized wall/setup times",
    )
    summary_parser.add_argument(
        "--reuse-end-to-end-synthmorph-root",
        type=Path,
        help=(
            "read existing caseNN SynthMorph end-to-end outputs when no run record "
            "exists; reused outputs contribute accuracy but no timing"
        ),
    )
    summary_parser.set_defaults(function=summarize)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.limit < 1 or args.threads < 1:
        raise ValueError("limit and threads must be positive")
    return args.function(args)


if __name__ == "__main__":
    raise SystemExit(main())
