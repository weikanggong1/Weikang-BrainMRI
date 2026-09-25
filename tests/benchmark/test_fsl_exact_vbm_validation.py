import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest


SCRIPT = Path(__file__).parents[2] / "benchmark" / "fsl_exact_vbm_validation.py"
SPEC = importlib.util.spec_from_file_location("fsl_exact_vbm_validation", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.asarray(data, dtype=np.float32), np.eye(4)), path)


def test_summarize_keeps_case_details_private(tmp_path):
    study = tmp_path / "study"
    work = tmp_path / "work"
    template = study / "assets" / "template_GM_v1.nii.gz"
    reference_mask = tmp_path / "MNI152_T1_2mm_brain_mask_dil.nii.gz"
    shape = (4, 5, 6)
    grid = np.indices(shape).sum(0).astype(np.float32)
    gm = 0.1 + grid / (2 * grid.max())
    jacobian = 0.8 + grid / (5 * grid.max())
    mask = np.ones(shape, dtype=np.uint8)
    _save(template, gm)
    _save(reference_mask, mask)

    case = MODULE.case_paths(study, "case01")
    assert case.raw_t1.name == "T1_orig.nii.gz"
    for path, data in (
        (case.raw_t1, gm),
        (case.fsl_gm, gm),
        (case.fsl_warped_gm, gm),
        (case.fsl_jacobian, jacobian),
        (case.fsl_modulated_gm, gm * jacobian),
    ):
        _save(path, data)
    case.fsl_matrix.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(case.fsl_matrix, np.eye(4))
    case.fsl_timing.write_text(
        json.dumps(
            {
                **{name: 1.0 for name in MODULE.PREPROCESS_STAGES},
                "fsl_reg_ukb": 2.0,
                "modulate_ukb": 0.5,
            }
        )
    )

    weights = tmp_path / "weights"
    weights.mkdir()
    for filename in MODULE.EXPECTED_WEIGHTS:
        (weights / filename).write_bytes(f"test {filename}".encode())
    args = SimpleNamespace(
        study_root=study,
        template=None,
        reference_mask=reference_mask,
        weights=weights,
        work_dir=work,
        case_count=1,
        device="cpu",
        threads=1,
        layers=list(MODULE.LAYERS),
        backends=list(MODULE.BACKENDS),
        private_json=work / "private" / "summary.private.json",
        public_json=work / "summary.public.json",
        public_csv=work / "summary.public.csv",
    )
    inputs = MODULE.validate_inputs(args, require_weights=True, check_device=False)
    recorded_tf32 = {
        "cuda_execution": False,
        "matmul": False,
        "cudnn": False,
        "reduced_precision_tensor_dtype": False,
    }
    environment = {
        "platform": "test-platform",
        "torch": "test-torch",
        "cuda_runtime": None,
        "device_name": "CPU",
    }
    source_digest = MODULE.package_source_digest()
    context = MODULE.validation_context(
        args,
        inputs,
        source_digest=source_digest,
        harness_digest=MODULE.script_digest(),
        tf32=recorded_tf32,
        environment=environment,
    )
    case_hashes = {case.case_id: MODULE.case_input_hashes(case)}

    flirt = MODULE.flirt_paths(work, case.case_id)
    flirt["directory"].mkdir(parents=True, exist_ok=True)
    np.savetxt(flirt["matrix"], np.eye(4))
    _save(flirt["moved"], gm)
    flirt_provenance = MODULE.flirt_provenance(
        context, case_hashes[case.case_id]
    )
    MODULE.atomic_json(
        flirt["record"],
        {
            "status": "success",
            "run_signature": MODULE.signature(flirt_provenance),
            "synchronized_compute_sec": 1.0,
            "output_save_sec": 0.1,
            "output_sha256": {
                "matrix": MODULE.sha256(flirt["matrix"]),
                "moved": MODULE.sha256(flirt["moved"]),
            },
            "provenance_private": flirt_provenance,
        },
    )

    mask_hash = MODULE.array_fingerprint(mask)
    for layer in MODULE.LAYERS:
        for backend in MODULE.BACKENDS:
            paths = MODULE.candidate_paths(work, layer, backend, case.case_id)
            paths["directory"].mkdir(parents=True, exist_ok=True)
            _save(paths["warped_gm"], gm)
            _save(paths["jacobian"], jacobian)
            _save(paths["modulated_gm"], gm * jacobian)
            provenance = MODULE.candidate_provenance(
                args,
                case,
                layer,
                backend,
                context,
                case_hashes[case.case_id],
            )
            MODULE.atomic_json(
                paths["record"],
                {
                    "status": "success",
                    "run_signature": MODULE.signature(provenance),
                    "synchronized_compute_sec": 1.0,
                    "output_save_sec": 0.1,
                    "output_sha256": {
                        name: MODULE.sha256(paths[name])
                        for name in MODULE.OUTPUT_FILENAMES
                    },
                    "provenance_private": provenance,
                    "registration": {
                        "reference_mask_source": "explicit",
                        "pre_nonlinear_signature": {
                            "combined_sha256": f"shared-{layer}",
                            "reference_mask_sha256": mask_hash,
                        },
                    },
                },
            )

    manifest_provenance = MODULE.run_manifest_provenance(
        context,
        [case],
        case_hashes,
        args.layers,
        args.backends,
    )
    MODULE.atomic_json(
        work / "private" / "run.private.json",
        {
            "status": "success",
            "run_signature": MODULE.signature(manifest_provenance),
            "provenance_private": manifest_provenance,
            "package_source_sha256": source_digest,
            "script_sha256": MODULE.script_digest(),
            "case_count": 1,
            "case_ids_private": [case.case_id],
            "layers": args.layers,
            "backends": args.backends,
            "device": args.device,
            "threads": args.threads,
            "tf32": recorded_tf32,
            "execution_environment": environment,
            "constructor_setup_sec": [],
            "planned_record_count": 7,
            "cache_hits": {"flirt": 0, "candidate": 0, "total": 0},
            "fresh_record_count": 7,
            "all_records_fresh": True,
            "invocation_wall_sec": 2.0,
            "batch_wall_sec": 2.0,
        },
    )
    assert MODULE.summarize(args) == 0

    public = json.loads(args.public_json.read_text())
    public_text = args.public_json.read_text() + args.public_csv.read_text()
    assert public["cohort"]["case_count"] == 1
    assert public["shared_chain_audit"][
        "pre_nonlinear_signature_equal_count"
    ] == {layer: 1 for layer in MODULE.LAYERS}
    assert public["execution"]["hardware"]["tf32"] == recorded_tf32
    assert public["execution"]["batch_wall_sec"] == 2.0
    assert not public["reference_fsl_timing_provenance"][
        "controlled_speed_claim_allowed"
    ]
    assert (
        "fsl_observed_wall_over_candidate_compute_and_save"
        in public["paired_timing"]["matched_gm"]["fnirt"]
    )
    assert "case01" not in public_text
    assert str(study.resolve()) not in public_text
    assert (
        work / "private" / "summaries" / "case01" / "summary.private.json"
    ).is_file()


def test_affine_rmsdiff_and_image_metrics_identity():
    identity = np.eye(4)
    assert MODULE.affine_rmsdiff_mm(identity, identity) == 0.0
    values = np.arange(27, dtype=np.float32).reshape(3, 3, 3)
    metrics = MODULE.image_metrics(values, values, np.ones_like(values, bool), 4.0)
    assert np.isclose(metrics["pearson"], 1.0)
    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["dice"] == 1.0


def test_completed_record_rejects_changed_output(tmp_path):
    output = tmp_path / "output.bin"
    output.write_bytes(b"first")
    record = tmp_path / "run.private.json"
    MODULE.atomic_json(
        record,
        {
            "status": "success",
            "run_signature": "expected",
            "output_sha256": {"output": MODULE.sha256(output)},
        },
    )
    assert MODULE.completed_record(
        record, "expected", {"output": output}
    ) is not None
    output.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="output hash mismatch"):
        MODULE.completed_record(record, "expected", {"output": output})
