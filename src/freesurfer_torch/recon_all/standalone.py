"""Run the standard single-T1 reconstruction from a package-owned runtime."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


REQUIRED_OUTPUTS = (
    "mri/orig.mgz", "mri/aseg.mgz", "mri/aparc+aseg.mgz",
    "mri/ribbon.mgz", "mri/wmparc.mgz", "stats/aseg.stats", "stats/synthseg.vol.csv",
    "stats/lh.aparc.stats", "stats/rh.aparc.stats",
    "stats/lh.aparc.DKTatlas.stats", "stats/rh.aparc.DKTatlas.stats",
    "stats/lh.aparc.a2009s.stats", "stats/rh.aparc.a2009s.stats",
    "surf/lh.white", "surf/rh.white", "surf/lh.pial", "surf/rh.pial",
    "surf/lh.thickness", "surf/rh.thickness", "surf/lh.area", "surf/rh.area",
    "surf/lh.volume", "surf/rh.volume", "label/lh.aparc.annot",
    "label/rh.aparc.annot", "surf/lh.sphere.reg", "surf/rh.sphere.reg",
)
NEURAL_TOOLS = ("mri_synthstrip", "mri_synthseg", "mri_synthmorph")
RUNTIME_PROFILE = {
    "id": "fs820-single-t1-v8-all-v1", "input_count": 1,
    "fresh_subject": True, "empty_subjects_dir": True,
    "extra_flags_allowed": False,
    "argv_suffix": ["-all", "-parallel", "-openmp", "4", "-itkthreads", "1"],
    "threads": 4, "itk_threads": 1, "target_system": "Linux",
}
MODEL_FILES = (
    "synthstrip.1.pt", "synthmorph.affine.2.h5", "synthmorph.deform.3.h5",
    "synthseg_2.0.h5", "synthseg_segmentation_labels_2.0.npy",
    "synthseg_segmentation_names_2.0.npy", "synthseg_topological_classes_2.0.npy",
    "entowm.fsm31.t1.nstd00-30.nstd21-108.h5", "entowm.ctab",
    "mca-dura.both-lh.nstd21.fhs.h5", "mca-dura.ctab",
    "vsinus.no-sp.m.all.nstd10-070.h5", "sclimbic.volstats.csv",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _package_tree_sha256() -> str:
    root = Path(__file__).resolve().parents[1]
    files = {str(path.relative_to(root)): _sha256(path)
             for path in sorted(root.rglob("*.py"))}
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


def _external_models(bundle: Path, manifest: dict, models_dir: str | Path | None) -> Path | None:
    if "external_models" not in manifest:
        if models_dir is not None:
            raise ValueError("--models-dir requires a bundle with external_models")
        return None
    rows = manifest["external_models"]
    if not isinstance(rows, list) or {row.get("path") for row in rows if isinstance(row, dict)} != {
            f"models/{name}" for name in MODEL_FILES} or len(rows) != len(MODEL_FILES):
        raise ValueError("Bundle has an invalid external model inventory")
    if models_dir is None:
        from ..weights import configured_dir
        models_dir = os.environ.get("FREESURFER_TORCH_WEIGHTS") or configured_dir()
    if models_dir is None:
        raise ValueError("Configure weights or pass --models-dir for this bundle")
    root = Path(models_dir).expanduser().resolve(strict=True)
    if not root.is_dir() or root == bundle or root.is_relative_to(bundle):
        raise ValueError("External models must be a directory outside the bundle")
    source_home = manifest.get("fs_home")
    if source_home and root.is_relative_to(Path(source_home).resolve()):
        raise ValueError("External models cannot be inside the source FreeSurfer installation")
    for row in rows:
        name = row["path"].removeprefix("models/")
        path = root / name
        if not path.is_file() or path.is_symlink() or path.stat().st_size != row.get("bytes") \
                or _sha256(path) != row.get("sha256"):
            raise ValueError(f"External model is missing or changed: {path}")
    return root


def _check_bundle(bundle: Path, *, development: bool,
                  models_dir: str | Path | None = None) -> dict:
    required = ["bin/recon-all", *(f"bin/{name}" for name in NEURAL_TOOLS),
                "build-stamp.txt", "average/RB_all_withskull_2020_01_02.gca"]
    manifest_file = bundle / "manifest.json"
    if manifest_file.exists() or manifest_file.is_symlink():
        if not manifest_file.resolve().is_relative_to(bundle) or not manifest_file.is_file():
            raise FileNotFoundError(manifest_file)
        manifest = json.loads(manifest_file.read_text())
    else:
        manifest = {}
    external_root = _external_models(bundle, manifest, models_dir)
    if external_root is None:
        required += [f"models/{name}" for name in MODEL_FILES[:7]]
    missing = [name for name in required if not (bundle / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Bundle is missing {missing}")
    if development:
        return {**manifest, "standalone_verified": False, "development_bundle": True,
                "_resolved_models_dir": str(external_root) if external_root else None}
    if not manifest_file.resolve().is_relative_to(bundle) or not manifest_file.is_file():
        raise FileNotFoundError(manifest_file)
    if manifest.get("standalone_verified") is not True:
        raise ValueError("Bundle has not passed clean-environment validation")
    expected_code = (manifest.get("verification", {}).get("software", {})
                     .get("package_tree_sha256"))
    if not isinstance(expected_code, str) or expected_code != _package_tree_sha256():
        raise ValueError("Verified bundle was validated with different package code")
    if json.dumps(manifest.get("runtime_profile"), sort_keys=True) != json.dumps(
            RUNTIME_PROFILE, sort_keys=True):
        raise ValueError("Bundle has a different or missing single-T1 runtime profile")
    audit = manifest.get("inactive_command_audit", {})
    config = audit.get("resolved_config", {})
    if audit.get("profile_id") != RUNTIME_PROFILE["id"] or \
            config.get("path") != "etc/scoped-reference-recon-config.yaml" or \
            not isinstance(config.get("sha256"), str) or len(config["sha256"]) != 64:
        raise ValueError("Bundle has a missing or invalid effective configuration audit")
    rows = manifest.get("files")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Bundle manifest has no file inventory")
    tracked = set()
    for row in rows:
        if row["path"] in tracked:
            raise ValueError(f"Duplicate manifest path: {row['path']}")
        tracked.add(row["path"])
        path = bundle / row["path"]
        if not path.resolve().is_relative_to(bundle) or not path.is_file() or _sha256(path) != row["sha256"]:
            raise ValueError(f"Bundle file is missing or changed: {path}")
    if not set(required).issubset(tracked):
        raise ValueError("Bundle manifest does not cover the required files")
    if external_root is not None and any(row["path"] in tracked for row in manifest["external_models"]):
        raise ValueError("External models must not also be inside the bundle inventory")
    if config["path"] not in tracked or _sha256(bundle / config["path"]) != config["sha256"]:
        raise ValueError("Bundle's audited effective configuration is missing or changed")
    library_links = {
        bundle / "lib" / name: bundle / "lib" / Path(row["resolved_source"]).name
        for row in manifest.get("libraries", []) if not row["system_runtime"]
        for name in row["names"]
    }
    for parent, dirs, files in os.walk(bundle, followlinks=False):
        for name in dirs + files:
            path = Path(parent) / name
            if path.is_symlink():
                target = library_links.get(path)
                if target is None or path.resolve() != target or not target.is_file() \
                        or str(target.relative_to(bundle)) not in tracked:
                    raise ValueError(f"Bundle has an untracked symlink: {path}")
            relative = str(path.relative_to(bundle))
            if path.is_file() and not path.is_symlink() and relative != "manifest.json" \
                    and not relative.startswith("metadata/") and relative not in tracked:
                raise ValueError(f"Untracked bundle file: {path}")
    manifest["_resolved_models_dir"] = str(external_root) if external_root else None
    return manifest


def run_recon_all(
    t1: str | Path, subject: str, subjects_dir: str | Path,
    bundle_root: str | Path, *, device: str = "cuda:0", threads: int = 4,
    license_file: str | Path | None = None, development_bundle: bool = False,
    models_dir: str | Path | None = None,
) -> dict:
    """Run and record one subject. An independently verified bundle is required by default."""
    image = Path(t1).expanduser().resolve(strict=True)
    bundle = Path(bundle_root).expanduser().resolve(strict=True)
    output_root = Path(subjects_dir).expanduser().resolve()
    if not image.is_file():
        raise ValueError("T1 input must be a file")
    if not subject or subject in (".", "..") or Path(subject).name != subject:
        raise ValueError("subject must be one safe directory name")
    if threads < 1:
        raise ValueError("threads must be positive")
    if not device.startswith("cuda"):
        raise ValueError("the GPU recon-all entry requires a CUDA device")
    manifest = _check_bundle(bundle, development=development_bundle, models_dir=models_dir)
    if not development_bundle and threads != RUNTIME_PROFILE["threads"]:
        raise ValueError("The verified single-T1 profile requires exactly 4 threads")
    license_path = Path(license_file or os.environ.get("FS_LICENSE", "")).expanduser()
    if not license_path.is_file():
        raise FileNotFoundError("Provide a valid --license file for the native programs")
    subject_dir = output_root / subject
    if subject_dir.exists():
        raise FileExistsError(subject_dir)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("Use a fresh, empty subjects directory for standalone reconstruction")
    output_root.mkdir(parents=True, exist_ok=True)
    if not (bundle / "bin/recon-all").is_file():
        raise FileNotFoundError(bundle / "bin/recon-all")

    allowed = ("HOME", "TMPDIR", "LANG", "LC_ALL", "CUDA_VISIBLE_DEVICES")
    env = {key: os.environ[key] for key in allowed if key in os.environ}
    if development_bundle and "PYTHONPATH" in os.environ:
        env["PYTHONPATH"] = os.environ["PYTHONPATH"]
    env.update(
        FREESURFER_HOME=str(bundle), FREESURFER=str(bundle),
        SUBJECTS_DIR=str(output_root),
        FS_LICENSE=str(license_path.resolve()), FS_TORCH_PYTHON=sys.executable,
        FS_TORCH_DEVICE=device, OMP_NUM_THREADS=str(threads),
        PATH=f"{bundle / 'bin'}:/usr/bin:/bin",
        LD_LIBRARY_PATH=str(bundle / "lib"),
    )
    if manifest["_resolved_models_dir"]:
        env["FS_TORCH_MODEL_DIR"] = manifest["_resolved_models_dir"]
    command = [str(bundle / "bin/recon-all"), "-i", str(image), "-s", subject,
               "-sd", str(output_root), "-all", "-parallel", "-openmp", str(threads),
               "-itkthreads", "1"]
    log_path = output_root / f"{subject}.recon-all.log"
    began = time.monotonic()
    started = datetime.now(timezone.utc).isoformat()
    with log_path.open("w") as stream:
        completed = subprocess.run(command, env=env, stdout=stream,
                                   stderr=subprocess.STDOUT, check=False)
    missing_outputs = [name for name in REQUIRED_OUTPUTS if not (subject_dir / name).is_file()]
    effective_config = subject_dir / "scripts/recon-config.yaml"
    actual_config_sha = _sha256(effective_config) if effective_config.is_file() else None
    expected_config_sha = (manifest.get("inactive_command_audit", {})
                           .get("resolved_config", {}).get("sha256"))
    config_match = expected_config_sha is None or actual_config_sha == expected_config_sha
    report = {
        "subject": subject, "input_sha256": _sha256(image),
        "bundle": str(bundle), "bundle_manifest_standalone_verified":
            bool(manifest.get("standalone_verified")),
        "models_dir": manifest["_resolved_models_dir"],
        "external_models": {row["path"]: row["sha256"]
                            for row in manifest.get("external_models", [])},
        "device": device, "threads": threads, "parallel_hemispheres": True,
        "itk_threads": 1, "started_utc": started,
        "elapsed_seconds": time.monotonic() - began,
        "return_code": completed.returncode, "missing_outputs": missing_outputs,
        "effective_config_sha256": actual_config_sha,
        "effective_config_matches_profile": config_match,
        "subject_dir": str(subject_dir), "log": str(log_path),
    }
    report_path = output_root / f"{subject}.recon-all.run.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    if completed.returncode or missing_outputs or not config_match:
        raise RuntimeError(f"reconstruction failed; inspect {report_path} and {log_path}")
    return report


def run_recon_all_batch(
    jobs: list[dict], bundle_root: str | Path, *,
    devices: tuple[str, ...] = ("cuda:0",), threads: int = 4,
    license_file: str | Path | None = None, development_bundle: bool = False,
    models_dir: str | Path | None = None,
) -> list[dict]:
    """Run independent subjects concurrently, at most one per CUDA device.

    Each job supplies ``t1``, ``subject``, and its own empty ``subjects_dir``.
    Reports retain input order. Completed jobs keep their outputs if another
    job fails; after every worker finishes, failures raise RuntimeError.
    """
    if not isinstance(devices, (tuple, list)) or not devices \
            or any(not isinstance(device, str) or not re.fullmatch(r"cuda:\d+", device)
                   for device in devices) or len(set(devices)) != len(devices):
        raise ValueError("devices must be distinct CUDA device names")
    bundle = Path(bundle_root).expanduser().resolve()
    prepared, roots = [], []
    for index, job in enumerate(jobs):
        if not isinstance(job, dict) or set(job) != {"t1", "subject", "subjects_dir"}:
            raise ValueError(f"job {index} must contain t1, subject, and subjects_dir")
        image = Path(job["t1"]).expanduser().resolve(strict=True)
        subject = job["subject"]
        root = Path(job["subjects_dir"]).expanduser().resolve()
        if not image.is_file() or not isinstance(subject, str) or not subject \
                or subject in (".", "..") or Path(subject).name != subject:
            raise ValueError(f"job {index} has an invalid T1 or subject")
        if root.exists() and (not root.is_dir() or any(root.iterdir())):
            raise FileExistsError(f"job {index} requires an empty subjects_dir: {root}")
        if root == bundle or root.is_relative_to(bundle):
            raise ValueError(f"job {index} subjects_dir cannot be inside the bundle: {root}")
        if any(root == other or root.is_relative_to(other) or other.is_relative_to(root)
               for other in roots):
            raise ValueError(f"job {index} has an overlapping subjects_dir: {root}")
        prepared.append((image, subject, root))
        roots.append(root)

    assignments = [[] for _ in devices]
    for index, job in enumerate(prepared):
        assignments[index % len(devices)].append((index, job))

    def run_on_device(device, assigned):
        completed = []
        for index, (image, subject, root) in assigned:
            try:
                report = run_recon_all(
                    image, subject, root, bundle_root, device=device, threads=threads,
                    license_file=license_file, development_bundle=development_bundle,
                    models_dir=models_dir)
            except Exception as error:
                completed.append((index, None, f"{type(error).__name__}: {error}"))
            else:
                completed.append((index, report, None))
        return completed

    reports, errors = [None] * len(prepared), []
    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        futures = [pool.submit(run_on_device, device, assigned)
                   for device, assigned in zip(devices, assignments) if assigned]
        for future in futures:
            for index, report, error in future.result():
                reports[index] = report
                if error:
                    errors.append(f"job {index} ({prepared[index][1]}): {error}")
    if errors:
        raise RuntimeError("batch reconstruction failed: " + "; ".join(errors))
    return reports


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input", required=True, type=Path)
    parser.add_argument("-s", "--subject", required=True)
    parser.add_argument("-sd", "--subjects-dir", required=True, type=Path)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--models-dir", type=Path,
                        help="Directory containing the verified external recon-all models")
    parser.add_argument("--license", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--development-bundle", action="store_true",
                        help="allow an unverified bundle only for development")
    args = parser.parse_args(argv)
    print(json.dumps(run_recon_all(
        args.input, args.subject, args.subjects_dir, args.bundle,
        device=args.device, threads=args.threads, license_file=args.license,
        development_bundle=args.development_bundle, models_dir=args.models_dir,
    ), indent=2))


if __name__ == "__main__":
    main()
