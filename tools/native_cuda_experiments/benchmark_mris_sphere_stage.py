"""Run one frozen mris_sphere input through official, rebuilt, and CUDA binaries."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import nibabel as nib
import numpy as np


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def compare(reference, current, max_abs_mm):
    vertices_a, faces_a = nib.freesurfer.read_geometry(str(reference))
    vertices_b, faces_b = nib.freesurfer.read_geometry(str(current))
    if vertices_a.shape != vertices_b.shape or faces_a.shape != faces_b.shape:
        return {"shape_match": False, "pass": False}
    difference = np.abs(vertices_a - vertices_b)
    faces_equal = bool(np.array_equal(faces_a, faces_b))
    return {
        "shape_match": True,
        "vertices": len(vertices_a),
        "faces": len(faces_a),
        "faces_equal": faces_equal,
        "coordinate_values_different": int(np.count_nonzero(vertices_a != vertices_b)),
        "max_abs_coordinate_mm": float(difference.max()),
        "mean_abs_coordinate_mm": float(difference.mean()),
        "p99_abs_coordinate_mm": float(np.percentile(difference, 99)),
        "pass": faces_equal and bool(np.isfinite(vertices_b).all())
                and float(difference.max()) <= max_abs_mm,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--existing-sphere", type=Path,
                        help="completed official recon-all sphere for replay check")
    parser.add_argument("--official-bin", type=Path, required=True)
    parser.add_argument("--clean-bin", type=Path, required=True)
    parser.add_argument("--cuda-bin", type=Path, required=True)
    parser.add_argument("--fs-home", type=Path, required=True)
    parser.add_argument("--license", type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="fresh output directory")
    parser.add_argument("--max-abs-mm", type=float, default=1e-5)
    args = parser.parse_args()
    paths = {name: value.resolve(strict=True) for name, value in (
        ("input", args.input), ("official", args.official_bin),
        ("clean", args.clean_bin), ("patched", args.cuda_bin),
        ("fs_home", args.fs_home), ("license", args.license))}
    if args.existing_sphere:
        paths["existing_sphere"] = args.existing_sphere.resolve(strict=True)
    if not args.gpu_uuid.startswith("GPU-") or args.max_abs_mm < 0:
        parser.error("require a GPU UUID and nonnegative coordinate tolerance")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    report = {"hostname": platform.node(), "input": str(paths["input"]),
              "input_sha256": digest(paths["input"]), "gpu_uuid": args.gpu_uuid,
              "max_abs_mm_gate": args.max_abs_mm,
              "binary_sha256": {name: digest(paths[name]) for name in
                                ("official", "clean", "patched")}, "runs": {}}
    if "existing_sphere" in paths:
        report["existing_sphere_sha256"] = digest(paths["existing_sphere"])
    for label, binary, use_cuda in (
        ("official", "official", False), ("clean", "clean", False),
        ("patched_cpu", "patched", False), ("patched_cuda", "patched", True)):
        output = root / (label + ".sphere")
        command = [str(paths[binary]), "-threads", "4", "-seed", "1234",
                   str(paths["input"]), str(output)]
        env = os.environ.copy()
        env.update(FREESURFER_HOME=str(paths["fs_home"]),
                   FS_LICENSE=str(paths["license"]),
                   CUDA_VISIBLE_DEVICES=args.gpu_uuid,
                   OMP_NUM_THREADS="4", FS_OMP_NUM_THREADS="4")
        env.pop("FS_SPHERE_CUDA_GRADIENTS", None)
        if use_cuda:
            env["FS_SPHERE_CUDA_GRADIENTS"] = "1"
        log = root / (label + ".log")
        load_start = os.getloadavg()[0]
        began = time.monotonic()
        with log.open("w") as stream:
            result = subprocess.run(command, env=env, stdout=stream,
                                    stderr=subprocess.STDOUT, check=False)
        run = {"command": command, "elapsed_seconds": time.monotonic() - began,
               "load_1m_start": load_start, "load_1m_end": os.getloadavg()[0],
               "exit_code": result.returncode, "log": str(log),
               "output": str(output), "output_exists": output.is_file(),
               "cuda_active": "MRISaverageGradients: CUDA active" in log.read_text(errors="replace")}
        report["runs"][label] = run
        (root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        if result.returncode or not output.is_file() or (use_cuda and not run["cuda_active"]):
            raise SystemExit(f"{label} failed; inspect {log} and report.json")
    baseline = root / "official.sphere"
    comparisons = {}
    for label, left, right in (
        ("clean_vs_official", baseline, root / "clean.sphere"),
        ("patched_cpu_vs_clean", root / "clean.sphere", root / "patched_cpu.sphere"),
        ("patched_cuda_vs_cpu", root / "patched_cpu.sphere", root / "patched_cuda.sphere"),
        ("patched_cuda_vs_official", baseline, root / "patched_cuda.sphere")):
        comparisons[label] = compare(left, right, args.max_abs_mm)
    if "existing_sphere" in paths:
        comparisons["official_replay_vs_existing"] = compare(
            paths["existing_sphere"], baseline, args.max_abs_mm)
    report["comparisons"] = comparisons
    report["all_pass"] = all(item["pass"] for item in comparisons.values())
    (root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"all_pass": report["all_pass"], "runs": report["runs"],
                      "comparisons": comparisons}, indent=2))
    if not report["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
