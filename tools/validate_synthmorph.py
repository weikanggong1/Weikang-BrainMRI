#!/usr/bin/env python3
"""Compare the original SynthMorph CLI with the PyTorch public API.

Uses a public template and a deterministic, smoothly deformed/intensity-changed
copy. This measures implementation agreement, not population registration accuracy.
Run under the project's Python with the FreeSurfer module already loaded.
"""
import argparse
import contextlib
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
from scipy.ndimage import map_coordinates
import surfa as sf
import torch

from fnit import SynthMorph


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False,
                                   default=lambda x: x.item() if isinstance(x, np.generic) else x.tolist()) + "\n")
    temporary.replace(path)


def checksum(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_case(image, folder):
    folder.mkdir(parents=True, exist_ok=True)
    source = sf.load_volume(str(image)).astype("float32")
    if len(source.shape) != 3:
        raise ValueError("Validation template must be a single-frame volume")
    coords = np.indices(source.shape, dtype=np.float32)
    normal = np.stack([(coords[d] - (n-1)/2) / ((n-1)/2) for d, n in enumerate(source.shape)])
    x, y, z = normal
    envelope = np.exp(-1.5 * (x*x + y*y + z*z))
    displacement = np.stack((
        0.35 + 1.6 * np.sin(np.pi*y) * np.cos(np.pi*z) * envelope,
        -0.25 + 1.3 * np.sin(np.pi*z) * np.cos(np.pi*x) * envelope,
        0.20 + 1.1 * np.sin(np.pi*x) * np.cos(np.pi*y) * envelope,
    )).astype("float32")
    warped = map_coordinates(source.data, coords + displacement, order=1, mode="constant", cval=0,
                             prefilter=False)
    scale = float(np.max(source.data))
    if scale <= 0:
        raise ValueError("Template must contain positive intensities")
    fixed_data = (np.maximum(warped, 0) / scale)**1.15 * scale * (1 + 0.08*x)
    moving_path, fixed_path = folder / "moving.nii.gz", folder / "fixed.nii.gz"
    source.save(moving_path)
    sf.Volume(fixed_data.astype("float32"), geometry=source.geom).save(fixed_path)
    sf.Volume(displacement.transpose(1, 2, 3, 0), geometry=source.geom).save(folder / "generation_pull_displacement_vox.nii.gz")
    description = {"source": str(image), "source_sha256": checksum(image),
                   "moving": str(moving_path), "fixed": str(fixed_path),
                   "shape": list(source.shape), "vox2ras": source.geom.vox2world.matrix.tolist(),
                   "generation": {"amplitudes_vox": [1.6, 1.3, 1.1], "translation_vox": [0.35, -0.25, 0.2],
                                  "envelope": "exp(-1.5*(x*x+y*y+z*z)); x,y,z in [-1,1]",
                                  "intensity_gamma": 1.15, "multiplicative_bias": "1+0.08*x",
                                  "interpolation": "SciPy map_coordinates order=1, zero exterior"},
                   "scope": "Public-template implementation comparison; not population accuracy validation",
                   "input_pair_rmse": float(np.sqrt(np.mean((source.data-fixed_data)**2)))}
    write_json(folder / "case.json", description)
    return moving_path, fixed_path, description


def errors(reference, candidate):
    reference, candidate = np.asarray(reference), np.asarray(candidate)
    if reference.shape != candidate.shape:
        return {"shape_equal": False, "reference_shape": list(reference.shape), "candidate_shape": list(candidate.shape)}
    if not (np.isfinite(reference).all() and np.isfinite(candidate).all()):
        return {"shape_equal": True, "finite": False}
    delta = candidate.astype(np.float64) - reference.astype(np.float64)
    rmse = float(np.sqrt(np.mean(delta**2)))
    rms = float(np.sqrt(np.mean(reference.astype(np.float64)**2)))
    return {"shape_equal": True, "finite": True, "mae": float(np.mean(np.abs(delta))),
            "max_abs": float(np.max(np.abs(delta))), "rmse": rmse,
            "normalised_rmse": rmse / rms if rms else (0.0 if rmse == 0 else None),
            "normalisation": "reference root-mean-square", "reference_rms": rms}


def geometry(reference, candidate):
    return {"equal": bool(sf.transform.image_geometry_equal(reference, candidate, tol=1e-5)),
            "reference_shape": list(map(int, reference.shape)), "candidate_shape": list(map(int, candidate.shape)),
            "vox2ras_max_abs": float(np.max(np.abs(reference.vox2world.matrix-candidate.vox2world.matrix)))}


def paths(folder, mode):
    extension = ".lta" if mode in ("affine", "rigid") else ".nii.gz"
    return {"moved": folder / "moved.nii.gz", "fixed_moved": folder / "fixed_moved.nii.gz",
            "transform": folder / ("forward" + extension), "inverse": folder / ("inverse" + extension)}


def load_transformation(path):
    return sf.load_affine(str(path)) if str(path).endswith(".lta") else sf.load_warp(str(path))


def pull_coordinates(transformation):
    """Saved LTA/warp -> source voxel coordinates on target voxel grid."""
    if isinstance(transformation, sf.Warp):
        return np.asarray(transformation.convert(format=sf.Warp.Format.abs_crs).data, dtype=np.float32)
    matrix = transformation.convert(space="voxel").inv().matrix
    coords = np.indices(transformation.target.shape, dtype=np.float32)
    return (np.einsum("ij,jxyz->ixyz", matrix[:3, :3], coords)
            + matrix[:3, 3, None, None, None]).transpose(1, 2, 3, 0).astype("float32")


def jacobian(field, foreground):
    # Native voxel coordinates: det(d source_indices / d target_indices).
    a, b, c = [np.gradient(field[..., d], edge_order=1) for d in range(3)]
    determinant = a[0]*(b[1]*c[2]-b[2]*c[1]) - a[1]*(b[0]*c[2]-b[2]*c[0]) + a[2]*(b[0]*c[1]-b[1]*c[0])
    def stats(values):
        return {"voxels": int(values.size), "fraction_nonpositive": float(np.mean(values <= 0)),
                "min": float(values.min()), "p01": float(np.percentile(values, 1)),
                "median": float(np.median(values)), "max": float(values.max())}
    result = {"all": stats(determinant), "convention": "native source voxel indices differentiated by target voxel indices"}
    if np.any(foreground):
        result["foreground"] = stats(determinant[foreground])
    return result


def inverse_consistency(forward, inverse, target_geometry, foreground, stride=3):
    """Sample inverse(forward(x))-x independently of either framework."""
    slices = tuple(slice(1, n-1, stride) for n in forward.shape[:3])
    domain = np.indices(forward.shape[:3], dtype=np.float32).transpose(1, 2, 3, 0)[slices]
    source = forward[slices]
    valid = np.ones(source.shape[:-1], dtype=bool)
    for d, n in enumerate(inverse.shape[:3]):
        valid &= (source[..., d] >= 1) & (source[..., d] <= n-2)
    samples = np.stack([map_coordinates(inverse[..., d], source.transpose(3, 0, 1, 2),
                                       order=1, mode="nearest", prefilter=False) for d in range(3)], axis=-1)
    delta = samples - domain
    distance_vox = np.linalg.norm(delta, axis=-1)
    delta_mm = np.einsum("ij,...j->...i", target_geometry.vox2world.matrix[:3, :3], delta)
    distance_mm = np.linalg.norm(delta_mm, axis=-1)
    def stats(mask):
        if not np.any(mask):
            return {"points": 0}
        result = {"points": int(mask.sum())}
        for unit, values in (("vox", distance_vox[mask]), ("mm", distance_mm[mask])):
            result[unit] = {"mean": float(values.mean()), "rms": float(np.sqrt(np.mean(values**2))),
                            "p95": float(np.percentile(values, 95)), "max": float(values.max())}
        return result
    return {"all_valid": stats(valid), "foreground": stats(valid & foreground[slices]),
            "stride": stride, "grid_points": int(valid.size), "valid_fraction": float(valid.mean()),
            "sampling": "SciPy trilinear; points at least one voxel inside inverse domain"}


def compare(mode, ref_paths, cand_paths, moving, fixed):
    output = {}
    for name in ("moved", "fixed_moved"):
        reference, candidate = [sf.load_volume(str(p[name])) for p in (ref_paths, cand_paths)]
        output[name] = {"data": errors(reference.data, candidate.data),
                        "geometry": geometry(reference.geom, candidate.geom)}
    transforms = {key: [load_transformation(p[key]) for p in (ref_paths, cand_paths)]
                  for key in ("transform", "inverse")}
    for name, (reference, candidate) in transforms.items():
        entry = {"source_geometry": geometry(reference.source, candidate.source),
                 "target_geometry": geometry(reference.target, candidate.target)}
        if mode in ("affine", "rigid"):
            entry["world_matrix"] = errors(reference.convert(space="world").matrix, candidate.convert(space="world").matrix)
            entry["voxel_matrix"] = errors(reference.convert(space="voxel").matrix, candidate.convert(space="voxel").matrix)
        else:
            entry["displacement_ras_mm"] = errors(reference.convert(format=sf.Warp.Format.disp_ras).data,
                                                   candidate.convert(format=sf.Warp.Format.disp_ras).data)
        output[name] = entry
    foregrounds = [np.asarray(v.data) > 0.1 * np.max(v.data) for v in (fixed, moving)]
    native = {key: [pull_coordinates(t) for t in values] for key, values in transforms.items()}
    output["native_forward_pull_coordinates_vox"] = errors(*native["transform"])
    output["native_inverse_pull_coordinates_vox"] = errors(*native["inverse"])
    output["field_diagnostics"] = {}
    for index, backend in enumerate(("reference", "candidate")):
        forward, inverse = native["transform"][index], native["inverse"][index]
        output["field_diagnostics"][backend] = {
            "forward_jacobian": jacobian(forward, foregrounds[0]),
            "inverse_jacobian": jacobian(inverse, foregrounds[1]),
            "inverse_after_forward": inverse_consistency(forward, inverse, fixed.geom, foregrounds[0]),
            "forward_after_inverse": inverse_consistency(inverse, forward, moving.geom, foregrounds[1]),
            "foreground_definition": "target intensity > 10 percent of target maximum"}
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="Public MNI152_T1_2mm template")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=Path(__file__).resolve().parents[1] / "weights")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--reference-device", default="cuda", help="cpu, cuda, or cuda:N")
    parser.add_argument("--extent", type=int, choices=(192, 256), default=192)
    parser.add_argument("--modes", nargs="+", choices=("affine", "rigid", "deform", "joint"), default=["affine", "rigid", "deform", "joint"])
    parser.add_argument("--hyper", type=float, default=0.5)
    parser.add_argument("--steps", type=int, default=7)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    args.image, args.out_dir, args.weights = [p.resolve() for p in (args.image, args.out_dir, args.weights)]
    if not args.weights.is_dir():
        parser.error("--weights must name the explicit project checkpoint directory")
    if not os.environ.get("FREESURFER_HOME"):
        parser.error("Load the FreeSurfer module first (FREESURFER_HOME is unset)")
    executable = shutil.which("mri_synthmorph")
    if not executable:
        parser.error("Original mri_synthmorph is not on PATH")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    moving_path, fixed_path, case = prepare_case(args.image, args.out_dir / "case")
    moving, fixed = sf.load_volume(str(moving_path)), sf.load_volume(str(fixed_path))
    report = {"scope": case["scope"], "case": case, "settings": {
        "extent": args.extent, "hyper": args.hyper, "steps": args.steps, "device": args.device,
        "reference_device": args.reference_device, "threads": args.threads,
        "candidate_weights": str(args.weights), "reference_weights": str(Path(os.environ["FREESURFER_HOME"]) / "models"),
        "reference_cli": executable, "NVIDIA_TF32_OVERRIDE": "0", "torch_version": torch.__version__,
        "normalised_rmse": "RMSE / root-mean-square(reference)",
        "acceptance": "Raw agreement metrics; no automatic numerical equivalence threshold"}, "modes": {}}
    for mode in args.modes:
        folder = args.out_dir / mode
        ref_dir, cand_dir = folder / "reference", folder / "candidate"
        ref_dir.mkdir(parents=True, exist_ok=True)
        cand_dir.mkdir(parents=True, exist_ok=True)
        ref_paths, cand_paths = paths(ref_dir, mode), paths(cand_dir, mode)
        entry = {"status": "running", "mode": mode}
        model = result = None
        write_json(folder / "report.json", entry)
        command = [executable, "register", "-m", mode, "-e", str(args.extent), "-r", str(args.hyper),
                   "-n", str(args.steps), "-j", str(args.threads), "-o", str(ref_paths["moved"]),
                   "-O", str(ref_paths["fixed_moved"]), "-t", str(ref_paths["transform"]),
                   "-T", str(ref_paths["inverse"])]
        env = os.environ.copy()
        env["NVIDIA_TF32_OVERRIDE"] = "0"
        if args.reference_device.startswith("cuda"):
            command.append("-g")
            if ":" in args.reference_device:
                env["CUDA_VISIBLE_DEVICES"] = args.reference_device.split(":", 1)[1]
        else:
            env["CUDA_VISIBLE_DEVICES"] = ""
        command.extend((str(moving_path), str(fixed_path)))
        entry["reference_command"] = command
        try:
            print(f"[{mode}] original CLI", flush=True)
            start = time.perf_counter()
            with open(ref_dir / "run.log", "w") as log:
                subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            entry["reference_process_seconds"] = time.perf_counter() - start
            print(f"[{mode}] PyTorch API", flush=True)
            with open(cand_dir / "run.log", "w") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                start = time.perf_counter()
                model = SynthMorph(weights=args.weights, device=args.device, model=mode, extent=args.extent,
                                   hyper=args.hyper, steps=args.steps)
                if args.device.startswith("cuda"):
                    torch.cuda.synchronize()
                entry["candidate_model_load_seconds"] = time.perf_counter() - start
                start = time.perf_counter()
                result = model(moving_path, fixed_path)
                if args.device.startswith("cuda"):
                    torch.cuda.synchronize()
                entry["candidate_pair_seconds"] = time.perf_counter() - start
                start = time.perf_counter()
                for name, path in cand_paths.items():
                    getattr(result, name).save(path)
                entry["candidate_save_seconds"] = time.perf_counter() - start
                print(json.dumps({k: v for k, v in entry.items() if k.endswith("seconds")}), flush=True)
            result = model = None
            gc.collect()
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
            print(f"[{mode}] compare saved images and transforms", flush=True)
            entry["comparison"] = compare(mode, ref_paths, cand_paths, moving, fixed)
            entry["status"] = "completed"
        except Exception as exc:
            entry["status"] = "failed"
            entry["error"] = str(exc)
            entry["traceback"] = traceback.format_exc()
            (folder / "failure.log").write_text(entry["traceback"])
            print(f"[{mode}] failed: {exc}", file=sys.stderr, flush=True)
        result = model = None
        gc.collect()
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
        entry["artifacts"] = {"reference": {k: str(v) for k, v in ref_paths.items()},
                              "candidate": {k: str(v) for k, v in cand_paths.items()},
                              "reference_log": str(ref_dir / "run.log"), "candidate_log": str(cand_dir / "run.log")}
        report["modes"][mode] = entry
        write_json(folder / "report.json", entry)
        write_json(args.out_dir / "report.json", report)
    return 0 if all(v["status"] == "completed" for v in report["modes"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
