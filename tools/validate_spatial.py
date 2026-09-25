"""Compare network sampling against TF, and final image application against Surfa.

Example (FreeSurfer sourced, package installed in the current Python):
python tools/validate_spatial.py --out-dir validation/spatial --device cuda
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


def build_cases(directory):
    rng = np.random.default_rng(19)
    arrays, cases = {}, []

    def case(name, operation, **kwargs):
        cases.append(dict(name=name, operation=operation, **kwargs))

    def field(shape, scale=0.2):
        indices = np.stack(np.meshgrid(*[np.arange(n) for n in shape], indexing="ij"), -1)
        return (scale * np.sin(indices * 0.37 + np.arange(3))).astype("float32")

    arrays["affine"] = np.array([[1.02, .08, 0, .2], [-.03, .95, .04, -.7],
                                 [.01, 0, 1.04, .8]], dtype="float32")
    angle = np.deg2rad(17)
    arrays["rotation"] = np.array([[np.cos(angle), -np.sin(angle), 0, .3],
                                   [np.sin(angle), np.cos(angle), 0, -.2],
                                   [0, 0, 1, .7]], dtype="float32")
    arrays["identity"] = np.eye(4, dtype="float32")[:3]
    arrays["half"] = arrays["identity"].copy()
    arrays["half"][:, 3] = (.5, -.5, 1.5)

    shapes = ((7, 9, 11), (1, 6, 5), (31, 35, 33))
    for index, shape in enumerate(shapes):
        image = f"image{index}"
        arrays[image] = rng.normal(size=(*shape, 2)).astype("float32")
        output_shape = (5, 8, 6) if index == 0 else shape
        warp = f"warp{index}"
        arrays[warp] = field(output_shape)
        for trans in ("identity", "half", "rotation", "affine", warp):
            for method in ("linear", "nearest"):
                for fill in (None, 0, -7):
                    case(f"resample_s{index}_{trans}_{method}_fill{fill}", "transform",
                         image=image, trans=trans, shape=output_shape, method=method, fill=fill)
        case(f"dense_s{index}", "dense", matrix="affine", shape=output_shape)
        case(f"dense_right_warp_s{index}", "dense", matrix="rotation", shape=output_shape,
             warp=warp)

    coordinates = np.stack(np.meshgrid(
        [-1e-4, 0, .5], [.5, 8, 8 + 1e-4, 100], [-100, .5, 4.5, 10, 10 + 1e-4],
        indexing="ij"), -1).astype("float32")
    base = np.stack(np.meshgrid(np.arange(3), np.arange(4), np.arange(5), indexing="ij"), -1)
    arrays["boundary"] = (coordinates - base).astype("float32")
    for method in ("linear", "nearest"):
        for fill in (None, 0, -7):
            case(f"boundary_{method}_fill{fill}", "transform", image="image0",
                 trans="boundary", shape=(3, 4, 5), method=method, fill=fill)

    arrays["left_warp"] = field(shapes[0], .35)
    for name, transforms in (
        ("affine_affine", ["affine", "rotation"]),
        ("affine_dense", ["affine", "warp0"]),
        ("dense_affine", ["left_warp", "rotation"]),
        ("dense_dense", ["left_warp", "warp0"]),
        ("affine_dense_affine", ["affine", "left_warp", "rotation"]),
    ):
        case(f"compose_{name}", "compose", transforms=transforms, shape=(5, 8, 6))
    for steps in (0, 5, 7):
        case(f"integrate_steps{steps}", "integrate", warp="left_warp", steps=steps)
    for trans in ("rotation", "warp0"):
        for method in ("linear", "nearest"):
            case(f"batch2_shared_{trans}_{method}", "transform", image="image0", trans=trans,
                 shape=(5, 8, 6), method=method, fill=0, batch=2)

    # World-space registration with unlike voxel sizes, orientations, origins and shapes.
    arrays["source_geometry"] = np.array([[0, -1.2, 0, 8], [1.6, 0, 0, -2],
                                          [0, 0, 2.1, 3], [0, 0, 0, 1]], dtype="float64")
    arrays["target_geometry"] = np.array([[1.1, 0, 0, -1], [0, 1.4, 0, 3],
                                          [0, 0, .9, 5], [0, 0, 0, 1]], dtype="float64")
    arrays["world_forward"] = np.eye(4)
    arrays["world_forward"][:3] = arrays["rotation"]
    arrays["image_target"] = rng.normal(size=(5, 8, 6, 1)).astype("float32")
    arrays["image_source"] = arrays["image0"][..., :1]
    forward = np.linalg.inv(arrays["source_geometry"]) @ np.linalg.inv(arrays["world_forward"]) @ arrays["target_geometry"]
    backward = np.linalg.inv(arrays["target_geometry"]) @ arrays["world_forward"] @ arrays["source_geometry"]
    arrays["forward_pull"], arrays["inverse_pull"] = forward.astype("float32"), backward.astype("float32")
    for direction, image, target, source_geom, target_geom, pull in (
        ("forward", "image_source", "image_target", "source_geometry", "target_geometry", "forward_pull"),
        ("inverse", "image_target", "image_source", "target_geometry", "source_geometry", "inverse_pull"),
    ):
        shape = arrays[target].shape[:3]
        mesh = np.stack(np.meshgrid(*[np.arange(n, dtype="float32") for n in shape], indexing="ij"))
        matrix = arrays[pull]
        warp = (matrix[:3, :3] @ mesh.reshape(3, -1) + matrix[:3, 3:4]).reshape(3, *shape) - mesh
        arrays[direction + "_crs"] = warp.transpose(1, 2, 3, 0)
        for kind in ("affine", "warp"):
            for method in ("linear", "nearest"):
                case(f"geometry_{direction}_{kind}_{method}", "geometry", image=image,
                     target=target, source_geom=source_geom, target_geom=target_geom,
                     pull=pull, direction=direction, kind=kind, method=method)
    np.savez_compressed(directory / "inputs.npz", **arrays)
    (directory / "cases.json").write_text(json.dumps(cases, indent=2) + "\n")


def backend_run(args):
    directory = Path(args.out_dir)
    data = np.load(directory / "inputs.npz")
    cases = json.loads((directory / "cases.json").read_text())
    outputs, errors = {}, {}
    if args.backend == "tf":
        import tensorflow as tf
        import voxelmorph as vxm
        from voxelmorph.tf import utils as spatial
        version = dict(tensorflow=tf.__version__, voxelmorph=getattr(vxm, "__version__", "unknown"))
        cast = lambda value: tf.convert_to_tensor(value, dtype=tf.float32)
        unpack = lambda value: value.numpy()
    else:
        import torch
        from freesurfer_torch.synthmorph import spatial
        torch.set_num_threads(2)
        version = dict(torch=torch.__version__, device=args.device)
        cast = lambda value: torch.as_tensor(value, dtype=torch.float32, device=args.device)
        unpack = lambda value: value.detach().cpu().numpy()
    version["spatial_source"] = spatial.__file__
    version["spatial_source_sha256"] = hashlib.sha256(Path(spatial.__file__).read_bytes()).hexdigest()

    def tensor(key):
        value = data[key]
        if args.backend == "torch" and value.ndim == 4:
            value = value.transpose(3, 0, 1, 2)[None]
        return cast(value)

    def canonical(value):
        value = unpack(value)
        if args.backend == "torch" and value.ndim == 5:
            return value[0].transpose(1, 2, 3, 0)
        return value[:3] if value.shape == (4, 4) else value

    for case in cases:
        name, operation = case["name"], case["operation"]
        try:
            if operation == "transform":
                kwargs = dict(shape=case["shape"], fill_value=case["fill"])
                if args.backend == "tf":
                    result = spatial.transform(tensor(case["image"]), tensor(case["trans"]),
                                               interp_method=case["method"], shift_center=False, **kwargs)
                    if case.get("batch") == 2:
                        result = tf.stack((result, spatial.transform(tensor(case["image"]) * .7,
                            tensor(case["trans"]), interp_method=case["method"], shift_center=False, **kwargs)))
                else:
                    image = tensor(case["image"])
                    if case.get("batch") == 2:
                        image = torch.cat((image, image * .7), 0)
                    result = spatial.transform(image, tensor(case["trans"]), method=case["method"], **kwargs)
                    if case.get("batch") == 2:
                        outputs[name] = unpack(result).transpose(0, 2, 3, 4, 1)
                        continue
                if case.get("batch") == 2:
                    outputs[name] = unpack(result)
                    continue
            elif operation == "dense":
                kwargs = dict(warp_right=tensor(case["warp"])) if "warp" in case else {}
                if args.backend == "tf":
                    result = spatial.affine_to_dense_shift(tensor(case["matrix"]), case["shape"], shift_center=False, **kwargs)
                else:
                    result = spatial.dense(tensor(case["matrix"]), case["shape"], **kwargs)
            elif operation == "compose":
                kwargs = dict(shift_center=False) if args.backend == "tf" else {}
                result = spatial.compose([tensor(key) for key in case["transforms"]], shape=case["shape"], **kwargs)
            elif operation == "integrate":
                result = spatial.integrate_vec(tensor(case["warp"]), nb_steps=case["steps"]) if args.backend == "tf" else spatial.integrate(tensor(case["warp"]), steps=case["steps"])
            else:
                import surfa as sf
                shape = data[case["target"]].shape[:3]
                source = sf.Volume(data[case["image"]][..., 0], geometry=sf.ImageGeometry(
                    data[case["image"]].shape[:3], vox2world=data[case["source_geom"]]))
                target = sf.ImageGeometry(shape, vox2world=data[case["target_geom"]])
                if case["kind"] == "affine":
                    matrix = data["world_forward"]
                    if case["direction"] == "inverse":
                        matrix = np.linalg.inv(matrix)
                    transformation = sf.Affine(matrix, source=source.geom, target=target, space="world")
                else:
                    transformation = sf.Warp(data[case["direction"] + "_crs"],
                        source=source.geom, target=target,
                        format=sf.Warp.Format.disp_crs).convert(format=sf.Warp.Format.disp_ras)
                if args.backend == "tf":
                    # The original registration/apply commands use Surfa here,
                    # with different nearest ties and boundary rules from TF.
                    result = source.transform(transformation, method=case["method"], fill=0)
                else:
                    from freesurfer_torch.synthmorph import apply_transform
                    result = apply_transform(source, transformation, method=case["method"])
                version["geometry_backend"] = f"surfa {getattr(sf, '__version__', 'unknown')}"
                outputs[name] = result.data[..., None]
                outputs[name + "__geometry"] = result.geom.vox2world.matrix
                continue
            outputs[name] = canonical(result)
        except Exception as error:
            errors[name] = f"{type(error).__name__}: {error}"
    np.savez_compressed(directory / f"{args.backend}.npz", **outputs)
    (directory / f"{args.backend}_metadata.json").write_text(json.dumps(dict(versions=version, errors=errors), indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tf-python", default=str(Path(os.environ.get("FREESURFER_HOME", "")) / "bin/fspython"))
    parser.add_argument("--backend", choices=("tf", "torch"))
    args = parser.parse_args()
    if args.backend:
        backend_run(args)
        return
    directory = Path(args.out_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    build_cases(directory)
    for backend, python in (("tf", args.tf_python), ("torch", sys.executable)):
        environment = os.environ.copy()
        environment.update(VXM_BACKEND="tensorflow", NEURITE_BACKEND="tensorflow",
                           TF_NUM_INTRAOP_THREADS="2", TF_NUM_INTEROP_THREADS="2")
        if backend == "tf":
            environment["CUDA_VISIBLE_DEVICES"] = ""
        with (directory / f"{backend}.log").open("w") as log:
            subprocess.run([python, str(Path(__file__).resolve()), "--out-dir", str(directory),
                            "--backend", backend, "--device", args.device], env=environment,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    reference, candidate = np.load(directory / "tf.npz"), np.load(directory / "torch.npz")
    report = {"cases": {}, "backends": {name: json.loads((directory / f"{name}_metadata.json").read_text()) for name in ("tf", "torch")}}
    for name in sorted(set(reference.files) | set(candidate.files)):
        if name not in reference or name not in candidate:
            report["cases"][name] = dict(passed=False, missing="tf" if name not in reference else "torch")
            continue
        first, second = reference[name], candidate[name]
        if first.shape != second.shape:
            report["cases"][name] = dict(passed=False, reference_shape=first.shape, candidate_shape=second.shape)
            continue
        delta = np.abs(first.astype("float64") - second.astype("float64"))
        tolerance = 0 if "nearest" in name and not name.endswith("__geometry") else 5e-5
        report["cases"][name] = dict(max_abs_error=float(delta.max()), mean_abs_error=float(delta.mean()),
                                    tolerance=tolerance, passed=bool(np.isfinite(delta).all() and delta.max() <= tolerance))
    report["passed"] = not any(info["errors"] for info in report["backends"].values()) and all(info["passed"] for info in report["cases"].values())
    (directory / "spatial_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    failures = {key: value for key, value in report["cases"].items() if not value["passed"]}
    print(json.dumps(dict(passed=report["passed"], checks=len(report["cases"]), failures=failures), indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
