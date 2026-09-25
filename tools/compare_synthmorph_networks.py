"""Run separately with FreeSurfer fspython (tf) and project Python (torch)."""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("backend", choices=("tf", "torch"))
p.add_argument("output")
p.add_argument("--weights", required=True)
p.add_argument("--mode", choices=("affine", "rigid", "deform", "joint"), default="deform")
p.add_argument("--extent", type=int, default=64)
p.add_argument("--hyper", type=float, default=0.5)
p.add_argument("--device", default="cuda")
a = p.parse_args()
size = a.extent
grid = np.stack(np.meshgrid(*[np.linspace(-1, 1, size)] * 3, indexing="ij"))
random = np.random.RandomState(417)
moving = np.exp(-np.sum(grid**2 * np.array([2, 3, 5])[:, None, None, None], axis=0))
fixed = np.exp(-np.sum((grid - np.array([0.05, -0.07, 0.03])[:, None, None, None])**2
                      * np.array([2.2, 2.9, 4.6])[:, None, None, None], axis=0))
moving = (moving + random.uniform(0, 0.05, moving.shape)).astype("float32")
fixed = (fixed + random.uniform(0, 0.05, fixed.shape)).astype("float32")
weights = {"affine": str(Path(a.weights) / "synthmorph.affine.2.h5"),
           "rigid": str(Path(a.weights) / "synthmorph.rigid.1.h5"),
           "deform": str(Path(a.weights) / "synthmorph.deform.3.h5")}
results = {}
start = time.time()
if a.backend == "tf":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    os.environ["NEURITE_BACKEND"] = os.environ["VXM_BACKEND"] = "tensorflow"
    if a.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import tensorflow as tf
    tf.config.experimental.enable_tensor_float_32_execution(False)
    import voxelmorph as vxm
    from synthmorph.registration import load_weights
    tf.config.threading.set_inter_op_parallelism_threads(4)
    tf.config.threading.set_intra_op_parallelism_threads(4)
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    inputs = [moving[None, ..., None], fixed[None, ..., None]]
    if a.mode in ("affine", "rigid"):
        model = vxm.networks.VxmAffineFeatureDetector(in_shape=(size,) * 3, make_dense=False,
                                                     half_res=False, bidir=True, return_feat=True,
                                                     rigid=a.mode == "rigid")
        load_weights(model, weights[a.mode])
        outputs = model(inputs)
        names = ("forward", "backward", "feat1", "feat2")
        for name, value in zip(names, outputs):
            value = value.numpy()
            if name.startswith("feat"):
                value = value.transpose(0, 4, 1, 2, 3)
            else:
                value = np.concatenate([value[0], np.array([[0, 0, 0, 1]], dtype="float32")])
            results[name] = value
    else:
        model = vxm.networks.HyperVxmJoint(in_shape=(size,) * 3, bidir=True, mid_space=True,
                                          skip_affine=a.mode == "deform", return_svf=True)
        if a.mode == "joint":
            load_weights(model, weights["affine"])
        load_weights(model, weights["deform"])
        outputs = model([np.array([[a.hyper]], dtype="float32"), *inputs])
        for name, value in zip(("forward", "backward", "velocity", "negative_velocity"), outputs):
            results[name] = value.numpy().transpose(0, 4, 1, 2, 3)
else:
    import torch
    from fnit.synthmorph.models import AffineNetwork, SynthMorphNetwork
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    inputs = [torch.from_numpy(v[None, None]).to(a.device) for v in (moving, fixed)]
    with torch.inference_mode():
        if a.mode in ("affine", "rigid"):
            model = AffineNetwork(weights[a.mode], rigid=a.mode == "rigid").eval().to(a.device)
            outputs = model(*inputs, half_res=False, return_features=True)
            results = {key: value.cpu().numpy() for key, value in
                       zip(("forward", "backward", "feat1", "feat2"), outputs)}
        else:
            model = SynthMorphNetwork(weights, model=a.mode, hyper=a.hyper, device=a.device)
            forward, backward, extra = model(*inputs, return_intermediates=True)
            results = {"forward": forward.cpu().numpy(), "backward": backward.cpu().numpy(),
                       "velocity": extra["velocity"].cpu().numpy()}
    reference = Path(a.output).with_name(Path(a.output).name.replace("torch", "tf"))
    if reference.exists():
        ref = np.load(reference)
        report = {}
        for key, value in results.items():
            diff = value.astype("float64") - ref[key].astype("float64")
            report[key] = {"mae": float(np.mean(np.abs(diff))), "max_abs": float(np.max(np.abs(diff))),
                           "rmse": float(np.sqrt(np.mean(diff**2))),
                           "reference_rms": float(np.sqrt(np.mean(ref[key].astype("float64")**2)))}
        print(json.dumps(report, indent=2), flush=True)
        Path(a.output + ".comparison.json").write_text(json.dumps(report, indent=2))
np.savez_compressed(a.output, **results)
print(json.dumps({"backend": a.backend, "mode": a.mode, "seconds": time.time() - start,
                  "output": a.output}), flush=True)
