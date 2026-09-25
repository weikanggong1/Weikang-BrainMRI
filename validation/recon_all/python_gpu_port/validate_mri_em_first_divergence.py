"""Check native/Python EM objectives and first gradient on fixed sub-01."""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from fnit.recon_all.mri_em_register import (
    _vnl_affine_inverse, atlas_label_peak, estimate_image_white_matter_peak, find_all_samples,
    read_gca, read_masked_input, scale_input_intensity,
)
from fnit.recon_all.mri_em_register_em_candidate import EMObjective


# Direct float32 readings at the native objective call and return, before EM
# and at its first two line-search trials. These are validation fixtures only.
NATIVE_PRE = np.array([
    [1.1575514078140259, .07681363821029663, -.1292986124753952, -19.754428863525391],
    [-.060859519988298416, 1.2991782426834106, .31405520439147949, -52.992832183837891],
    [.090609326958656311, -.28655186295509338, 1.0190079212188721, 4.7753071784973145],
    [0, 0, 0, 1],
], np.float32)
NATIVE_TINY = NATIVE_PRE.copy()
NATIVE_TINY[:3, :3] = np.array([
    [1.1575514078140259, .076813668012619019, -.12929858267307281],
    [-.060859505087137222, 1.2991782426834106, .31405520439147949],
    [.090609341859817505, -.28655186295509338, 1.0190079212188721],
], np.float32)
NATIVE_LARGE = NATIVE_PRE.copy()
NATIVE_LARGE[:3, :3] = np.array([
    [1.6830334663391113, .55765533447265625, .42873859405517578],
    [.15606524050235748, 1.4581756591796875, .52038729190826416],
    [.27358043193817139, -.15276971459388733, 1.141355037689209],
], np.float32)
NATIVE_COST = {
    "pre": np.float32(3.9178197383880615),
    "large": np.float32(77444.578125),
    "tiny": np.float32(3.9178192615509033),
}
NATIVE_FIRST_GRADIENT = np.array([
    [-.015293490141630173, -.013994289562106133, -.01624096743762493, 0.],
    [-.006313320714980364, -.004627417307347059, -.0060050347819924355, 0.],
    [-.005325142294168472, -.003893560264259577, -.0035607577301561832, 0.],
])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("atlas", type=Path)
    parser.add_argument("nu", type=Path)
    parser.add_argument("mask", type=Path)
    args = parser.parse_args()
    started = time.perf_counter()
    atlas = read_gca(args.atlas)
    masked = read_masked_input(args.nu, args.mask)
    peak, _, _ = estimate_image_white_matter_peak(atlas, masked)
    source = scale_input_intensity(masked, atlas_label_peak(atlas, 2), peak)
    objective = EMObjective(atlas, find_all_samples(atlas), source)
    trace = json.loads(Path(__file__).with_name("em_native_trace_fs_sub01.json").read_text())
    full_trace = json.loads(Path(__file__).with_name("em_native_full_trace_fs_sub01.json").read_text())
    assert len(objective.samples.labels) == trace["sample_count"]
    costs = {name: np.float32(objective.cost(matrix)) for name, matrix in (
        ("pre", NATIVE_PRE), ("large", NATIVE_LARGE), ("tiny", NATIVE_TINY)
    )}
    trace_cost_errors = []
    for trial in trace["trials"]:
        matrix = np.eye(4, dtype=np.float32)
        matrix[:3] = np.asarray(trial["matrix_rows_1_to_3"], np.float32)
        trace_cost_errors.append(float(np.float32(objective.cost(matrix)) -
                                       np.float32(trial["cost"])))
    full_trace_cost_errors = []
    for trial in full_trace["trials"]:
        matrix = np.eye(4, dtype=np.float32)
        matrix[:3] = np.asarray(trial["matrix_rows_1_to_3"], np.float32)
        full_trace_cost_errors.append(float(np.float32(objective.cost(matrix)) -
                                            np.float32(trial["cost"])))
    scale = np.diag(np.array([2, 2, 2, 1], np.float32))
    prior_matrix_errors = []
    for label, index in (("first", 0), ("trial_13", 12)):
        matrix = np.eye(4, dtype=np.float32)
        matrix[:3] = np.asarray(trace["trials"][index]["matrix_rows_1_to_3"], np.float32)
        observed = _vnl_affine_inverse(matrix) @ scale
        expected = np.asarray(trace["prior_to_source_matrices"][label], np.float32)
        prior_matrix_errors.append(int(np.count_nonzero(observed != expected)))
    gradient_error = float(np.max(np.abs(objective.gradient(NATIVE_PRE) -
                                         NATIVE_FIRST_GRADIENT)))
    result = {
        "samples": len(objective.samples.labels),
        "native_cost": {key: float(value) for key, value in NATIVE_COST.items()},
        "python_cost": {key: float(value) for key, value in costs.items()},
        "first_divergence": "pre" if costs["pre"] != NATIVE_COST["pre"] else
                            "tiny" if costs["tiny"] != NATIVE_COST["tiny"] else None,
        "native_tiny_improvement": float(NATIVE_COST["pre"] - NATIVE_COST["tiny"]),
        "python_tiny_improvement": float(costs["pre"] - costs["tiny"]),
        "first_gradient_max_abs_error": gradient_error,
        "native_trace_objectives_exact": len(trace_cost_errors) -
                                         int(np.count_nonzero(trace_cost_errors)),
        "native_trace_objectives_total": len(trace_cost_errors),
        "full_native_trace_objectives_exact": len(full_trace_cost_errors) -
                                              int(np.count_nonzero(full_trace_cost_errors)),
        "full_native_trace_objectives_total": len(full_trace_cost_errors),
        "full_native_trace_max_abs_error": max(map(abs, full_trace_cost_errors)),
        "prior_matrix_mismatched_elements": prior_matrix_errors,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "backend": "CPU NumPy/SciPy; no PyTorch operations",
    }
    print(json.dumps(result, indent=2))
    return int(any(costs[key] != NATIVE_COST[key] for key in NATIVE_COST) or
               any(trace_cost_errors) or any(full_trace_cost_errors) or
               any(prior_matrix_errors) or
               gradient_error > 2e-7)


if __name__ == "__main__":
    raise SystemExit(main())
