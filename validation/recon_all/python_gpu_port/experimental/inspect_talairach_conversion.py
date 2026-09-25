"""Inspect archived FreeSurfer affine geometry without invoking native tools."""

import argparse
import json
from pathlib import Path

import numpy as np
import surfa as sf


def read_xfm(path: Path) -> np.ndarray:
    lines = path.read_text().splitlines()
    start = lines.index("Linear_Transform =") + 1
    rows = [[float(x) for x in line.strip().rstrip(";").split()]
            for line in lines[start:start + 3]]
    return np.array(rows + [[0, 0, 0, 1]], np.float64)


def matmul_f32(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    out = np.zeros((4, 4), np.float32)
    left, right = np.asarray(left, np.float32), np.asarray(right, np.float32)
    for row in range(4):
        for col in range(4):
            val = np.float32(0)
            for index in range(4):
                val = np.float32(val + np.float32(left[row, index] * right[index, col]))
            out[row, col] = val
    return out


def native_style(aff: sf.Affine) -> tuple[np.ndarray, np.ndarray]:
    src = np.asarray(aff.source.vox2world.matrix, np.float32)
    dst = np.asarray(aff.target.vox2world.matrix, np.float32)
    src_inv = np.linalg.inv(src.astype(np.float64)).astype(np.float32)
    dst_inv = np.linalg.inv(dst.astype(np.float64)).astype(np.float32)
    voxel = matmul_f32(dst_inv, matmul_f32(aff.matrix, src))
    world = matmul_f32(dst, matmul_f32(voxel, src_inv))
    return voxel, world


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("aff", type=Path)
    parser.add_argument("vox", type=Path)
    parser.add_argument("xfm", type=Path)
    args = parser.parse_args()
    aff = sf.load_affine(str(args.aff))
    vox = sf.load_affine(str(args.vox))
    xfm = read_xfm(args.xfm)
    to_vox = aff.convert(space="voxel")
    back = to_vox.convert(space="world")
    native_vox_back = vox.convert(space="world")
    trial_vox, trial_world = native_style(aff)
    _, trial_native_vox_world = native_style(vox.convert(space="world"))
    output = {
        "aff_space": str(aff.space), "vox_space": str(vox.space),
        "aff_matrix": np.asarray(aff.matrix).tolist(),
        "native_vox_matrix": np.asarray(vox.matrix).tolist(),
        "surfa_vox_matrix": np.asarray(to_vox.matrix).tolist(),
        "native_xfm_matrix": xfm.tolist(),
        "surfa_world_roundtrip_matrix": np.asarray(back.matrix).tolist(),
        "native_vox_to_world_matrix": np.asarray(native_vox_back.matrix).tolist(),
        "trial_vox_matrix": trial_vox.tolist(),
        "trial_world_matrix": trial_world.tolist(),
        "trial_native_vox_world_matrix": trial_native_vox_world.tolist(),
        "trial_vox_vs_native_max_abs": float(np.max(np.abs(trial_vox - vox.matrix))),
        "trial_world_vs_xfm_max_abs": float(np.max(np.abs(trial_world - xfm))),
        "surfa_vox_vs_native_max_abs": float(np.max(np.abs(to_vox.matrix - vox.matrix))),
        "surfa_world_vs_xfm_max_abs": float(np.max(np.abs(back.matrix - xfm))),
        "native_vox_to_world_vs_xfm_max_abs": float(np.max(np.abs(native_vox_back.matrix - xfm))),
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
