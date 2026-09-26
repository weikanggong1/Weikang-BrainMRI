"""Test the source-order inverse alone over the frozen RH smoothwm raw H."""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import nibabel.freesurfer.io as fsio
import numpy as np
import torch
from numba import njit

from fnit.recon_all.mris_register_nonlinear import sphere_vertex_normals, tangent_basis
from fnit.recon_all.mris_register_smoothwm import _three_hop_neighbors
from fnit.recon_all.topology_vnl_svd import add, mul, svd_inverse_3, svdc_3


@njit
def solve(gram, rhs):
    coeff = np.zeros((len(gram), 3), np.float32)
    condition = np.empty(len(gram), np.float32)
    for vertex in range(len(gram)):
        _, singular, _ = svdc_3(gram[vertex])
        wmax = np.max(np.abs(singular))
        wmin = np.min(np.abs(singular))
        condition[vertex] = np.float32(1e8 if wmin < 1.1920928955078125e-7 else wmax / wmin)
        inverse = svd_inverse_3(gram[vertex])
        for row in range(3):
            value = np.float32(0)
            for col in range(3):
                value = add(value, mul(inverse[row, col], rhs[vertex, col]))
            coeff[vertex, row] = value
    return coeff, condition


def calculate(xyz, faces, eigen_mean=False):
    normals = sphere_vertex_normals(xyz, faces)
    e1, e2 = tangent_basis(normals)
    neighbors, active = _three_hop_neighbors(faces, len(xyz))
    result = torch.empty(len(xyz), dtype=torch.float32)
    fallback_count = 0
    for first in range(0, len(xyz), 2048):
        stop = min(first + 2048, len(xyz))
        delta = xyz[neighbors[first:stop]] - xyz[first:stop, None]
        u = (delta * e1[first:stop, None]).sum(2)
        v = (delta * e2[first:stop, None]).sum(2)
        z = (delta * normals[first:stop, None]).sum(2)
        rsq = u * u + v * v
        valid = active[first:stop] & (rsq > 1e-12)
        design = torch.stack((u * u, 2 * u * v, v * v), dim=2) * valid[:, :, None]
        height = z * valid
        gram = torch.zeros((stop - first, 3, 3), dtype=torch.float32)
        rhs = torch.zeros((stop - first, 3), dtype=torch.float32)
        for slot in range(design.shape[1]):
            row = design[:, slot]
            gram += row[:, :, None] * row[:, None, :]
            rhs += row * height[:, slot, None]
        coeff, condition = solve(gram.numpy(), rhs.numpy())
        if eigen_mean:
            h00 = np.float32(np.float32(2) * coeff[:, 0]).astype(np.float64)
            h01 = np.float32(np.float32(2) * coeff[:, 1]).astype(np.float64)
            h11 = np.float32(np.float32(2) * coeff[:, 2]).astype(np.float64)
            center = (h00 + h11) / 2.0
            spread = np.sqrt(((h00 - h11) / 2.0) ** 2 + h01 ** 2)
            low = np.float32(center - spread)
            high = np.float32(center + spread)
            fitted = torch.from_numpy(np.float32(np.float32(low + high) * np.float32(0.5)))
        else:
            fitted = torch.from_numpy(np.float32(coeff[:, 0] + coeff[:, 2]))
        k = torch.where(valid, z / rsq.clamp_min(1e-30), 0)
        largest = k.masked_fill(~valid, -torch.inf).max(1).values
        smallest = k.masked_fill(~valid, torch.inf).min(1).values
        fallback = torch.from_numpy(condition >= 500000)
        fallback_count += int(fallback.sum())
        result[first:stop] = torch.where(fallback, (largest + smallest) / 2, fitted)
    return result.numpy(), fallback_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eigen-mean", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    root = Path(__file__).resolve().parent
    smoothwm = root / "rh.smoothwm"
    xyz, faces = fsio.read_geometry(str(smoothwm))
    native = np.fromfile(root / "native_smoothwm_curvature.bin", dtype="<f4")
    started = perf_counter()
    predicted, fallback_count = calculate(torch.from_numpy(xyz.astype(np.float32)),
                                          torch.from_numpy(faces.astype(np.int64)),
                                          args.eigen_mean)
    elapsed = perf_counter() - started
    assert predicted.shape == native.shape
    delta = np.abs(predicted - native)
    different = np.flatnonzero(predicted != native)
    report = {"smoothwm_sha256": hashlib.sha256(smoothwm.read_bytes()).hexdigest(),
              "native_raw_sha256": hashlib.sha256(native.tobytes()).hexdigest(),
              "predicted_raw_sha256": hashlib.sha256(predicted.tobytes()).hexdigest(),
              "vertices": len(native), "eigen_mean": args.eigen_mean,
              "exact_vertices": int(len(native) - len(different)),
              "first_different_vertex": int(different[0]) if len(different) else None,
              "max_abs_error": float(delta.max()),
              "median_abs_error": float(np.median(delta)),
              "p95_abs_error": float(np.percentile(delta, 95)),
              "fallback_vertices": fallback_count,
              "seconds_excluding_io": elapsed,
              "selected": {str(index): {"candidate": float(predicted[index]),
                                       "native": float(native[index])} for index in (0, 57378)}}
    (root / ("rh_smoothwm_full_vnl_eigen.json" if args.eigen_mean
             else "rh_smoothwm_full_vnl.json")).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
