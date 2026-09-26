"""Compare two independent RH curvature-fit vertices with installed checkpoints."""

import hashlib
import json
from pathlib import Path

import nibabel.freesurfer.io as fsio
import numpy as np
import torch

from fnit.recon_all.mris_register_nonlinear import sphere_vertex_normals, tangent_basis
from fnit.recon_all.mris_register_smoothwm import _three_hop_neighbors


def compare(candidate, native):
    a, b = np.asarray(candidate), np.asarray(native, dtype=np.asarray(candidate).dtype)
    if a.shape != b.shape:
        return {"candidate_shape": a.shape, "native_shape": b.shape, "exact": False}
    different = np.flatnonzero(a.ravel() != b.ravel())
    first = int(different[0]) if len(different) else None
    return {"shape": a.shape, "exact_elements": int(a.size - len(different)),
            "first_different_flat_index": first,
            "candidate_at_first_difference": float(a.ravel()[first]) if first is not None else None,
            "native_at_first_difference": float(b.ravel()[first]) if first is not None else None,
            "max_abs_error": float(np.max(np.abs(a - b))) if a.size else 0.0}


def main():
    torch.set_num_threads(4)
    root = Path(__file__).resolve().parent
    xyz, faces = fsio.read_geometry(str(root / "rh.smoothwm"))
    xyz = torch.from_numpy(xyz.astype(np.float32))
    faces = torch.from_numpy(faces.astype(np.int64))
    normals = sphere_vertex_normals(xyz, faces)
    e1, e2 = tangent_basis(normals)
    neighbors, active = _three_hop_neighbors(faces, len(xyz))
    native = json.loads((root / "native_smoothwm_fit.json").read_text())
    raw_native = np.fromfile(root / "native_smoothwm_curvature.bin", dtype="<f4")
    result = {"smoothwm_sha256": hashlib.sha256((root / "rh.smoothwm").read_bytes()).hexdigest(),
              "native_fit_sha256": hashlib.sha256((root / "native_smoothwm_fit.json").read_bytes()).hexdigest(),
              "vertices": {}}
    for index in (0, 57378):
        ref = native[str(index)]
        ids = neighbors[index, active[index]]
        delta = xyz[ids] - xyz[index]
        u = (delta * e1[index]).sum(1)
        v = (delta * e2[index]).sum(1)
        z = (delta * normals[index]).sum(1)
        rsq = u * u + v * v
        valid = rsq > 1e-12
        design = torch.stack((u * u, 2 * u * v, v * v), dim=1) * valid[:, None]
        height = z * valid
        gram = torch.zeros(3, 3)
        rhs = torch.zeros(3, 1)
        for slot in range(len(ids)):
            row = design[slot]
            gram += row[:, None] * row[None, :]
            rhs[:, 0] += row * height[slot]
        left, singular, right = torch.linalg.svd(gram)
        inverse_singular = torch.where(singular >= 1e-4 * singular[:1],
                                       singular.clamp_min(1e-30).reciprocal(), 0)
        inverse = right.T @ (inverse_singular[:, None] * left.T)
        coeff = right.T @ (inverse_singular[:, None] * (left.T @ rhs))
        candidate = {"position": xyz[index].numpy(), "normal": normals[index].numpy(),
                     "tangent_e1": e1[index].numpy(), "tangent_e2": e2[index].numpy(),
                     "neighbors": ids.numpy(), "design": design.numpy(),
                     "height": height[:, None].numpy(), "gram": gram.numpy(),
                     "rhs": rhs.numpy(), "inverse": inverse.numpy(),
                     "coefficients": coeff.numpy()}
        stages = {name: compare(value, ref[name]) for name, value in candidate.items()}
        stages["raw_curvature"] = compare(np.array([float(coeff[0, 0] + coeff[2, 0])], np.float32),
                                           np.array([float(raw_native[index])], np.float32))
        result["vertices"][str(index)] = {"stages": stages,
                                           "native_condition_number": ref["condition_number"],
                                           "candidate_condition_number": float(singular[0] / singular[-1]),
                                           "candidate_gram": gram.tolist(),
                                           "native_gram": ref["gram"]}
    (root / "rh_smoothwm_fit_comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
