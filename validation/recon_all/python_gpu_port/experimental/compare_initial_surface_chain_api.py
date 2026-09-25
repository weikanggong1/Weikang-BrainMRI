"""Compare a connected Python surface-chain output with a completed subject."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np


MODULES = ("pretess_python", "tessellate_gpu", "extract_main_component_python",
           "smooth_surface_python", "inflate_python", "sphere_quick_python",
           "initial_surface_chain")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def surface_core(path: Path) -> tuple[int, int, bytes, bytes, bytes]:
    data = path.read_bytes()
    if data[:3] != b"\xff\xff\xfe":
        raise ValueError(f"not a triangular FreeSurfer surface: {path}")
    start = data.index(b"\n\n", 3) + 2
    nvertices = int.from_bytes(data[start:start + 4], "big")
    nfaces = int.from_bytes(data[start + 4:start + 8], "big")
    xyz_start = start + 8
    face_start = xyz_start + 12 * nvertices
    face_end = face_start + 12 * nfaces
    geom_start = data.index(b"valid = ", face_end)
    geom_end = data.index(b"\n", data.index(b"cras   = ", geom_start)) + 1
    return nvertices, nfaces, data[xyz_start:face_start], data[face_start:face_end], data[geom_start:geom_end]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--native-pretess-dir", type=Path, required=True)
    parser.add_argument("--run-report", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    stages = json.loads(args.run_report.read_text())
    comparisons = {}
    all_exact = True
    for hemi, label in (("lh", 255), ("rh", 127)):
        actual_pretess = nib.load(str(args.output_root / "mri" / f"filled-pretess{label}.mgz"))
        native_pretess = nib.load(str(args.native_pretess_dir / f"filled-pretess{label}.mgz"))
        pretess_exact = bool(np.array_equal(np.asanyarray(actual_pretess.dataobj),
                                          np.asanyarray(native_pretess.dataobj))
                             and np.array_equal(actual_pretess.affine, native_pretess.affine))
        surfaces = {}
        for name in ("orig", "smoothwm", "inflated", "qsphere"):
            actual = args.output_root / "surf" / f"{hemi}.{name}.nofix"
            native = args.subject / "surf" / f"{hemi}.{name}.nofix"
            a, b = surface_core(actual), surface_core(native)
            row = {"vertices": a[0], "faces": a[1],
                   "ordered_vertex_bytes_exact": a[0] == b[0] and a[2] == b[2],
                   "ordered_face_bytes_exact": a[1] == b[1] and a[3] == b[3],
                   "volume_geometry_bytes_exact": a[4] == b[4],
                   "python_sha256": sha256(actual), "native_sha256": sha256(native)}
            surfaces[name] = row
            all_exact &= all(row[key] for key in ("ordered_vertex_bytes_exact",
                                               "ordered_face_bytes_exact",
                                               "volume_geometry_bytes_exact"))
        comparisons[hemi] = {"pretess_voxels_and_affine_exact": pretess_exact,
                             "surfaces": surfaces}
        all_exact &= pretess_exact
    result = {"subject": args.subject.name, "all_exact": bool(all_exact),
              "input_filled_sha256": sha256(args.subject / "mri" / "filled.mgz"),
              "input_norm_sha256": sha256(args.subject / "mri" / "norm.mgz"),
              "source_sha256": {name: sha256(args.source / "src" / "fnit" /
                                              "recon_all" / f"{name}.py") for name in MODULES},
              "stages": stages["hemispheres"], "comparisons": comparisons}
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"all_exact": bool(all_exact), "report": str(args.report)}))
    if not all_exact:
        raise AssertionError("connected surface-chain parity failed")


if __name__ == "__main__":
    main()
