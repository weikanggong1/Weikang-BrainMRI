"""Compare raw-input white rip passes with the pinned source probe."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.freesurfer import read_geometry

STAGE = np.dtype([("flags", "<i4", 3), ("val", "<f4")])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--hemisphere", choices=("lh", "rh"), default="lh")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--require-exact", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(args.module))
    from fnit.recon_all.place_surface_geometry import surface_ras_to_voxel
    from fnit.recon_all.place_surface_normals import initial_vertex_normals
    from fnit.recon_all.place_surface_rip import rip_white_preaparc_pass
    from fnit.recon_all.place_surface_smoothing import average_vertex_positions
    from fnit.recon_all.place_surface_volume import prepare_placement_volume

    brain = nib.load(args.subject / "mri/brain.finalsurfs.mgz")
    wm = nib.load(args.subject / "mri/wm.mgz")
    seg = nib.load(args.subject / "mri/aseg.presurf.mgz")
    volume, _ = prepare_placement_volume(
        np.asarray(brain.dataobj), np.asarray(wm.dataobj), surface="white", mid_gray=0,
    )
    vertices, faces, metadata = read_geometry(args.subject / f"surf/{args.hemisphere}.orig", read_metadata=True)
    xyz = average_vertex_positions(vertices, faces, 5)
    normals = initial_vertex_normals(xyz, faces)
    affine = surface_ras_to_voxel(seg.header, metadata)
    ripped = values = None
    report = {"hemisphere": args.hemisphere, "reference": "instrumented pinned FreeSurfer 8.2 source", "passes": []}
    for call in (1, 2):
        start = time.perf_counter()
        ripped, values = rip_white_preaparc_pass(
            xyz, normals, faces, np.asarray(seg.dataobj), volume, affine,
            hemisphere=args.hemisphere, ripped=ripped, values=values,
        )
        expected = np.fromfile(args.probe / f"{args.hemisphere}.rip_probe.call{call}.after_bg", dtype=STAGE)
        result = {}
        for name, actual, reference in (
            ("ripped", ripped, expected["flags"][:, 0]),
            ("value", values, expected["val"]),
        ):
            mismatch = np.flatnonzero(actual != reference)
            result[name] = {
                "exact_elements": int(len(actual) - len(mismatch)),
                "total_elements": len(actual),
                "first_mismatch_vertices": mismatch[:20].tolist(),
            }
        report["passes"].append({
            "call": call, "seconds_including_jit": time.perf_counter() - start,
            "ripped_vertices": int(ripped.sum()), "comparisons": result,
        })
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if args.require_exact and any(
        item["exact_elements"] != item["total_elements"]
        for stage in report["passes"] for item in stage["comparisons"].values()
    ):
        raise SystemExit("white rip pass differs from pinned source")


if __name__ == "__main__":
    main()
