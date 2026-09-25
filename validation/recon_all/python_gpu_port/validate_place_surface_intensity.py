"""Compare raw-input first pial intensity gradient with a pinned-source probe."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np


STATE = np.dtype([("floats", "<f4", 9), ("flags", "<i4", 3)])


def compare(actual: np.ndarray, expected: np.ndarray) -> dict:
    difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    return {
        "exact_elements": int(np.count_nonzero(actual == expected)),
        "total_elements": int(expected.size),
        "max_abs_error": float(difference.max(initial=0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--require-exact", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(args.module))
    from fnit.recon_all.place_surface_border import compute_border_values_first_pass
    from fnit.recon_all.place_surface_geometry import surface_ras_to_voxel
    from fnit.recon_all.place_surface_intensity import intensity_gradient
    from fnit.recon_all.place_surface_normals import initial_vertex_normals
    from fnit.recon_all.place_surface_rip import rip_outside_label
    from fnit.recon_all.place_surface_smoothing import average_marked_values
    from fnit.recon_all.place_surface_volume import prepare_placement_volume

    start = time.perf_counter()
    brain = nib.load(args.subject / "mri/brain.finalsurfs.mgz")
    wm = np.asarray(nib.load(args.subject / "mri/wm.mgz").dataobj)
    segmentation = np.asarray(nib.load(args.subject / "mri/aseg.presurf.mgz").dataobj)
    xyz, faces, metadata = nib.freesurfer.read_geometry(args.subject / "surf/lh.white", read_metadata=True)
    xyz = xyz.astype(np.float32)
    normals = initial_vertex_normals(xyz, faces)
    ripped = rip_outside_label(len(xyz), nib.freesurfer.read_label(args.subject / "label/lh.cortex+hipamyg.label"))
    stats = dict(
        line.split()[:2]
        for line in (args.subject / "surf/autodet.gw.stats.lh.dat").read_text().splitlines()
        if len(line.split()) >= 2
    )
    volume, bright = prepare_placement_volume(np.asarray(brain.dataobj), wm, surface="pial", mid_gray=float(stats["MID_GRAY"]))
    placement = volume.copy()
    placement[bright == 130] = 0
    affine = surface_ras_to_voxel(brain.header, metadata)
    thresholds = np.array([
        float(stats[f"pial_{name}"])
        for name in ("inside_hi", "border_hi", "border_low", "outside_low", "outside_hi")
    ])
    border = compute_border_values_first_pass(
        volume, segmentation, xyz, normals, xyz, ripped,
        np.full(len(xyz), -1.0, dtype=np.float32), affine, thresholds,
        hemisphere="lh", surface="pial",
    )
    values = average_marked_values(border[0], border[4], ripped, faces, 5)
    sigmas = border[5]
    candidate = intensity_gradient(
        placement, xyz, normals, ripped, values, sigmas,
        affine, brain.header.get_zooms()[:3],
        weight=0.2, sigma_global=2.0,
    )

    before = np.fromfile(args.probe / "lh.gradient.clear", dtype=STATE)
    after = np.fromfile(args.probe / "lh.gradient.intensity", dtype=STATE)
    intensity_input = np.fromfile(args.probe / "lh.gradient.intensity_input", dtype="<f4").reshape(-1, 2)
    checked = {
        "initial_xyz": compare(xyz, before["floats"][:, :3]),
        "initial_normals": compare(normals, before["floats"][:, 3:6]),
        "ripped": compare(ripped, before["flags"][:, 0]),
        "placement_volume": compare(placement, np.asarray(nib.load(args.probe / "lh.gradient.ps.mgz").dataobj)),
        "target_values": compare(values, intensity_input[:, 0]),
        "vertex_sigma": compare(sigmas, intensity_input[:, 1]),
        "intensity_gradient": compare(candidate, after["floats"][:, 6:9]),
    }
    report = {
        "reference": "copied pinned FreeSurfer 8.2 first pial step source probe",
        "comparisons": checked,
        "seconds_including_io_and_jit": time.perf_counter() - start,
    }
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if args.require_exact and any(item["exact_elements"] != item["total_elements"] for item in checked.values()):
        raise SystemExit("intensity gradient differs from pinned source")


if __name__ == "__main__":
    main()
