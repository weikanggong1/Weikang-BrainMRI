"""Validate the first placement target search from original files.

The expected outputs are from an instrumented pinned-source binary. The
instrumentation is a validator only; none of the reconstruction functions
read the checkpoint files.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.freesurfer import read_geometry, read_label


STATE = np.dtype([("floats", "<f4", 16), ("flags", "<i4", 2)])


def compare(actual: np.ndarray, reference: np.ndarray) -> dict:
    if actual.shape != reference.shape:
        raise ValueError(f"shape mismatch: {actual.shape} versus {reference.shape}")
    difference = np.abs(actual.astype(np.float64) - reference.astype(np.float64))
    return {
        "exact_elements": int(np.count_nonzero(actual == reference)),
        "total_elements": int(actual.size),
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
    from fnit.recon_all.place_surface_normals import initial_vertex_normals
    from fnit.recon_all.place_surface_rip import rip_outside_label, rip_white_preaparc_pass
    from fnit.recon_all.place_surface_smoothing import average_vertex_positions
    from fnit.recon_all.place_surface_volume import prepare_placement_volume

    brain = nib.load(args.subject / "mri/brain.finalsurfs.mgz")
    source = np.asarray(brain.dataobj)
    wm = np.asarray(nib.load(args.subject / "mri/wm.mgz").dataobj)
    segmentation_image = nib.load(args.subject / "mri/aseg.presurf.mgz")
    segmentation = np.asarray(segmentation_image.dataobj)
    report: dict = {"reference": "instrumented pinned FreeSurfer 8.2 source", "cases": {}}
    for hemisphere in ("lh", "rh"):
        stats = dict(
            line.split()[:2]
            for line in (args.subject / f"surf/autodet.gw.stats.{hemisphere}.dat").read_text().splitlines()
            if len(line.split()) >= 2
        )
        for surface in ("white", "pial"):
            start = time.perf_counter()
            prefix = f"{hemisphere}{'.pial' if surface == 'pial' else ''}.border_probe"
            input_surface = args.subject / "surf" / f"{hemisphere}.{'orig' if surface == 'white' else 'white'}"
            vertices, faces, metadata = read_geometry(input_surface, read_metadata=True)
            xyz = average_vertex_positions(vertices, faces, 5 if surface == "white" else 0)
            normals = initial_vertex_normals(xyz, faces)
            volume, labels = prepare_placement_volume(source, wm, surface=surface, mid_gray=float(stats["MID_GRAY"]))
            before = np.fromfile(args.probe / f"{prefix}.before", dtype=STATE)
            checked = {
                "initial_xyz": compare(xyz, before["floats"][:, :3]),
                "initial_normals": compare(normals, before["floats"][:, 3:6]),
                "volume_voxels": compare(volume, np.asarray(nib.load(args.probe / f"{prefix}.volume.mgz").dataobj)),
                "bright_labels": compare(labels, np.asarray(nib.load(args.probe / f"{prefix}.bright_labels.mgz").dataobj)),
                "surface_ras_to_voxel": compare(
                    surface_ras_to_voxel(brain.header, metadata),
                    np.fromfile(args.probe / f"{prefix}.transform", dtype="<f4").reshape(4, 4),
                ),
            }
            if surface == "pial":
                label = read_label(args.subject / "label" / f"{hemisphere}.cortex+hipamyg.label")
                ripped = rip_outside_label(len(xyz), label)
                values = np.full(len(xyz), -1, dtype=np.float32)
            else:
                ripped = values = None
                rip_affine = surface_ras_to_voxel(segmentation_image.header, metadata)
                for _ in range(2):
                    ripped, values = rip_white_preaparc_pass(
                        xyz, normals, faces, segmentation, volume, rip_affine,
                        hemisphere=hemisphere, ripped=ripped, values=values,
                    )
            checked["ripped"] = compare(ripped, before["flags"][:, 0])
            checked["initial_value"] = compare(values, before["floats"][:, 9])
            after = np.fromfile(args.probe / f"{prefix}.after", dtype=STATE)
            thresholds = np.array([
                float(stats[f"{surface}_{name}"])
                for name in ("inside_hi", "border_hi", "border_low", "outside_low", "outside_hi")
            ])
            border = compute_border_values_first_pass(
                volume, segmentation, xyz, normals, xyz, ripped, values,
                surface_ras_to_voxel(brain.header, metadata), thresholds,
                hemisphere=hemisphere, surface=surface,
            )
            for name, actual, expected in zip(
                ("border_value", "border_distance", "border_gradient", "border_target_xyz", "border_mark", "border_sigma"),
                border,
                (after["floats"][:, 9], after["floats"][:, 10], after["floats"][:, 11],
                 after["floats"][:, 12:15], after["flags"][:, 1], after["floats"][:, 15]),
            ):
                checked[name] = compare(actual, expected)
            report["cases"][f"{hemisphere}_{surface}"] = {
                "vertices": len(xyz), "seconds_including_io_and_jit": time.perf_counter() - start,
                "comparisons": checked,
            }
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if args.require_exact and any(
        result["exact_elements"] != result["total_elements"]
        for case in report["cases"].values() for result in case["comparisons"].values()
    ):
        raise SystemExit("raw-input first-pass target differs from pinned source")


if __name__ == "__main__":
    main()
