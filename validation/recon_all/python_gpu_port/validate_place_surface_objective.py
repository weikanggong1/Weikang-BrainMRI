"""Check LH pial placement objectives from original MRI and surface snapshots."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import nibabel as nib
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=Path, required=True)
    parser.add_argument("--source-step", type=Path, required=True)
    parser.add_argument("--source-second", type=Path)
    parser.add_argument("--require-exact", action="store_true")
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--probe-input", type=Path)
    parser.add_argument("--native-log", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.module))
    from fnit.recon_all.place_surface_border import compute_border_values_first_pass
    from fnit.recon_all.place_surface_geometry import surface_ras_to_voxel
    from fnit.recon_all.place_surface_normals import initial_vertex_normals
    from fnit.recon_all.place_surface_objective import (
        intensity_error, tangential_spring_energy, surface_total_area, pial_placement_sse,
    )
    from fnit.recon_all.place_surface_rip import rip_outside_label
    from fnit.recon_all.place_surface_smoothing import average_marked_values
    from fnit.recon_all.place_surface_volume import prepare_placement_volume

    subject = args.subject
    brain = nib.load(subject / "mri/brain.finalsurfs.mgz")
    wm = np.asarray(nib.load(subject / "mri/wm.mgz").dataobj)
    segmentation = np.asarray(nib.load(subject / "mri/aseg.presurf.mgz").dataobj)
    xyz, faces, metadata = nib.freesurfer.read_geometry(subject / "surf/lh.white", read_metadata=True)
    xyz = xyz.astype(np.float32)
    normals = initial_vertex_normals(xyz, faces)
    ripped = rip_outside_label(len(xyz), nib.freesurfer.read_label(subject / "label/lh.cortex+hipamyg.label"))
    stats = dict(
        line.split()[:2]
        for line in (subject / "surf/autodet.gw.stats.lh.dat").read_text().splitlines()
        if len(line.split()) >= 2
    )
    volume, bright = prepare_placement_volume(
        np.asarray(brain.dataobj), wm, surface="pial", mid_gray=float(stats["MID_GRAY"]),
    )
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
    native = {}
    for match in re.finditer(
        r"PY_OBJ_REF (initial|step1|step2|step) rms=(\S+) sse=(\S+)"
        r"(?: orig_area=(\S+) total_area=(\S+))?",
        args.native_log.read_text(),
    ):
        label = "step1" if match[1] == "step" else match[1]
        native[label] = {"rms": float(match[2]), "full_sse": float(match[3])}
        if match[4] is not None:
            native[label]["orig_area"] = float(match[4])
            native[label]["total_area"] = float(match[5])
    snapshots = {"initial": xyz}
    paths = {"step1": args.source_step}
    if args.source_second:
        paths["step2"] = args.source_second
    for label, path in paths.items():
        vertices, step_faces = nib.freesurfer.read_geometry(path)
        if not np.array_equal(faces, step_faces):
            raise SystemExit(f"{label} ordered faces differ from raw LH white")
        snapshots[label] = vertices.astype(np.float32)
    if set(snapshots) != set(native):
        raise SystemExit(f"native objective labels {sorted(native)} differ from snapshots {sorted(snapshots)}")
    orig_area = surface_total_area(xyz, faces)
    checkpoints = {}
    for label, vertices in snapshots.items():
        vertex_normals = normals if label == "initial" else initial_vertex_normals(vertices, faces)
        intensity_sse, rms, count = intensity_error(placement, vertices, values, ripped, affine)
        spring = tangential_spring_energy(vertices, vertex_normals, faces, ripped)
        total_area = surface_total_area(vertices, faces)
        full_sse = pial_placement_sse(intensity_sse, spring, orig_area, total_area)
        checkpoint = {
            "intensity_sse": intensity_sse, "intensity_rms": rms, "eligible_vertices": count,
            "tangential_spring_energy": spring, "orig_area": float(orig_area),
            "total_area": float(total_area), "full_sse": full_sse,
            "rms_abs_error": abs(rms - native[label]["rms"]),
            "full_sse_abs_error": abs(full_sse - native[label]["full_sse"]),
        }
        if "total_area" in native[label]:
            checkpoint["native_total_area_abs_error"] = abs(total_area - native[label]["total_area"])
        checkpoints[label] = checkpoint
    report = {
        "reference": "pinned FreeSurfer 8.2 copied-source LH pial full-precision objective probe",
        "inputs": "original MRI, WM, aseg, LH white, label and frozen autodet stats",
        "target_values_match_probe": None,
        "checkpoints": checkpoints,
        "native": native,
        "accepted_steps_by_fixed_native_rule": {
            label: bool(
                100 * (native[previous]["full_sse"] - native[label]["full_sse"])
                / native[previous]["full_sse"] >= 0.0001
                and native[label]["rms"] <= native[previous]["rms"] - 0.05
            )
            for previous, label in (("initial", "step1"), ("step1", "step2"))
            if previous in native and label in native
        },
    }
    if args.probe_input:
        source_values = np.fromfile(args.probe_input, dtype="<f4").reshape(-1, 2)[:, 0]
        report["target_values_match_probe"] = bool(np.array_equal(values, source_values))
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if args.require_exact:
        if any(entry["rms_abs_error"] > 1e-12 for entry in checkpoints.values()):
            raise SystemExit("Python intensity RMS differs from pinned source")
        if any(entry["full_sse_abs_error"] > 1e-6 for entry in checkpoints.values()):
            raise SystemExit("Python complete SSE differs from pinned source")
        if any(entry.get("native_total_area_abs_error", 0) for entry in checkpoints.values()):
            raise SystemExit("Python total surface area differs from pinned source")
        if args.probe_input and not report["target_values_match_probe"]:
            raise SystemExit("Python target values differ from pinned source")



if __name__ == "__main__":
    main()
