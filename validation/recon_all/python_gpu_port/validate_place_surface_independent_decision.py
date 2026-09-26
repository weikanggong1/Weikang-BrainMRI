"""Independent fixed-input pial dt/reject replay, bounded by requested steps."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import nibabel as nib
import numpy as np


RAM_LH = {
    1: "probe_installed_ram_official_step1_exact",
    2: "probe_installed_ram_official_step2_exact",
    5: "probe_installed_ram_official_step5",
    10: "probe_installed_ram_official_step10",
    12: "probe_installed_ram_official_step12",
    13: "probe_installed_ram_official_step13",
    14: "probe_installed_ram_official_step14",
    20: "probe_installed_ram_official_step20_exact",
    21: "probe_installed_ram_official_step21_exact",
    26: "probe_installed_ram_official_pass0",
    32: "probe_installed_ram_official_pass1",
    36: "probe_installed_ram_official_pass2",
    37: "probe_installed_ram_official_step37_limited",
    41: "probe_installed_ram_official_pass3",
}


RAM_RH = {
    1: "probe_installed_rh_ram_official_step1_exact",
    2: "probe_installed_rh_ram_official_step2_exact",
    41: "probe_installed_rh_ram_official_pass3",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(actual: np.ndarray, reference: np.ndarray) -> dict:
    a = np.asarray(actual, dtype=np.float32)
    b = np.asarray(reference, dtype=np.float32)
    same = a.view(np.uint32) == b.view(np.uint32)
    different = np.flatnonzero(~np.all(same, axis=1))
    return {
        "exact_vertices": int(len(a) - len(different)),
        "total_vertices": len(a),
        "exact_components": int(same.sum()),
        "total_components": int(same.size),
        "first_different_vertices": different[:12].tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--native-log-reference", type=Path, required=True)
    parser.add_argument("--installed-binary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--hemisphere", choices=("lh", "rh"), default="lh")
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--resume-step", type=int, default=0)
    parser.add_argument("--resume-report", type=Path)
    args = parser.parse_args()
    if not 1 <= args.max_steps <= 41 or args.resume_step not in (0, 21, 37):
        parser.error("bounded diagnostic covers steps 1..41, resuming at step 21 or 37")
    if args.resume_step >= args.max_steps or bool(args.resume_report) != bool(args.resume_step):
        parser.error("resume needs a prior independent report and a later maximum step")
    if args.hemisphere == "rh" and args.resume_step:
        parser.error("RH diagnostic currently runs from step 1")

    sys.path.insert(0, str(args.module))
    from fnit.recon_all.place_surface_border import compute_border_values_first_pass
    from fnit.recon_all.place_surface_collision import asynchronous_first_step
    from fnit.recon_all.place_surface_curvature import quadratic_curvature, tangent_basis, two_ring_neighbors
    from fnit.recon_all.place_surface_decision import pial_step_decision
    from fnit.recon_all.place_surface_geometry import surface_ras_to_voxel
    from fnit.recon_all.place_surface_gradient_average import average_signed_gradients
    from fnit.recon_all.place_surface_intensity import intensity_gradient
    from fnit.recon_all.place_surface_normals import initial_vertex_normals
    from fnit.recon_all.place_surface_objective import (
        intensity_error, pial_placement_sse, surface_total_area, tangential_spring_energy,
    )
    from fnit.recon_all.place_surface_repulsion import (
        original_vertex_normals, surface_repulsion_gradient, vertex_buckets,
    )
    from fnit.recon_all.place_surface_rip import rip_outside_label
    from fnit.recon_all.place_surface_smoothing import average_marked_values, _ordered_neighbors
    from fnit.recon_all.place_surface_spring import spring_gradient
    from fnit.recon_all.place_surface_step import unconstrained_step_with_offsets
    from fnit.recon_all.place_surface_volume import prepare_placement_volume

    hemi = args.hemisphere
    white = args.subject / f"surf/{hemi}.white"
    label = args.subject / f"label/{hemi}.cortex+hipamyg.label"
    stats_path = args.subject / f"surf/autodet.gw.stats.{hemi}.dat"
    brain_path = args.subject / "mri/brain.finalsurfs.mgz"
    wm_path = args.subject / "mri/wm.mgz"
    aseg_path = args.subject / "mri/aseg.presurf.mgz"
    xyz, faces, metadata = nib.freesurfer.read_geometry(white, read_metadata=True)
    xyz = xyz.astype(np.float32)
    ripped = np.asarray(rip_outside_label(
        len(xyz), nib.freesurfer.read_label(label)), dtype=np.bool_)
    brain = nib.load(brain_path)
    wm = np.asarray(nib.load(wm_path).dataobj)
    aseg = np.asarray(nib.load(aseg_path).dataobj)
    stats = dict(line.split()[:2] for line in stats_path.read_text().splitlines()
                 if len(line.split()) >= 2)
    volume, bright = prepare_placement_volume(
        np.asarray(brain.dataobj), wm, surface="pial",
        mid_gray=float(stats["MID_GRAY"]),
    )
    placement = volume.copy()
    placement[bright == 130] = 0
    affine = surface_ras_to_voxel(brain.header, metadata)
    thresholds = np.array([float(stats[f"pial_{name}"]) for name in
                           ("inside_hi", "border_hi", "border_low", "outside_low", "outside_hi")])
    normals = initial_vertex_normals(xyz, faces)
    border = compute_border_values_first_pass(
        volume, aseg, xyz, normals, xyz, ripped,
        np.full(len(xyz), -1.0, dtype=np.float32), affine, thresholds,
        hemisphere=hemi, surface="pial",
    )
    values = average_marked_values(border[0], border[4], ripped, faces, 5)
    neighbor_indices, neighbor_valid, _ = _ordered_neighbors(faces, len(xyz))
    ordered = (neighbor_indices, neighbor_valid)
    two_offsets, two_candidates = two_ring_neighbors(
        faces, len(xyz), ordered_neighbors=ordered)
    fixed_normals = original_vertex_normals(xyz, faces)
    original_area = surface_total_area(xyz, faces)

    def objective(current: np.ndarray) -> dict:
        current_normals = initial_vertex_normals(current, faces)
        intensity_sse, rms, count = intensity_error(
            placement, current, values, ripped, affine)
        spring = tangential_spring_energy(
            current, current_normals, faces, ripped,
            ordered_neighbors=ordered)
        area = surface_total_area(current, faces)
        return {
            "sse": pial_placement_sse(intensity_sse, spring, original_area, area),
            "rms": rms,
            "eligible_vertices": count,
            "intensity_sse": intensity_sse,
            "spring_energy": spring,
            "total_area": float(area),
        }

    sigma, n_averages = 2.0, 16

    def gradient(current: np.ndarray, cropped: np.ndarray) -> np.ndarray:
        current_normals = initial_vertex_normals(current, faces)
        intensity = intensity_gradient(
            placement, current, current_normals, ripped, values, border[5],
            affine, brain.header.get_zooms()[:3], weight=0.2, sigma_global=sigma,
        )
        offsets, candidates = vertex_buckets(current, xyz, ripped)
        repulsion = surface_repulsion_gradient(
            current, current_normals, xyz, fixed_normals, ripped,
            offsets, candidates, weight=5.0, cropped=cropped,
        )
        with_repulsion = np.float32(intensity + repulsion)
        averaged = average_signed_gradients(
            with_repulsion, faces, ripped, n_averages, ordered_neighbors=ordered)
        normal = spring_gradient(
            current, current_normals, faces, ripped, weight=0.3,
            direction="normal", ordered_neighbors=ordered)
        basis = tangent_basis(current_normals)
        scalar = quadratic_curvature(
            current, current_normals, basis, ripped, two_offsets, two_candidates)
        with_curvature = np.float32(np.float32(averaged + normal)
                                    + np.float32(scalar[:, None] * current_normals))
        tangent = spring_gradient(
            current, current_normals, faces, ripped, weight=0.3,
            direction="tangent", ordered_neighbors=ordered)
        return np.float32(with_curvature + tangent)

    # The installed log is read only for post-decision comparison below. Neither
    # its dt nor its rejection flag is assigned to the independent state.
    log = args.native_log_reference.read_text()
    reference_dt = {}
    reference_rejected = set()
    reference_pass = {}
    for outer in range(4):
        chunk = log.split(f"Iteration {outer} =========================================", 1)[1].split(
            "maximum number of reductions reached", 1)[0]
        for index, value in re.findall(r"(?m)^(\d{3}): dt: ([0-9.]+)", chunk):
            if float(value) == 0.0:  # Initial state at an outer-pass boundary.
                continue
            reference_dt[int(index)] = float(value)
            reference_pass[int(index)] = outer
        reference_rejected.update(int(index) for index in re.findall(
            r"RMS increased, rejecting step\n(\d{3}): dt:", chunk))
    references = {}
    for step, directory in (RAM_LH if hemi == "lh" else RAM_RH).items():
        if step > args.max_steps:
            continue
        path = args.root / directory / "surf" / f"{hemi}.pial.ram"
        native_xyz, native_faces = nib.freesurfer.read_geometry(path)
        if not np.array_equal(faces, native_faces):
            raise ValueError(f"step {step}: native ordered faces differ")
        references[step] = (path, native_xyz.astype(np.float32))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    initial = objective(xyz)
    last_sse, last_rms = initial["sse"], initial["rms"]
    current, cropped = xyz.copy(), np.zeros(len(xyz), dtype=np.int32)
    dt, reductions, outer_pass = 0.5, 0, 0
    if args.resume_step:
        prior = json.loads(args.resume_report.read_text())
        if prior["steps"][-1]["step"] != args.resume_step:
            raise ValueError("resume report did not reach the requested step")
        if prior["first_mismatch"] is not None and not (
                args.resume_step == 37
                and prior["first_mismatch"]["reason"] == "coordinates"
                and prior["first_mismatch"]["step"] == 37):
            raise ValueError("resume report has an unresolved earlier mismatch")
        state = np.load(args.output_dir / f"{hemi}.step{args.resume_step:02d}.npz")
        current, cropped = state["xyz"], state["cropped"]
        prior_trial = prior["steps"][-1]["trials"][-1]
        last_sse, last_rms = prior_trial["objective"]["sse"], prior_trial["objective"]["rms"]
        dt, reductions = prior_trial["next_dt"], prior_trial["reductions"]
        if compare(current, references[args.resume_step][1])["exact_vertices"] != len(current):
            raise ValueError("resume state differs from installed RAM checkpoint")
        if args.resume_step == 37:
            for pass_index, prior_step in enumerate((26, 32, 36), 1):
                prior_xyz = np.load(args.output_dir / f"{hemi}.step{prior_step:02d}.npz")["xyz"]
                prior_normals = initial_vertex_normals(prior_xyz, faces)
                sigma = 2.0 / (1 << pass_index)
                n_averages = 16 >> pass_index
                border = compute_border_values_first_pass(
                    volume, aseg, prior_xyz, prior_normals, xyz, ripped,
                    values, affine, thresholds, hemisphere=hemi, surface="pial",
                    sigma=sigma,
                )
                values = average_marked_values(border[0], border[4], ripped, faces, 5)
                original_area = surface_total_area(prior_xyz, faces)
            outer_pass = 3
    report = {
        "scope": f"Independent {hemi.upper()} pial dt/reject decisions on fixed fs_sub01; bounded requested steps",
        "decision_inputs": "Python SSE/RMS from frozen MRI, white surface, target values and independently accepted geometry; no native log values influence dt or rejection",
        "native_reference_role": "Post-decision assertion of dt/reject and selected RAM coordinates only",
        "source_rule": "check_tol=0, l_location=0, tol=1e-4, REDUCTION_PCT=0.5, MAX_REDUCTIONS=2; orig_area reset at each MRISpositionSurface entry",
        "sha256": {str(path): digest(path) for path in (
            args.installed_binary, white, label, stats_path, brain_path,
            wm_path, aseg_path, args.native_log_reference,
            *(path for path, _ in references.values()),
        )},
        "initial_objective": initial,
        "resume_step": args.resume_step,
        "resume_state_sha256": digest(args.output_dir / f"{hemi}.step{args.resume_step:02d}.npz") if args.resume_step else None,
        "resume_report_sha256": digest(args.resume_report) if args.resume_report else None,
        "resume_prior_mismatch": prior["first_mismatch"] if args.resume_step else None,
        "steps": [],
        "first_mismatch": None,
    }
    for step in range(args.resume_step + 1, args.max_steps + 1):
        force = gradient(current, cropped)
        momentum = force.copy()
        stale_trial = None
        trials = []
        accepted = None
        for attempt in range(3):
            used_dt = dt
            proposal, displacement = unconstrained_step_with_offsets(
                current, momentum, ripped, dt=used_dt)
            candidate, order = asynchronous_first_step(
                current, faces, proposal, ripped, fast=True,
                offsets=displacement, accepted_offsets=momentum,
                stale_mht_trial=stale_trial, ordered_neighbors=ordered)
            blocked = np.any(proposal != current, axis=1) & np.all(
                candidate == current, axis=1)
            trial_cropped = np.where(
                ripped, cropped, np.where(blocked, cropped + 1, 0)).astype(np.int32)
            result = objective(candidate)
            next_dt, reductions, reduced, rejected, stop = pial_step_decision(
                last_sse, last_rms, result["sse"], result["rms"], dt, reductions)
            trials.append({
                "attempt": attempt + 1, "dt": used_dt, "objective": result,
                "reduced": reduced, "rejected": rejected,
                "next_dt": next_dt, "reductions": reductions,
                "processed_vertices": len(order),
                "cropped_vertices": int(np.count_nonzero(trial_cropped)),
            })
            dt = next_dt
            cropped = trial_cropped
            if rejected:
                stale_trial = candidate
                if stop:
                    break
                continue
            accepted = candidate
            last_sse, last_rms = result["sse"], result["rms"]
            break
        if accepted is None:
            report["first_mismatch"] = {"step": step, "reason": "all trials rejected"}
            break
        current = accepted
        np.savez(args.output_dir / f"{hemi}.step{step:02d}.npz", xyz=current, cropped=cropped)
        comparison = compare(current, references[step][1]) if step in references else None
        observed_reject = any(item["rejected"] for item in trials)
        schedule_match = (trials[-1]["dt"] == reference_dt[step]
                          and observed_reject == (step in reference_rejected)
                          and outer_pass == reference_pass[step])
        entry = {
            "step": step, "outer_pass": outer_pass, "trials": trials,
            "native_reference_dt": reference_dt[step],
            "native_reference_rejected": step in reference_rejected,
            "native_reference_pass": reference_pass[step],
            "schedule_match": schedule_match,
            "coordinate_comparison": comparison,
        }
        report["steps"].append(entry)
        if not schedule_match or comparison is not None and (
                comparison["exact_components"] != comparison["total_components"]):
            report["first_mismatch"] = {
                "step": step,
                "reason": "decision" if not schedule_match else "coordinates",
                "comparison": comparison,
            }
            break
        if stop:
            if outer_pass == 3:
                break
            outer_pass += 1
            sigma = 2.0 / (1 << outer_pass)
            n_averages = 16 >> outer_pass
            current_normals = initial_vertex_normals(current, faces)
            border = compute_border_values_first_pass(
                volume, aseg, current, current_normals, xyz, ripped,
                values, affine, thresholds, hemisphere=hemi, surface="pial",
                sigma=sigma,
            )
            values = average_marked_values(border[0], border[4], ripped, faces, 5)
            original_area = surface_total_area(current, faces)
            next_initial = objective(current)
            last_sse, last_rms = next_initial["sse"], next_initial["rms"]
            dt, reductions = 0.5, 0
            report.setdefault("outer_pass_starts", []).append({
                "outer_pass": outer_pass,
                "after_step": step,
                "sigma": sigma,
                "gradient_averages": n_averages,
                "orig_area": float(original_area),
                "initial_objective": next_initial,
            })
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "steps": len(report["steps"]),
        "last_step": report["steps"][-1]["step"] if report["steps"] else None,
        "first_mismatch": report["first_mismatch"],
        "last_coordinate_comparison": next(
            (entry["coordinate_comparison"] for entry in reversed(report["steps"])
             if entry["coordinate_comparison"] is not None), None),
    }, indent=2))


if __name__ == "__main__":
    main()
