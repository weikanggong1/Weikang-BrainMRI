"""Replay LH pial placement checkpoints from original MRI inputs against pinned source."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np


STATE = np.dtype([("floats", "<f4", 9), ("flags", "<i4", 3)])
STAGES = (
    "intensity", "surface_repulsion", "pre_normal_spring",
    "normal_spring", "curvature", "tangential_spring",
)


def comparison(actual: np.ndarray, expected: np.ndarray) -> dict:
    exact = actual == expected
    delta = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    mismatched = np.flatnonzero(~np.all(exact, axis=1)) if actual.ndim == 2 else np.flatnonzero(~exact)
    return {
        "exact_elements": int(np.count_nonzero(exact)),
        "total_elements": int(expected.size),
        "max_abs_error": float(delta.max(initial=0)),
        "first_mismatched_vertices": mismatched[:10].tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=Path, required=True)
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--gradient-probes", type=Path, nargs="+")
    parser.add_argument("--gradient-probe-root", type=Path)
    parser.add_argument("--source-steps", type=Path, nargs="+", required=True)
    parser.add_argument("--native-log", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-second", type=Path)
    parser.add_argument("--output-third", type=Path)
    parser.add_argument("--require-exact", action="store_true")
    parser.add_argument("--fast-collision", action="store_true")
    parser.add_argument("--gradient-averages", type=int, default=2)
    parser.add_argument("--sigma", type=float, default=2.0)
    parser.add_argument("--source-files", type=Path, nargs="*", default=[])
    parser.add_argument("--installed-checkpoint", action="append", default=[], metavar="STEP:PATH")
    parser.add_argument("--installed-log", type=Path)
    args = parser.parse_args()
    step_count = len(args.source_steps)
    installed_checkpoints = {
        int(number): Path(path)
        for number, path in (spec.split(":", 1) for spec in args.installed_checkpoint)
    }
    if any(index < 1 or index > step_count for index in installed_checkpoints):
        parser.error("installed checkpoint must belong to the replayed step range")
    if (args.gradient_probes is None) == (args.gradient_probe_root is None):
        parser.error("provide either --gradient-probes or --gradient-probe-root")
    if args.gradient_probes is not None and len(args.gradient_probes) != step_count:
        parser.error("gradient probe count must equal source step count")
    sys.path.insert(0, str(args.module))
    from fnit.recon_all.place_surface_border import compute_border_values_first_pass
    from fnit.recon_all.place_surface_collision import asynchronous_first_step
    from fnit.recon_all.place_surface_curvature import quadratic_curvature, tangent_basis, two_ring_neighbors
    from fnit.recon_all.place_surface_geometry import surface_ras_to_voxel
    from fnit.recon_all.place_surface_gradient_average import average_signed_gradients
    from fnit.recon_all.place_surface_intensity import intensity_gradient
    from fnit.recon_all.place_surface_normals import initial_vertex_normals
    from fnit.recon_all.place_surface_objective import (
        intensity_error, tangential_spring_energy, surface_total_area, pial_placement_sse,
    )
    from fnit.recon_all.place_surface_repulsion import original_vertex_normals, surface_repulsion_gradient, vertex_buckets
    from fnit.recon_all.place_surface_rip import rip_outside_label
    from fnit.recon_all.place_surface_smoothing import average_marked_values, _ordered_neighbors
    from fnit.recon_all.place_surface_spring import spring_gradient
    from fnit.recon_all.place_surface_step import unconstrained_step_with_offsets
    from fnit.recon_all.place_surface_volume import prepare_placement_volume

    started = time.perf_counter()
    subject = args.subject
    brain = nib.load(subject / "mri/brain.finalsurfs.mgz")
    wm = np.asarray(nib.load(subject / "mri/wm.mgz").dataobj)
    segmentation = np.asarray(nib.load(subject / "mri/aseg.presurf.mgz").dataobj)
    original, faces, metadata = nib.freesurfer.read_geometry(subject / "surf/lh.white", read_metadata=True)
    original = original.astype(np.float32)
    ripped = np.asarray(rip_outside_label(
        len(original), nib.freesurfer.read_label(subject / "label/lh.cortex+hipamyg.label"),
    ), dtype=np.bool_)
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
    initial_normals = initial_vertex_normals(original, faces)
    border = compute_border_values_first_pass(
        volume, segmentation, original, initial_normals, original, ripped,
        np.full(len(original), -1.0, dtype=np.float32), affine, thresholds,
        hemisphere="lh", surface="pial",
    )
    values = average_marked_values(border[0], border[4], ripped, faces, 5)
    fixed_normals = original_vertex_normals(original, faces)
    neighbor_indices, neighbor_valid, _ = _ordered_neighbors(faces, len(original))
    ordered = (neighbor_indices, neighbor_valid)
    two_offsets, two_candidates = two_ring_neighbors(faces, len(original), ordered_neighbors=ordered)
    orig_area = surface_total_area(original, faces)
    native_log = args.native_log.read_text()
    native = {
        label: {"rms": float(rms), "sse": float(sse), "area": float(area)}
        for label, rms, sse, area in re.findall(
            r"PY_OBJ_REF (initial|step\d+) rms=(\S+) sse=(\S+) orig_area=\S+ total_area=(\S+)",
            native_log,
        )
    }
    native_command = next(
        line.strip() for line in native_log.splitlines()
        if line.startswith("/") and "mris_place_surface" in line
    )
    expected_native = {"initial"} | {f"step{index}" for index in range(1, step_count + 1)}
    if not expected_native.issubset(native):
        raise SystemExit(f"missing native objectives: {sorted(expected_native - set(native))}")

    installed_objectives = {}
    if args.installed_log:
        for number, dt_printed, sse_printed, rms_printed in re.findall(
            r"(?m)^([0-9]{3}): dt: ([0-9.]+), sse=([0-9.]+), rms=([0-9.]+)",
            args.installed_log.read_text(),
        ):
            index = int(number)
            if index in installed_objectives or index > step_count:
                break
            installed_objectives[index] = {
                "dt": float(dt_printed), "sse": float(sse_printed),
                "rms": float(rms_printed),
            }
    report = {
        "reference": (
            f"pinned FreeSurfer 8.2 copied-source {step_count}-iteration LH pial probe; "
            f"n_averages={args.gradient_averages}, sigma={args.sigma:g}"
        ),
        "inputs": "original T1, WM, aseg, LH white, label and frozen autodet stats",
        "native_source_inputs_used_for_python_inference": False,
        "native_reference_command": native_command,
        "native_reference_command_sha256": hashlib.sha256(native_command.encode()).hexdigest(),
        "native_source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in args.source_files
        },
        "source_parameters": {
            "n_averages": args.gradient_averages,
            "sigma": args.sigma,
            "target_value_averages": 5,
        },
        "fast_collision": args.fast_collision,
        "native_objectives": native,
        "installed_reference_class": (
            "installed FreeSurfer 8.2.0-1 RAM first-outer-pass checkpoint; later passes and cleanup skipped"
            if installed_checkpoints else None
        ),
        "installed_checkpoint_paths": {str(index): str(path) for index, path in installed_checkpoints.items()},
        "installed_objectives_printed": installed_objectives,
        "steps": [],
    }
    current = original.copy()
    cropped = np.zeros(len(original), dtype=np.int32)

    def objective(label: str, vertices: np.ndarray, normals: np.ndarray) -> dict:
        intensity_sse, rms, count = intensity_error(placement, vertices, values, ripped, affine)
        spring = tangential_spring_energy(vertices, normals, faces, ripped, ordered_neighbors=ordered)
        area = surface_total_area(vertices, faces)
        sse = pial_placement_sse(intensity_sse, spring, orig_area, area)
        return {
            "eligible_vertices": count, "area": float(area), "rms": rms, "sse": sse,
            "native_area_abs_error": abs(area - native[label]["area"]),
            "native_rms_abs_error": abs(rms - native[label]["rms"]),
            "native_sse_abs_error": abs(sse - native[label]["sse"]),
        }

    report["initial_objective"] = objective("initial", current, initial_normals)
    native_rejections = {
        int(index): {"rms_printed": float(rms), "sse_printed": float(sse), "new_dt": float(new_dt)}
        for rms, sse, new_dt, index in re.findall(
            r"rms = ([0-9.]+)/[0-9.]+, sse=([0-9.]+)/[0-9.]+, "
            r"time step reduction [0-9]+ of [0-9]+ to ([0-9.]+).*"
            r"\n\s*RMS increased, rejecting step\n([0-9]{3}): dt:",
            native_log,
        )
    }
    native_dt = {
        int(index): float(value)
        for index, value in re.findall(r"(?m)^([0-9]{3}): dt:\s*([0-9.]+)", native_log)
    }
    dt = 0.5
    last_rms = report["initial_objective"]["rms"]
    last_sse = report["initial_objective"]["sse"]
    reductions = 0
    failure = None
    for index, source_path in enumerate(args.source_steps, 1):
        step_started = time.perf_counter()
        normals = initial_vertex_normals(current, faces)
        if args.gradient_probe_root is None:
            probe = args.gradient_probes[index - 1]
            gradient_path = lambda stage: probe / f"lh.gradient.{stage}"
        else:
            probe = args.gradient_probe_root
            gradient_path = lambda stage: probe / f"lh.gradient.step{index:02d}.{stage}"
        clear = np.fromfile(gradient_path("clear"), dtype=STATE)
        checks = {
            "start_xyz": comparison(current, clear["floats"][:, :3]),
            "active_normals": comparison(normals[~ripped], clear["floats"][~ripped, 3:6]),
            "ripped_flags": comparison(ripped.astype(np.int32), clear["flags"][:, 0]),
        }
        if args.gradient_probe_root is not None:
            checks["cropped_at_start"] = comparison(
                cropped, np.fromfile(gradient_path("clear.cropped"), dtype=np.int32),
            )
        intensity = intensity_gradient(
            placement, current, normals, ripped, values, border[5], affine,
            brain.header.get_zooms()[:3], weight=0.2, sigma_global=args.sigma,
        )
        offsets, candidates = vertex_buckets(current, original, ripped)
        repulsion = surface_repulsion_gradient(
            current, normals, original, fixed_normals, ripped, offsets, candidates,
            weight=5.0, cropped=cropped,
        )
        with_repulsion = np.float32(intensity + repulsion)
        averaged = average_signed_gradients(
            with_repulsion, faces, ripped, args.gradient_averages, ordered_neighbors=ordered,
        )
        normal = spring_gradient(current, normals, faces, ripped, weight=0.3, direction="normal", ordered_neighbors=ordered)
        with_normal = np.float32(averaged + normal)
        basis = tangent_basis(normals)
        scalar = quadratic_curvature(current, normals, basis, ripped, two_offsets, two_candidates)
        with_curvature = np.float32(with_normal + np.float32(scalar[:, None] * normals))
        tangent = spring_gradient(current, normals, faces, ripped, weight=0.3, direction="tangent", ordered_neighbors=ordered)
        final = np.float32(with_curvature + tangent)
        for stage, value in zip(
            STAGES, (intensity, with_repulsion, averaged, with_normal, with_curvature, final),
        ):
            expected = np.fromfile(gradient_path(stage), dtype=STATE)
            checks[stage] = comparison(value, expected["floats"][:, 6:9])
        entry = {
            "iteration": index, "cropped_at_start": int(np.count_nonzero(cropped)),
            "dt": dt, "native_dt_printed": native_dt.get(index), "gradient": checks,
        }
        report["steps"].append(entry)
        wrong = next(
            (name for name, check in checks.items()
             if check["exact_elements"] != check["total_elements"]), None,
        )
        if wrong:
            failure = f"step {index} first gradient difference: {wrong}"
            break
        previous = current
        if index in native_rejections:
            rejected_proposal, rejected_offsets = unconstrained_step_with_offsets(
                previous, final, ripped, dt=dt,
            )
            rejected, _ = asynchronous_first_step(
                previous, faces, rejected_proposal, ripped, fast=args.fast_collision,
                offsets=rejected_offsets, ordered_neighbors=ordered,
            )
            rejected_blocked = np.any(rejected_proposal != previous, axis=1) & np.all(
                rejected == previous, axis=1,
            )
            cropped = np.where(
                ripped, cropped, np.where(rejected_blocked, cropped + 1, 0),
            ).astype(np.int32)
            rejected_intensity, rejected_rms, _ = intensity_error(
                placement, rejected, values, ripped, affine,
            )
            rejected_normals = initial_vertex_normals(rejected, faces)
            rejected_spring = tangential_spring_energy(
                rejected, rejected_normals, faces, ripped, ordered_neighbors=ordered,
            )
            rejected_area = surface_total_area(rejected, faces)
            rejected_sse = pial_placement_sse(
                rejected_intensity, rejected_spring, orig_area, rejected_area,
            )
            reference = native_rejections[index]
            entry["rejected_trial"] = {
                "dt": dt, "rms": rejected_rms, "sse": rejected_sse,
                "rms_printed": reference["rms_printed"],
                "sse_printed": reference["sse_printed"],
                "cropped_after_rejection": int(np.count_nonzero(cropped)),
            }
            if (round(rejected_rms, 4) != reference["rms_printed"]
                    or round(rejected_sse, 1) != reference["sse_printed"]
                    or rejected_rms <= last_rms):
                failure = f"step {index} rejected trial differs"
                break
            dt *= 0.5
            reductions += 1
            entry["dt"] = dt
        proposal, offsets = unconstrained_step_with_offsets(previous, final, ripped, dt=dt)
        current, order = asynchronous_first_step(
            previous, faces, proposal, ripped, fast=args.fast_collision,
            offsets=offsets, ordered_neighbors=ordered,
        )
        blocked = np.any(proposal != previous, axis=1) & np.all(current == previous, axis=1)
        cropped = np.where(ripped, cropped, np.where(blocked, cropped + 1, 0)).astype(np.int32)
        if args.gradient_probe_root is not None:
            after = np.fromfile(gradient_path("after_collision"), dtype=STATE)
            entry["collision_geometry"] = comparison(current, after["floats"][:, :3])
            entry["cropped"] = comparison(
                cropped, np.fromfile(gradient_path("after_collision.cropped"), dtype=np.int32),
            )
        source, source_faces = nib.freesurfer.read_geometry(source_path)
        source = source.astype(np.float32)
        entry["geometry"] = comparison(current, source)
        vertex_distance = np.linalg.norm(
            current.astype(np.float64) - source.astype(np.float64), axis=1,
        )
        entry["geometry_distance_mm"] = {
            "median": float(np.median(vertex_distance)),
            "p99": float(np.percentile(vertex_distance, 99)),
            "max": float(vertex_distance.max(initial=0)),
            "vertices_over_1e-4_mm": int(np.count_nonzero(vertex_distance > 1e-4)),
        }
        entry["ordered_faces_exact"] = bool(np.array_equal(faces, source_faces))
        if index in installed_checkpoints:
            installed_xyz, installed_faces = nib.freesurfer.read_geometry(
                installed_checkpoints[index],
            )
            if not np.array_equal(faces, installed_faces):
                failure = f"step {index} installed ordered faces differ"
                break
            installed_distance = np.linalg.norm(
                current.astype(np.float64) - installed_xyz.astype(np.float64), axis=1,
            )
            entry["installed_geometry_distance_mm"] = {
                "exact_vertices": int(np.count_nonzero(np.all(current == installed_xyz, axis=1))),
                "median": float(np.median(installed_distance)),
                "p99": float(np.percentile(installed_distance, 99)),
                "max": float(installed_distance.max(initial=0)),
                "vertices_over_1e-4_mm": int(np.count_nonzero(installed_distance > 1e-4)),
                "vertices_over_0.1_mm": int(np.count_nonzero(installed_distance > 0.1)),
            }
        entry["processed_vertices"] = len(order)
        entry["cropped_after_step"] = int(np.count_nonzero(cropped))
        entry["python_coordinate_sha256"] = hashlib.sha256(current.tobytes()).hexdigest()
        entry["source_coordinate_sha256"] = hashlib.sha256(source.tobytes()).hexdigest()
        entry["objective"] = objective(f"step{index}", current, initial_vertex_normals(current, faces))
        if index in installed_objectives:
            entry["installed_printed_objective_error"] = {
                "sse": entry["objective"]["sse"] - installed_objectives[index]["sse"],
                "rms": entry["objective"]["rms"] - installed_objectives[index]["rms"],
            }
        entry["seconds_excluding_initial_prep"] = time.perf_counter() - step_started
        if index == 2 and args.output_second:
            np.save(args.output_second, current)
        if index == 3 and args.output_third:
            np.save(args.output_third, current)
        if "collision_geometry" in entry and entry["collision_geometry"]["exact_elements"] != current.size:
            failure = f"step {index} collision coordinates differ"
            break
        if "cropped" in entry and entry["cropped"]["exact_elements"] != len(cropped):
            failure = f"step {index} cropped counters differ"
            break
        if entry["geometry"]["exact_elements"] != source.size or not entry["ordered_faces_exact"]:
            failure = f"step {index} geometry differs"
            break
        if (entry["objective"]["native_area_abs_error"] != 0
                or entry["objective"]["native_rms_abs_error"] > 1e-12
                or entry["objective"]["native_sse_abs_error"] > 1e-6):
            failure = f"step {index} objective differs"
            break
        if index in native_dt and abs(native_dt[index] - dt) > 5e-4:
            failure = f"step {index} timestep differs"
            break
        rms, sse = entry["objective"]["rms"], entry["objective"]["sse"]
        reduce = (100.0 * (last_sse - sse) / last_sse < 1e-4
                  or rms > last_rms - 0.05)
        if reduce:
            reductions += 1
            dt *= 0.5
        entry["reduced_after_step"] = reduce
        entry["reductions_so_far"] = reductions
        last_rms, last_sse = rms, sse
        if reductions > 2 and index < step_count:
            failure = f"first outer pass ended at step {index}"
            break

    report["total_seconds_including_io_jit"] = time.perf_counter() - started
    report["first_failure"] = failure
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if args.require_exact and (
        failure or len(report["steps"]) != step_count
        or report["initial_objective"]["native_area_abs_error"] != 0
        or report["initial_objective"]["native_rms_abs_error"] > 1e-12
        or report["initial_objective"]["native_sse_abs_error"] > 1e-6
    ):
        raise SystemExit(failure or "initial objective/step count differs")


if __name__ == "__main__":
    main()
