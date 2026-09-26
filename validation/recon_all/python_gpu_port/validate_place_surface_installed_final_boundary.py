"""Compare the independent pial optimizer output with installed and archived final surfaces."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import nibabel.freesurfer as fs
import numpy as np


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(a: np.ndarray, b: np.ndarray) -> dict:
    lhs, rhs = a.astype(np.float32), b.astype(np.float32)
    exact = lhs.view(np.uint32) == rhs.view(np.uint32)
    different = np.flatnonzero(~np.all(exact, axis=1))
    distances = np.linalg.norm(lhs.astype(np.float64) - rhs.astype(np.float64), axis=1)
    return {
        "exact_vertices": int(len(lhs) - len(different)),
        "total_vertices": len(lhs),
        "exact_components": int(exact.sum()),
        "total_components": int(exact.size),
        "maximum_distance_mm": float(distances.max()),
        "p99_distance_mm": float(np.quantile(distances, 0.99)),
        "first_different_vertices": different[:12].tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--hemisphere", choices=("lh", "rh"), default="lh")
    parser.add_argument("--metrics-output-dir", type=Path)
    parser.add_argument("--optimizer-candidate", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.module))
    from fnit.recon_all.surface_area_gpu import vertex_area
    from fnit.recon_all.place_surface_final_cleanup import (
        pin_medial_wall, repair_intersections,
    )
    import torch
    torch.set_num_threads(1)

    hemi = args.hemisphere
    surf = args.subject / "surf"
    paths = {
        "installed_optimizer_ram": args.root / (
            "probe_installed_ram_official_pass3" if hemi == "lh"
            else "probe_installed_rh_ram_official_pass3") / "surf" / f"{hemi}.pial.ram",
        "python_optimizer_step41": args.optimizer_candidate or (
            args.root / f"independent_{hemi}_pial_first_outer_20260926" / f"{hemi}.step41.npz"),
        "installed_final_cli": args.root / (
            "probe_installed_official_run" if hemi == "lh"
            else "probe_installed_rh_official_run") / "surf" / f"{hemi}.pial.objective",
        "archived_pial_T1": surf / f"{hemi}.pial.T1",
        "archived_pial": surf / f"{hemi}.pial",
        "archived_area_pial": surf / f"{hemi}.area.pial",
        "archived_curv_pial": surf / f"{hemi}.curv.pial",
        "archived_thickness": surf / f"{hemi}.thickness",
        "archived_volume": surf / f"{hemi}.volume",
        "white": surf / f"{hemi}.white",
        "cortex_label": args.subject / "label" / f"{hemi}.cortex.label",
        "placement_label": args.subject / "label" / f"{hemi}.cortex+hipamyg.label",
    }
    meshes = {}
    faces = None
    for name in ("installed_optimizer_ram", "installed_final_cli", "archived_pial_T1", "archived_pial"):
        xyz, topology = fs.read_geometry(str(paths[name]))
        if faces is None:
            faces = topology
        if not np.array_equal(faces, topology):
            raise ValueError(f"{name}: ordered faces differ")
        meshes[name] = xyz.astype(np.float32)
    candidate = np.load(paths["python_optimizer_step41"])["xyz"]
    if candidate.shape != meshes["installed_optimizer_ram"].shape:
        raise ValueError("candidate vertex count differs")
    meshes["python_optimizer_step41"] = candidate
    white, white_faces = fs.read_geometry(str(paths["white"]))
    if not np.array_equal(faces, white_faces):
        raise ValueError("white ordered faces differ")
    cortex = fs.read_label(str(paths["cortex_label"])).astype(np.int64)
    outside_cortex = np.ones(len(candidate), dtype=np.bool_)
    outside_cortex[cortex] = False
    pinned = pin_medial_wall(candidate, white, cortex)
    meshes["python_after_medial_pin"] = pinned
    placement_vertices = fs.read_label(str(paths["placement_label"])).astype(np.int64)
    ripped = np.ones(len(candidate), dtype=np.bool_)
    ripped[placement_vertices] = False
    repaired, cleanup = repair_intersections(pinned, faces, ripped)
    meshes["python_after_intersection_repair"] = repaired
    pairs = (
        ("python_optimizer_step41", "installed_optimizer_ram"),
        ("installed_optimizer_ram", "installed_final_cli"),
        ("python_after_medial_pin", "installed_final_cli"),
        ("python_after_intersection_repair", "installed_final_cli"),
        ("installed_final_cli", "archived_pial_T1"),
        ("archived_pial_T1", "archived_pial"),
        ("python_optimizer_step41", "archived_pial"),
    )
    report = {
        "scope": f"{hemi.upper()} installed FreeSurfer 8.2.0-1 pial; optimizer candidate before pinning/intersection cleanup",
        "optimizer_candidate_source": str(paths["python_optimizer_step41"]),
        "sha256": {name: digest(path) for name, path in paths.items()},
        "ordered_faces_exact": True,
        "medial_pin": {"outside_cortex_vertices": int(outside_cortex.sum())},
        "intersection_cleanup": cleanup,
        "mesh_comparisons": {f"{a}_vs_{b}": compare(meshes[a], meshes[b]) for a, b in pairs},
    }
    official_area = fs.read_morph_data(str(paths["archived_area_pial"])).astype(np.float32)
    candidate_area = vertex_area(candidate, faces, device="cpu")
    pinned_area = vertex_area(pinned, faces, device="cpu")
    repaired_area = vertex_area(repaired, faces, device="cpu")
    final_area = vertex_area(meshes["archived_pial"], faces, device="cpu")
    if len(official_area) != len(candidate_area):
        raise ValueError("archived per-vertex area length differs")
    report["area_pial"] = {
        "method": "same standalone vertex_area CPU kernel on candidate and archived final geometry",
        "kernel_final_vs_official": {
            "exact_vertices": int(np.count_nonzero(final_area.view(np.uint32) == official_area.view(np.uint32))),
            "max_abs_mm2": float(np.max(np.abs(final_area - official_area))),
        },
        "candidate_vs_official": {
            "exact_vertices": int(np.count_nonzero(candidate_area.view(np.uint32) == official_area.view(np.uint32))),
            "max_abs_mm2": float(np.max(np.abs(candidate_area - official_area))),
            "outliers_at_0.001_plus_0.001_relative_mm2": int(np.count_nonzero(
                np.abs(candidate_area - official_area) > 0.001 + 0.001 * np.abs(official_area)
            )),
        },
        "after_pin_vs_official": {
            "exact_vertices": int(np.count_nonzero(pinned_area.view(np.uint32) == official_area.view(np.uint32))),
            "max_abs_mm2": float(np.max(np.abs(pinned_area - official_area))),
            "outliers_at_0.001_plus_0.001_relative_mm2": int(np.count_nonzero(
                np.abs(pinned_area - official_area) > 0.001 + 0.001 * np.abs(official_area)
            )),
        },
        "after_repair_vs_official": {
            "exact_vertices": int(np.count_nonzero(repaired_area.view(np.uint32) == official_area.view(np.uint32))),
            "max_abs_mm2": float(np.max(np.abs(repaired_area - official_area))),
            "outliers_at_0.001_plus_0.001_relative_mm2": int(np.count_nonzero(
                np.abs(repaired_area - official_area) > 0.001 + 0.001 * np.abs(official_area)
            )),
        },
    }
    report["other_vertex_metrics"] = {
        "curvature": "archive map hashed; candidate metric comparison gated on exact final geometry",
        "thickness": "archive map hashed; candidate metric comparison gated on exact final geometry",
        "volume": "archive map hashed; candidate metric comparison gated on exact final geometry",
    }
    if args.metrics_output_dir is not None:
        from fnit.recon_all.surface_curvature_gpu import curvature_map
        from fnit.recon_all.surface_roi_gpu import vertex_volume_map
        from fnit.recon_all.surface_thickness_gpu import thickness_map

        args.metrics_output_dir.mkdir(parents=True, exist_ok=True)
        independent_pial = args.metrics_output_dir / f"{hemi}.pial"
        fs.write_geometry(str(independent_pial), repaired, faces)
        round_trip, round_trip_faces = fs.read_geometry(str(independent_pial))
        if not np.array_equal(round_trip.astype(np.float32).view(np.uint32), repaired.view(np.uint32)) or not np.array_equal(round_trip_faces, faces):
            raise ValueError("independent pial geometry write changed vertices or faces")

        def metric(name: str, archived: Path, compute) -> dict:
            output = args.metrics_output_dir / name
            started = time.perf_counter()
            compute(output)
            seconds = time.perf_counter() - started
            reference = fs.read_morph_data(str(archived)).astype(np.float32)
            candidate = fs.read_morph_data(str(output)).astype(np.float32)
            if reference.shape != candidate.shape:
                raise ValueError(f"{name}: vertex count differs")
            delta = np.abs(candidate.astype(np.float64) - reference.astype(np.float64))
            return {
                "seconds": seconds,
                "candidate_sha256": digest(output),
                "vertices": len(reference),
                "exact_float32_vertices": int(np.count_nonzero(
                    candidate.view(np.uint32) == reference.view(np.uint32))),
                "maximum_absolute_error": float(delta.max()),
                "p99_absolute_error": float(np.quantile(delta, 0.99)),
                "outliers_at_0.005_plus_0.001_relative": int(np.count_nonzero(
                    delta > 0.005 + 0.001 * np.abs(reference))),
            }

        report["independent_final_pial_sha256"] = digest(independent_pial)
        report["other_vertex_metrics"] = {
            "curvature": metric(
                f"{hemi}.curv.pial", paths["archived_curv_pial"],
                lambda output: curvature_map(independent_pial, output, device="cpu")),
            "thickness": metric(
                f"{hemi}.thickness", paths["archived_thickness"],
                lambda output: thickness_map(paths["white"], independent_pial, output, device="cpu")),
            "volume": metric(
                f"{hemi}.volume", paths["archived_volume"],
                lambda output: vertex_volume_map(
                    paths["white"], independent_pial, paths["cortex_label"], output, device="cpu")),
        }
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"mesh_comparisons": report["mesh_comparisons"], "area_pial": report["area_pial"]}, indent=2))


if __name__ == "__main__":
    main()
