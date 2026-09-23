#!/usr/bin/env python3
"""Locate reverse-output differences without rerunning or modifying a network."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates
import surfa as sf


def metrics(reference, candidate):
    delta = np.asarray(candidate, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    rmse = float(np.sqrt(np.mean(delta*delta)))
    scale = float(np.sqrt(np.mean(np.asarray(reference, dtype=np.float64)**2)))
    return {"mae": float(np.mean(np.abs(delta))), "max_abs": float(np.max(np.abs(delta))),
            "rmse": rmse, "normalised_rmse": rmse/scale if scale else None}


def coordinates(warp):
    # Reconstruct the actual Cython sampler's float32 (index + disp_crs)
    # arithmetic, rather than using the alternative direct abs_crs conversion.
    grid = np.indices(warp.target.shape, dtype=np.float32).transpose(1, 2, 3, 0)
    return grid + np.asarray(warp.convert(format=sf.Warp.Format.disp_crs).data, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True, help="validation/full192/report.json")
    parser.add_argument("--outdir", type=Path)
    parser.add_argument("--modes", nargs="+", default=["deform", "joint"])
    parser.add_argument("--relative-threshold", type=float, default=1e-3,
                        help="Log every voxel with abs difference > this fraction of source max-abs intensity")
    parser.add_argument("--boundary-tolerance", type=float, default=1e-3, help="Near-boundary distance in source voxels")
    args = parser.parse_args()
    outdir = args.outdir or args.report.parent / "reverse_output_diagnosis"
    outdir.mkdir(parents=True, exist_ok=True)
    original = json.loads(args.report.read_text())
    source_path = original["case"]["fixed"]
    source = sf.load_volume(source_path)
    intensity_scale = float(np.max(np.abs(source.data)))
    threshold = args.relative_threshold * intensity_scale
    summary = {"report": str(args.report.resolve()), "source_fixed_image": source_path,
               "source_shape": list(map(int, source.shape)), "intensity_scale_max_abs": intensity_scale,
               "relative_threshold": args.relative_threshold, "absolute_threshold": threshold,
               "boundary_tolerance_vox": args.boundary_tolerance,
               "sampling_domain": "Surfa: every source coordinate satisfies 0 <= coordinate < dimension",
               "limitation": "Coordinates reconstructed from saved RAS warps describe reapplication; export can round a tiny negative native CRS coordinate to zero, so original in-memory coordinates cannot always be recovered.",
               "modes": {}}
    for mode in args.modes:
        entry = original["modes"][mode]
        artifacts = entry["artifacts"]
        volumes = [sf.load_volume(artifacts[side]["fixed_moved"]) for side in ("reference", "candidate")]
        warps = [sf.load_warp(artifacts[side]["inverse"]) for side in ("reference", "candidate")]
        values = [np.asarray(volume.data) for volume in volumes]
        coords = [coordinates(warp) for warp in warps]
        valid, near, distances = [], [], []
        for loc in coords:
            inside = np.ones(loc.shape[:-1], dtype=bool)
            distance = np.full(loc.shape[:-1], np.inf, dtype=np.float32)
            for d, size in enumerate(source.shape):
                inside &= (loc[..., d] >= 0) & (loc[..., d] < size)
                distance = np.minimum(distance, np.minimum(np.abs(loc[..., d]), np.abs(loc[..., d]-size)))
            valid.append(inside)
            distances.append(distance)
            near.append(distance <= args.boundary_tolerance)
        difference = np.abs(values[1].astype(np.float64)-values[0].astype(np.float64))
        large = difference > threshold
        support = [value != 0 for value in values]
        support_xor = support[0] ^ support[1]
        crossing = valid[0] ^ valid[1]
        near_either = near[0] | near[1]
        reapplied = [np.asarray(source.transform(warp).data) for warp in warps]
        clamped = [map_coordinates(source.data, loc.transpose(3, 0, 1, 2), order=1,
                                   mode="nearest", prefilter=False) for loc in coords]

        def voxel_record(index):
            point = tuple(map(int, index))
            result = {"i": point[0], "j": point[1], "k": point[2],
                      "abs_difference": float(difference[point]), "support_xor": bool(support_xor[point]),
                      "saved_coordinate_boundary_crossing": bool(crossing[point]),
                      "near_saved_boundary": bool(near_either[point])}
            for side_index, side in enumerate(("reference", "candidate")):
                result[side+"_intensity"] = float(values[side_index][point])
                result[side+"_reapplied_saved_warp"] = float(reapplied[side_index][point])
                result[side+"_clamped_source_sample"] = float(clamped[side_index][point])
                result[side+"_saved_coordinate_valid"] = bool(valid[side_index][point])
                result[side+"_boundary_distance_vox"] = float(distances[side_index][point])
                for d, axis in enumerate("ijk"):
                    result[side+"_source_"+axis] = float(coords[side_index][point][d])
            return result

        indices = np.argwhere(large)
        if len(indices):
            indices = indices[np.argsort(difference[tuple(indices.T)])[::-1]]
        maximum = np.unravel_index(np.argmax(difference), difference.shape)
        largest = voxel_record(maximum)
        csv_path = outdir / f"{mode}_all_above_threshold.csv"
        with open(csv_path, "w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(largest))
            writer.writeheader()
            writer.writerows(voxel_record(index) for index in indices)
        mode_report = {
            "artifacts": {side: {key: artifacts[side][key] for key in ("fixed_moved", "inverse")}
                          for side in ("reference", "candidate")},
            "output_error": metrics(*values), "maximum_difference_voxel": largest,
            "voxels": int(difference.size), "large_difference_voxels": int(large.sum()),
            "all_large_difference_voxels_csv": str(csv_path),
            "threshold_counts": {str(fraction): int(np.count_nonzero(difference > intensity_scale*fraction))
                                 for fraction in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1)},
            "support": {"reference_nonzero": int(support[0].sum()), "candidate_nonzero": int(support[1].sum()),
                        "nonzero_xor": int(support_xor.sum()), "large_with_nonzero_xor": int((large & support_xor).sum())},
            "saved_coordinate_boundary": {"all_domain_crossings": int(crossing.sum()),
                                          "large_with_domain_crossing": int((large & crossing).sum()),
                                          "large_near_boundary": int((large & near_either).sum()),
                                          "large_away_from_boundary": int((large & ~near_either).sum())},
            "saved_inverse_coordinate_error_vox": metrics(coords[0], coords[1]),
            "reapplied_saved_warp_error": metrics(*reapplied),
            "original_output_vs_reapplied": {side: metrics(values[i], reapplied[i])
                                            for i, side in enumerate(("reference", "candidate"))},
            "error_excluding_large_voxels": metrics(values[0][~large], values[1][~large]) if np.any(~large) else None,
        }
        summary["modes"][mode] = mode_report
        (outdir / "report.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
        print(json.dumps({"mode": mode, "large_voxels": int(large.sum()),
                          "maximum": largest, "boundary": mode_report["saved_coordinate_boundary"]}), flush=True)


if __name__ == "__main__":
    main()
