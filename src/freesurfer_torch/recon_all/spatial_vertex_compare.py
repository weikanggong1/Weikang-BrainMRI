"""Diagnostic nearest-vertex comparison when native surface indices differ.

This compares locations in FreeSurfer surface RAS. Nearest neighbours can
cross folds or reuse vertices; spatial agreement does not establish native
vertex correspondence. Run ``compare_subject`` for strict acceptance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel.freesurfer as fs
import numpy as np
from scipy.spatial import cKDTree

from .compare_subject import MAP_SURFACES


def _summary(values, *, prefix):
    if not len(values):
        return {prefix + name: None for name in ("mean", "p95", "p99", "max")}
    return {
        prefix + "mean": float(np.mean(values)),
        prefix + "p95": float(np.quantile(values, 0.95)),
        prefix + "p99": float(np.quantile(values, 0.99)),
        prefix + "max": float(np.max(values)),
    }


def _surface(subject, hemi, name, nvertices=None, faces=None):
    vertices, triangles = fs.read_geometry(subject / "surf" / f"{hemi}.{name}")
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices)
            or not np.isfinite(vertices).all()):
        raise ValueError(f"Invalid {hemi}.{name} coordinates")
    if nvertices is not None and (len(vertices) != nvertices or not np.array_equal(triangles, faces)):
        raise ValueError(f"{hemi}.{name} does not share native topology with {hemi}.white")
    return vertices, triangles


def _nearest(source, target, max_distance_mm):
    distance, index = cKDTree(target).query(source)
    covered = distance <= max_distance_mm
    result = {
        "source_vertices": len(source), "target_vertices": len(target),
        "matched_count": int(covered.sum()), "coverage_fraction": float(covered.mean()),
        "unique_target_fraction": float(np.unique(index[covered]).size / len(target)),
        **_summary(distance, prefix="distance_mm_"),
    }
    return index, distance, covered, result


def _metric(source, target, index, covered, tolerance, min_coverage):
    paired = target[index]
    finite = np.isfinite(source) & np.isfinite(paired)
    valid = covered & finite
    difference = np.abs(paired[valid].astype(np.float64) - source[valid].astype(np.float64))
    limits = tolerance["atol"] + tolerance["rtol"] * np.abs(source[valid])
    nonfinite = int((covered & ~finite).sum())
    outliers = int(np.count_nonzero(difference > limits)) + nonfinite
    matched = int(covered.sum())
    coverage = float(covered.mean())
    return {
        "status": "passed" if coverage >= min_coverage and outliers == 0 else "failed",
        "source_vertices": len(source), "matched_count": matched,
        "coverage_fraction": coverage, "finite_pair_count": int(valid.sum()),
        "nonfinite_pair_count": nonfinite, "tolerance": tolerance,
        "outlier_count": outliers,
        "outlier_fraction": float(outliers / matched) if matched else None,
        "mae": float(np.mean(difference)) if len(difference) else None,
        "p95_abs_error": float(np.quantile(difference, 0.95)) if len(difference) else None,
        "p99_abs_error": float(np.quantile(difference, 0.99)) if len(difference) else None,
        "max_abs_error": float(np.max(difference)) if len(difference) else None,
    }


def _direction(reference, candidate, index, covered, tolerance, min_coverage):
    return _metric(reference, candidate, index, covered, tolerance, min_coverage)


def compare_spatial_vertices(reference, candidate, *, max_distance_mm=2.0,
                             min_coverage=0.99, tolerances=None):
    """Compare both hemispheres using bidirectional white-vertex nearest neighbours.

    Unspecified metric tolerances are exact. ``spatial_thresholds_met`` never
    implies native ordered vertex equivalence. Distances above the spatial
    cutoff on either paired white or paired pial are excluded from metric
    error statistics and reduce coverage.
    """
    if not np.isfinite(max_distance_mm) or max_distance_mm <= 0:
        raise ValueError("max_distance_mm must be positive and finite")
    if not np.isfinite(min_coverage) or not 0 < min_coverage <= 1:
        raise ValueError("min_coverage must be in (0, 1]")
    reference, candidate = Path(reference).resolve(), Path(candidate).resolve()
    tolerances = tolerances or {}
    for name, limits in tolerances.items():
        if (not isinstance(limits, dict) or set(limits) - {"atol", "rtol"}
                or any(not np.isfinite(value) or value < 0 for value in limits.values())):
            raise ValueError(f"Invalid tolerance: {name}")

    hemispheres = {}
    for hemi in ("lh", "rh"):
        try:
            rw, rf = _surface(reference, hemi, "white")
            cw, cf = _surface(candidate, hemi, "white")
            rp, _ = _surface(reference, hemi, "pial", len(rw), rf)
            cp, _ = _surface(candidate, hemi, "pial", len(cw), cf)
        except (OSError, ValueError) as error:
            hemispheres[hemi] = {"status": "error", "reason": str(error)}
            continue
        ri, rd, rcovered, rgeo = _nearest(rw, cw, max_distance_mm)
        ci, cd, ccovered, cgeo = _nearest(cw, rw, max_distance_mm)
        rgeo["reciprocal_fraction"] = float(np.mean(ci[ri[rcovered]] == np.flatnonzero(rcovered))) if rcovered.any() else None
        cgeo["reciprocal_fraction"] = float(np.mean(ri[ci[ccovered]] == np.flatnonzero(ccovered))) if ccovered.any() else None
        _, _, _, rpgeo = _nearest(rp, cp, max_distance_mm)
        _, _, _, cpgeo = _nearest(cp, rp, max_distance_mm)
        rpaired = np.linalg.norm(rp - cp[ri], axis=1)
        cpaired = np.linalg.norm(cp - rp[ci], axis=1)
        rmetriccovered = rcovered & (rpaired <= max_distance_mm)
        cmetriccovered = ccovered & (cpaired <= max_distance_mm)
        shape = {
            "white": {"reference_to_candidate": rgeo, "candidate_to_reference": cgeo},
            "pial": {"reference_to_candidate": rpgeo, "candidate_to_reference": cpgeo},
            "pial_under_white_match": {
                "reference_to_candidate": {
                    "matched_count": int(rcovered.sum()),
                    "joint_coverage_fraction": float(rmetriccovered.mean()),
                    "fraction_above_max_distance": float(np.mean(rpaired[rcovered] > max_distance_mm)) if rcovered.any() else None,
                    **_summary(rpaired[rcovered], prefix="distance_mm_")},
                "candidate_to_reference": {
                    "matched_count": int(ccovered.sum()),
                    "joint_coverage_fraction": float(cmetriccovered.mean()),
                    "fraction_above_max_distance": float(np.mean(cpaired[ccovered] > max_distance_mm)) if ccovered.any() else None,
                    **_summary(cpaired[ccovered], prefix="distance_mm_")},
            },
        }
        metrics = {}
        for name in MAP_SURFACES:
            try:
                rv = fs.read_morph_data(reference / "surf" / f"{hemi}.{name}")
                cv = fs.read_morph_data(candidate / "surf" / f"{hemi}.{name}")
                if rv.shape != (len(rw),) or cv.shape != (len(cw),):
                    raise ValueError("Metric length does not match white vertex count")
                setting = tolerances.get(name, {})
                tolerance = {key: float(setting.get(key, 0.0)) for key in ("atol", "rtol")}
                metrics[name] = {
                    "reference_to_candidate": _direction(rv, cv, ri, rmetriccovered, tolerance, min_coverage),
                    "candidate_to_reference": _direction(cv, rv, ci, cmetriccovered, tolerance, min_coverage),
                }
            except (OSError, ValueError) as error:
                metrics[name] = {"status": "error", "reason": str(error)}
        coverage_ok = all(direction["coverage_fraction"] >= min_coverage for surface in ("white", "pial")
                          for direction in shape[surface].values())
        metrics_ok = all(item.get("status") != "error" and all(
            check["status"] == "passed" for check in item.values()) for item in metrics.values())
        hemispheres[hemi] = {"status": "passed" if coverage_ok and metrics_ok else "failed",
                             "shape": shape, "metrics": metrics}

    return {
        "schema_version": 1, "comparison_basis": "bidirectional white-surface nearest vertices",
        "native_index_consistency_proven": False,
        "limitation": "Spatial nearest neighbours can reuse vertices or cross cortical folds; use strict compare_subject for native correspondence.",
        "reference": str(reference), "candidate": str(candidate),
        "max_distance_mm": max_distance_mm, "min_coverage": min_coverage,
        "tolerances": tolerances, "unspecified_tolerances": "exact",
        "spatial_thresholds_met": all(value["status"] == "passed" for value in hemispheres.values()),
        "hemispheres": hemispheres,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-distance-mm", type=float, default=2.0)
    parser.add_argument("--min-coverage", type=float, default=0.99)
    parser.add_argument("--tolerances", type=Path, help="JSON object: metric -> {atol, rtol}; others exact")
    args = parser.parse_args(argv)
    report = compare_spatial_vertices(
        args.reference, args.candidate, max_distance_mm=args.max_distance_mm,
        min_coverage=args.min_coverage,
        tolerances=json.loads(args.tolerances.read_text()) if args.tolerances else None,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Spatial thresholds {'met' if report['spatial_thresholds_met'] else 'not met'}; {args.output}")
    return 0 if report["spatial_thresholds_met"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
