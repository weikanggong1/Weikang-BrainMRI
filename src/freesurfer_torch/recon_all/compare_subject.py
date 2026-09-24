"""Compare two independently produced, native FreeSurfer subject directories.

Run ``python -m freesurfer_torch.recon_all.compare_subject REF CAND --output report.json``.
Comparisons default to exact values. A JSON mapping such as
``{"thickness": {"atol": 0.005, "rtol": 0.001}, "coordinates": {"atol": 0.01}}``
sets explicit tolerances; unspecified metrics remain exact. Coordinates use
Euclidean displacement in mm, so their relative tolerance must be zero.
No registration, interpolation, missing-file exclusion, or ROI averaging is
used to rescue failed vertex comparisons. This checks artifacts, not whether
the candidate was actually generated independently of the reference.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shlex
import struct

import nibabel as nib
import nibabel.freesurfer as fs
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


SURFACES = ("white", "pial", "white.preaparc", "inflated")
MAP_SURFACES = {
    "thickness": ("white", "pial"),
    "area": ("white",),
    "area.pial": ("pial",),
    "area.mid": ("white", "pial"),
    "volume": ("white", "pial"),
    "curv": ("white",),
    "curv.pial": ("pial",),
    "sulc": ("inflated",),
    "white.H": ("white.preaparc",),
    "white.K": ("white.preaparc",),
    "inflated.H": ("inflated",),
    "inflated.K": ("inflated",),
}
VOLUMES = ("aseg.mgz", "aparc+aseg.mgz")
ANNOTATIONS = ("aparc", "aparc.DKTatlas", "aparc.a2009s")
INTEGER_COLUMNS = {"Index", "SegId", "NumVert", "NVoxels"}


def _numeric(reference, candidate, tolerance, max_ids):
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if reference.shape != candidate.shape:
        return {"status": "failed", "reason": "shape mismatch",
                "reference_shape": list(reference.shape), "candidate_shape": list(candidate.shape)}
    if reference.ndim != 1 or reference.size == 0:
        raise ValueError("Expected a nonempty, one-dimensional measurement array")
    finite = np.isfinite(reference) & np.isfinite(candidate)
    with np.errstate(invalid="ignore", over="ignore"):
        difference = candidate - reference
        error = np.abs(difference)
        limit = tolerance["atol"] + tolerance["rtol"] * np.abs(reference)
        bad = ~finite | (error > limit)
    ids = np.flatnonzero(bad)
    valid = finite & np.isfinite(difference)
    absolute = error[valid]
    result = {
        "status": "passed" if not len(ids) else "failed", "count": int(reference.size),
        "tolerance": tolerance, "outlier_count": int(len(ids)),
        "outlier_fraction": float(len(ids) / reference.size),
        "outlier_ids": ids[:max_ids].tolist(), "outlier_ids_truncated": len(ids) > max_ids,
        "nonfinite_count": int((~finite).sum()),
        "bias": float(difference[valid].mean()) if valid.any() else None,
        "mae": float(absolute.mean()) if len(absolute) else None,
        "rmse": float(np.linalg.norm(difference[valid]) / np.sqrt(valid.sum())) if valid.any() else None,
        "p95_abs_error": float(np.quantile(absolute, 0.95)) if len(absolute) else None,
        "p99_abs_error": float(np.quantile(absolute, 0.99)) if len(absolute) else None,
        "max_abs_error": float(absolute.max()) if len(absolute) else None,
        "max_error_id": int(np.flatnonzero(valid)[np.argmax(absolute)]) if len(absolute) else None,
    }
    return result


def _topology(vertices, faces):
    n = len(vertices)
    valid = (vertices.shape == (n, 3) and n > 0 and faces.ndim == 2
             and faces.shape[1] == 3 and len(faces) > 0
             and np.all((faces >= 0) & (faces < n)))
    if not valid:
        return {"valid": False, "reason": "invalid vertex or face array"}
    edges = np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1)
    edges, counts = np.unique(edges, axis=0, return_counts=True)
    graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n)).tocsr()
    components = int(connected_components(graph, directed=False, return_labels=False))
    degenerate = int(np.any(np.diff(np.sort(faces, axis=1), axis=1) == 0, axis=1).sum())
    duplicates = len(faces) - len(np.unique(np.sort(faces, axis=1), axis=0))
    result = {"vertices": n, "faces": len(faces), "edges": len(edges),
              "euler": n - len(edges) + len(faces), "components": components,
              "boundary_edges": int((counts == 1).sum()),
              "nonmanifold_edges": int((counts > 2).sum()),
              "degenerate_index_faces": degenerate, "duplicate_faces": duplicates}
    result["valid"] = (components == 1 and result["euler"] == 2
                       and np.all(counts == 2) and degenerate == 0 and duplicates == 0)
    return result


def _surface(reference, candidate, tolerance, max_ids):
    rv, rf = reference
    cv, cf = candidate
    rt, ct = _topology(rv, rf), _topology(cv, cf)
    same = len(rv) == len(cv) and np.array_equal(rf, cf)
    compatible = bool(same and rt["valid"] and ct["valid"])
    result = {"status": "failed", "reference_topology": rt, "candidate_topology": ct,
              "ordered_faces_equal": same, "vertex_correspondence_compatible": compatible}
    if not compatible:
        result["reason"] = "Native ordered topology differs or is not a closed connected genus-zero mesh"
        return result
    displacement = np.linalg.norm(cv - rv, axis=1)
    result["displacement_mm"] = _numeric(np.zeros(len(rv)), displacement, tolerance, max_ids)
    result["status"] = result["displacement_mm"]["status"]
    return result


def _annotation(reference, candidate, nvertices, min_dice, max_ids):
    rl, rc, rn = fs.read_annot(reference, orig_ids=True)
    cl, cc, cn = fs.read_annot(candidate, orig_ids=True)
    if rl.shape != (nvertices,) or cl.shape != (nvertices,):
        raise ValueError("Annotation vertex count does not match white surface")
    result = _labels(rl, cl, min_dice, max_ids)
    result["color_table_equal"] = np.array_equal(rc, cc) and rn == cn
    result["reference_label_names"] = {
        str(int(color[-1])): name.decode("utf-8", errors="replace") if isinstance(name, bytes) else str(name)
        for color, name in zip(rc, rn)}
    if not result["color_table_equal"]:
        result["status"] = "failed"
    return result


def _labels(reference, candidate, min_dice, max_ids):
    if reference.shape != candidate.shape:
        return {"status": "failed", "reason": "label array shape mismatch"}
    if not (np.isfinite(reference).all() and np.isfinite(candidate).all()
            and np.equal(reference, np.floor(reference)).all()
            and np.equal(candidate, np.floor(candidate)).all()):
        raise ValueError("Labels must contain finite integers")
    matches = reference == candidate
    bad = np.flatnonzero(~matches)
    rcounts = dict(zip(*np.unique(reference, return_counts=True)))
    ccounts = dict(zip(*np.unique(candidate, return_counts=True)))
    intersections = dict(zip(*np.unique(reference[matches], return_counts=True)))
    rows = {}
    passed = True
    for label in sorted(rcounts.keys() | ccounts.keys()):
        nr, nc = int(rcounts.get(label, 0)), int(ccounts.get(label, 0))
        intersection = int(intersections.get(label, 0))
        dice = 2 * intersection / (nr + nc)
        present = nr > 0 and nc > 0
        passed &= present and dice >= min_dice
        rows[str(int(label))] = {"reference_count": nr, "candidate_count": nc,
                                 "dice": dice, "count_difference": nc - nr,
                                 "present_in_both": present}
    return {"status": "passed" if passed else "failed", "min_dice": min_dice,
            "labels": rows, "mismatch_count": len(bad),
            "agreement": float(matches.mean()),
            "outlier_ids": bad[:max_ids].tolist() if reference.ndim == 1 else np.array(
                np.unravel_index(bad[:max_ids], reference.shape)).T.tolist(),
            "outlier_ids_truncated": len(bad) > max_ids}


def _read_stats(path):
    columns, rows, measures, command = None, {}, {}, None
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if line.startswith("# ColHeaders "):
            columns = line.split()[2:]
        elif line.startswith("# cmdline "):
            command = line[len("# cmdline "):]
        elif line.startswith("# Measure "):
            parts = [part.strip() for part in line[len("# Measure "):].split(",")]
            if len(parts) != 5:
                raise ValueError(f"Malformed Measure line: {line}")
            key = parts[0] + "." + parts[1]
            if key in measures:
                raise ValueError(f"Duplicate measure: {key}")
            measures[key] = {"value": float(parts[3]), "units": parts[4]}
            if not np.isfinite(measures[key]["value"]):
                raise ValueError(f"Nonfinite stats measure: {key}")
        elif line and not line.startswith("#"):
            if columns is None or len(line.split()) != len(columns):
                raise ValueError("Missing ColHeaders or malformed stats row")
            fields = dict(zip(columns, line.split()))
            name = fields.get("StructName", fields.get("SegId"))
            if name is None or name in rows:
                raise ValueError("Missing or duplicate stats row identifier")
            rows[name] = {key: float(value) for key, value in fields.items() if key != "StructName"}
            if any(not np.isfinite(value) or (key in INTEGER_COLUMNS and value != int(value))
                   for key, value in rows[name].items()):
                raise ValueError(f"Nonfinite value or noninteger count in stats row: {name}")
    if columns is None or not rows:
        raise ValueError("Stats table is absent or empty")
    return columns, rows, measures, command


def _stats(reference, candidate, tolerance, max_ids):
    rc, rr, rm, rcommand = _read_stats(reference)
    cc, cr, cm, ccommand = _read_stats(candidate)
    def volume_mode(command):
        flags = shlex.split(command or "")
        modes = [flag for flag in flags if flag in ("-th3", "-no-th3")]
        return modes[-1] if modes else "unspecified"

    result = {"status": "passed", "columns_equal": rc == cc,
              "reference_command": rcommand, "candidate_command": ccommand,
              "reference_volume_mode": volume_mode(rcommand), "candidate_volume_mode": volume_mode(ccommand),
              "missing_candidate_rows": sorted(rr.keys() - cr.keys()),
              "missing_reference_rows": sorted(cr.keys() - rr.keys()),
              "missing_candidate_measures": sorted(rm.keys() - cm.keys()),
              "missing_reference_measures": sorted(cm.keys() - rm.keys()), "rows": {}, "measures": {}}
    if rc != cc or rr.keys() != cr.keys() or rm.keys() != cm.keys():
        result["status"] = "failed"
    if (volume_mode(rcommand) != "unspecified" and volume_mode(ccommand) != "unspecified"
            and volume_mode(rcommand) != volume_mode(ccommand)):
        result["status"] = "failed"
        result["reason"] = "Explicit stats volume algorithm differs"
    for name in sorted(rr.keys() & cr.keys()):
        result["rows"][name] = {}
        for field in sorted(rr[name].keys() & cr[name].keys()):
            tol = {"atol": 0.0, "rtol": 0.0} if field in INTEGER_COLUMNS else tolerance("stats." + field)
            check = _numeric([rr[name][field]], [cr[name][field]], tol, max_ids)
            check.update(reference=rr[name][field], candidate=cr[name][field])
            result["rows"][name][field] = check
            if check["status"] != "passed":
                result["status"] = "failed"
    for name in sorted(rm.keys() & cm.keys()):
        check = _numeric([rm[name]["value"]], [cm[name]["value"]], tolerance("stats.measure." + name), max_ids)
        check["units_equal"] = rm[name]["units"] == cm[name]["units"]
        if not check["units_equal"]:
            check["status"] = "failed"
        result["measures"][name] = check
        if check["status"] != "passed":
            result["status"] = "failed"
    return result


def _read_synthseg_volumes(path):
    with path.open(newline="") as stream:
        rows = list(csv.reader(stream))
    if (len(rows) != 2 or len(rows[0]) < 3 or rows[0][0] != "subject"
            or rows[0][1] != "total intracranial" or len(rows[0]) != len(rows[1])
            or not rows[1][0] or len(set(rows[0][1:])) != len(rows[0]) - 1):
        raise ValueError("Malformed single-subject SynthSeg volume CSV")
    values = {name: float(value) for name, value in zip(rows[0][1:], rows[1][1:])}
    if any(not np.isfinite(value) for value in values.values()):
        raise ValueError("Nonfinite SynthSeg soft volume")
    return rows[0][1:], rows[1][0], values


def _synthseg_volumes(reference, candidate, tolerance, max_ids):
    reference_columns, reference_subject, reference_values = _read_synthseg_volumes(reference)
    candidate_columns, candidate_subject, candidate_values = _read_synthseg_volumes(candidate)
    result = {
        "status": "passed" if reference_columns == candidate_columns else "failed",
        "units": "mm^3", "volume_basis": "postprocessed soft posterior",
        "reference_subject": reference_subject, "candidate_subject": candidate_subject,
        "columns_equal": reference_columns == candidate_columns,
        "reference_columns": reference_columns, "candidate_columns": candidate_columns,
        "missing_candidate_columns": sorted(reference_values.keys() - candidate_values.keys()),
        "missing_reference_columns": sorted(candidate_values.keys() - reference_values.keys()),
        "columns": {}, "eTIV_column": "total intracranial",
    }
    for name in reference_columns:
        if name not in candidate_values:
            continue
        reference_value = reference_values[name]
        candidate_value = candidate_values[name]
        check = _numeric([reference_value], [candidate_value], tolerance, max_ids)
        check.update(reference=reference_value, candidate=candidate_value,
                     signed_error_mm3=candidate_value - reference_value,
                     abs_error_mm3=abs(candidate_value - reference_value),
                     relative_error=(candidate_value - reference_value) / reference_value
                     if reference_value else None,
                     allowed_error_mm3=tolerance["atol"] + tolerance["rtol"] * abs(reference_value))
        result["columns"][name] = check
        if check["status"] != "passed":
            result["status"] = "failed"
    return result


def compare_subject(reference, candidate, *, tolerances=None, min_label_dice=1.0,
                    min_annotation_dice=1.0, max_outlier_ids=100):
    """Return a JSON-safe report; missing or blocked required checks fail overall.

    ``tolerances`` maps map names, ``coordinates`` (absolute mm), and
    ``stats.COLUMN`` / ``stats.measure.NAME`` / ``synthseg.vol`` to ``atol`` and ``rtol``.
    All vertices must satisfy their tolerance. Dice thresholds apply to every
    present label separately, including small structures and background.
    """
    reference, candidate = Path(reference).resolve(), Path(candidate).resolve()
    tolerances = tolerances or {}
    for name, value in tolerances.items():
        if not isinstance(value, dict) or set(value) - {"atol", "rtol"}:
            raise ValueError(f"Invalid tolerance for {name}")
        for threshold in value.values():
            if not np.isfinite(threshold) or threshold < 0:
                raise ValueError(f"Tolerance must be finite and nonnegative: {name}")
    if tolerances.get("coordinates", {}).get("rtol", 0) != 0:
        raise ValueError("Coordinates require an absolute displacement tolerance; rtol must be zero")
    if max_outlier_ids < 1 or not (0 <= min_label_dice <= 1 and 0 <= min_annotation_dice <= 1):
        raise ValueError("Invalid outlier limit or Dice threshold")

    def tolerance(name):
        value = tolerances.get(name, {})
        return {"atol": float(value.get("atol", 0)), "rtol": float(value.get("rtol", 0))}

    checks, surfaces = {}, {}

    def run(relative, operation, dependencies=()):
        paths = reference / relative, candidate / relative
        missing = [name for name, path in zip(("reference", "candidate"), paths) if not path.is_file()]
        if missing:
            checks[relative] = {"status": "missing", "missing_in": missing}
        elif any(not checks[key].get("vertex_correspondence_compatible", False) for key in dependencies):
            checks[relative] = {"status": "blocked", "reason": "Native vertex correspondence unavailable",
                                "dependencies": list(dependencies)}
        else:
            try:
                checks[relative] = operation(*paths)
            except (OSError, ValueError, KeyError, IndexError, TypeError, EOFError,
                    struct.error, nib.filebasedimages.ImageFileError) as error:
                checks[relative] = {"status": "error", "reason": str(error)}

    for hemi in ("lh", "rh"):
        for surface in SURFACES:
            key = f"surf/{hemi}.{surface}"

            def compare_surface(r, c, surface=surface, key=key):
                pair = fs.read_geometry(r), fs.read_geometry(c)
                surfaces[key] = pair
                check = _surface(*pair, tolerance("coordinates"), max_outlier_ids)
                if surface != "white":
                    white = surfaces.get(f"surf/{hemi}.white")
                    same = white is not None and all(
                        len(mesh[0]) == len(anchor[0]) and np.array_equal(mesh[1], anchor[1])
                        for mesh, anchor in zip(pair, white))
                    check["ordered_topology_matches_white"] = same
                    if not same:
                        check.update(status="failed", vertex_correspondence_compatible=False,
                                     reason="Surface vertex indexing/topology does not match white")
                return check

            run(key, compare_surface)

        for metric, dependencies in MAP_SURFACES.items():
            key = f"surf/{hemi}.{metric}"
            anchors = tuple(f"surf/{hemi}.{name}" for name in dependencies)

            def compare_map(r, c, metric=metric, anchors=anchors):
                rv, cv = fs.read_morph_data(r), fs.read_morph_data(c)
                n = len(surfaces[anchors[0]][0][0])
                if rv.shape != (n,) or cv.shape != (n,):
                    raise ValueError("Measurement vertex count does not match its surface")
                check = _numeric(rv, cv, tolerance(metric), max_outlier_ids)
                check["id_space"] = "native vertex index"
                return check

            run(key, compare_map, anchors)

        white_key = f"surf/{hemi}.white"
        for annotation in ANNOTATIONS:
            run(f"label/{hemi}.{annotation}.annot", lambda r, c: _annotation(
                r, c, len(surfaces[white_key][0][0]), min_annotation_dice, max_outlier_ids), (white_key,))

        def compare_cortex(r, c):
            n = len(surfaces[white_key][0][0])
            pair = [np.asarray(fs.read_label(path), dtype=np.int64) for path in (r, c)]
            if any(np.any((ids < 0) | (ids >= n)) for ids in pair):
                raise ValueError("Invalid cortex.label vertex IDs: outside [0, nvertices)")
            # Official labels can repeat IDs. MRISlabel2Mask sets membership
            # idempotently; also preserve multiplicities to detect altered files.
            counts = [np.bincount(ids, minlength=n) for ids in pair]
            exact = {"atol": 0.0, "rtol": 0.0}
            check = _numeric(*counts, exact, max_outlier_ids)
            check["comparison"] = "per-vertex entry multiplicity"
            check["membership"] = _numeric(*(count > 0 for count in counts), exact, max_outlier_ids)
            for side, ids, count in zip(("reference", "candidate"), pair, counts):
                unique = int(np.count_nonzero(count))
                check[side] = {"entries": len(ids), "unique_vertices": unique,
                               "duplicate_entries": len(ids) - unique,
                               "repeated_vertex_count": int(np.count_nonzero(count > 1))}
            check["id_space"] = "native vertex index"
            return check

        run(f"label/{hemi}.cortex.label", compare_cortex, (white_key,))
        for suffix in ("aparc.stats", "aparc.pial.stats", "aparc.DKTatlas.stats", "aparc.a2009s.stats"):
            run(f"stats/{hemi}.{suffix}", lambda r, c: _stats(r, c, tolerance, max_outlier_ids))

    for volume in VOLUMES:
        def compare_volume(r, c):
            ri, ci = nib.load(r), nib.load(c)
            geometry = ri.shape == ci.shape and np.array_equal(ri.affine, ci.affine)
            if not geometry:
                return {"status": "failed", "reason": "Volume grid/affine differs; no resampling performed",
                        "reference_shape": list(ri.shape), "candidate_shape": list(ci.shape),
                        "reference_affine": ri.affine.tolist(), "candidate_affine": ci.affine.tolist()}
            check = _labels(np.asarray(ri.dataobj), np.asarray(ci.dataobj), min_label_dice, max_outlier_ids)
            voxel_volume = float(abs(np.linalg.det(ri.affine[:3, :3])))
            check["voxel_volume_mm3"] = voxel_volume
            for row in check["labels"].values():
                row["reference_volume_mm3"] = row["reference_count"] * voxel_volume
                row["candidate_volume_mm3"] = row["candidate_count"] * voxel_volume
            check["id_space"] = "voxel indices"
            return check

        run(f"mri/{volume}", compare_volume)
    run("stats/aseg.stats", lambda r, c: _stats(r, c, tolerance, max_outlier_ids))
    run("stats/synthseg.vol.csv", lambda r, c: _synthseg_volumes(
        r, c, tolerance("synthseg.vol"), max_outlier_ids))
    passed = all(check["status"] == "passed" for check in checks.values())
    return {"schema_version": 1, "reference": str(reference), "candidate": str(candidate),
            "passed": passed, "tolerances": tolerances, "unspecified_tolerances": "exact",
            "checks": checks, "failed_checks": [key for key, value in checks.items() if value["status"] != "passed"],
            "not_assessed": ["independent generation/provenance", "surface self-intersections",
                             "upstream conform and intermediate stages", "runtime/speedup"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tolerances", type=Path, help="JSON object: metric -> {atol, rtol}; others stay exact")
    parser.add_argument("--min-label-dice", type=float, default=1.0)
    parser.add_argument("--min-annotation-dice", type=float, default=1.0)
    parser.add_argument("--max-outlier-ids", type=int, default=100)
    args = parser.parse_args(argv)
    report = compare_subject(args.reference, args.candidate,
                             tolerances=json.loads(args.tolerances.read_text()) if args.tolerances else None,
                             min_label_dice=args.min_label_dice, min_annotation_dice=args.min_annotation_dice,
                             max_outlier_ids=args.max_outlier_ids)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"{'PASS' if report['passed'] else 'FAIL'}: {len(report['failed_checks'])} failed/missing/blocked checks; {args.output}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
