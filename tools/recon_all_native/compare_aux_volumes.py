#!/usr/bin/env python3
"""Compare FreeSurfer cortical ribbon and white-matter parcellation volumes."""

import argparse
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np


VOLUMES = ("ribbon.mgz", "wmparc.mgz")
MIN_DICE = 0.995
MIN_MACRO_DICE = 0.999


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compare_volume(reference, candidate, max_outlier_ids=100):
    result = {
        "status": "failed", "min_dice": MIN_DICE,
        "min_foreground_macro_dice": MIN_MACRO_DICE,
        "reference_sha256": sha256(reference) if reference.is_file() else None,
        "candidate_sha256": sha256(candidate) if candidate.is_file() else None,
    }
    if result["reference_sha256"] is None or result["candidate_sha256"] is None:
        result["status"] = "missing"
        return result
    try:
        ref, cand = nib.load(reference), nib.load(candidate)
        geometry = {
            "shape_equal": ref.shape == cand.shape,
            "affine_equal": bool(np.array_equal(ref.affine, cand.affine)),
            "header_zooms_equal": ref.header.get_zooms() == cand.header.get_zooms(),
        }
        result.update(geometry)
        if len(ref.shape) != 3 or not all(geometry.values()):
            result["reason"] = "3D volume shape, affine or header zooms differ"
            return result
        rv, cv = np.asarray(ref.dataobj), np.asarray(cand.dataobj)
        if not (np.isfinite(rv).all() and np.isfinite(cv).all()
                and np.equal(rv, np.rint(rv)).all()
                and np.equal(cv, np.rint(cv)).all()):
            result["reason"] = "Volume labels must be finite integers"
            return result
        matches = rv == cv
        bad = np.flatnonzero(~matches)
        rc = dict(zip(*np.unique(rv, return_counts=True)))
        cc = dict(zip(*np.unique(cv, return_counts=True)))
        intersection = dict(zip(*np.unique(rv[matches], return_counts=True)))
        labels = {}
        for label in sorted(rc.keys() | cc.keys()):
            nr, nc = int(rc.get(label, 0)), int(cc.get(label, 0))
            dice = 2 * int(intersection.get(label, 0)) / (nr + nc)
            labels[str(int(label))] = {
                "reference_count": nr, "candidate_count": nc,
                "present_in_both": nr > 0 and nc > 0, "dice": dice,
            }
        foreground = [row["dice"] for label, row in labels.items() if label != "0"]
        macro = float(np.mean(foreground)) if foreground else None
        result.update(
            labels=labels, foreground_macro_dice=macro,
            mismatch_count=int(len(bad)), agreement=float(matches.mean()),
            outlier_voxels=np.array(np.unravel_index(bad[:max_outlier_ids], rv.shape)).T.tolist(),
            outlier_voxels_truncated=len(bad) > max_outlier_ids,
        )
        if foreground and macro >= MIN_MACRO_DICE and all(
                row["present_in_both"] and row["dice"] >= MIN_DICE
                for row in labels.values()):
            result["status"] = "passed"
        return result
    except (OSError, ValueError, TypeError, nib.filebasedimages.ImageFileError) as error:
        result.update(status="error", reason=str(error))
        return result


def compare_aux_volumes(reference, candidate, max_outlier_ids=100):
    if max_outlier_ids < 1:
        raise ValueError("max_outlier_ids must be positive")
    reference, candidate = Path(reference).resolve(), Path(candidate).resolve()
    checks = {
        f"mri/{name}": compare_volume(reference / "mri" / name,
                                      candidate / "mri" / name, max_outlier_ids)
        for name in VOLUMES
    }
    failed = [name for name, check in checks.items() if check["status"] != "passed"]
    return {
        "reference": str(reference), "candidate": str(candidate),
        "passed": not failed, "checks": checks, "failed_checks": failed,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    report = compare_aux_volumes(args.reference, args.candidate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"{'PASS' if report['passed'] else 'FAIL'}: {args.output}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
