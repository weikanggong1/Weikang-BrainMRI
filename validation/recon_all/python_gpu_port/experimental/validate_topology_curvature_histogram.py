"""Independent first-pass topology curvature fit and histogram comparison.

Native plots and optional first-curvature dump are reference outputs only.
They are read after the predicted curvature arrays and PDFs are complete.
"""

from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import nibabel as nib
import nibabel.freesurfer.io as fsio
import numpy as np

from fnit.recon_all.topology_curvature_histogram import curvature_histograms
from fnit.recon_all.topology_principal_curvature import first_principal_curvatures


def _hist_mean(path: Path, width: np.float32) -> tuple[float, float]:
    """Reconstruct the unique unrounded source bin origin from its plot."""
    printed, counts = np.loadtxt(path).T
    counts = counts.astype(np.float32)
    center = np.float32(printed[0])
    starts = [center]
    for direction in (np.float32(-np.inf), np.float32(np.inf)):
        value = center
        for _ in range(16):
            value = np.nextafter(value, direction)
            starts.append(value)
    matches = []
    for first in starts:
        bins = np.empty(len(printed), np.float32)
        value = first
        for index in range(len(bins)):
            bins[index] = value
            value = np.float32(value + width)
        if all(f"{float(value):.6f}" == f"{float(saved):.6f}"
               for value, saved in zip(bins, printed)):
            matches.append((first, bins))
    if len(matches) != 1:
        raise ValueError(f"expected one unrounded histogram origin: {path}, got {len(matches)}")
    first, bins = matches[0]
    peak = np.max(counts)
    total = weighted = np.float32(0)
    for value, count in zip(bins, counts):
        if count < np.float32(peak / np.float32(100)):
            continue
        offset = np.float32(value - np.float32(width / np.float32(2)))
        weighted = np.float32(weighted + np.float32(offset * count))
        total = np.float32(total + count)
    return float(np.float32(weighted / total)), float(first)


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def validate(orig: Path, sphere: Path, histograms: Path,
             native_curvature: Path | None, predicted_curvature: Path | None = None,
             expected_reference_report: Path | None = None) -> dict[str, object]:
    original, faces = fsio.read_geometry(str(orig))
    canonical, canonical_faces = fsio.read_geometry(str(sphere))
    if not np.array_equal(faces, canonical_faces):
        raise ValueError("orig and qsphere face order differs")
    start = perf_counter()
    fit = first_principal_curvatures(original, canonical, faces)
    predicted = curvature_histograms(fit["k1"], fit["k2"])
    if predicted_curvature is not None:
        np.savez_compressed(predicted_curvature, k1=fit["k1"], k2=fit["k2"],
                            defect_labels=fit["defect_labels"])
    seconds = perf_counter() - start
    reference = nib.load(str(histograms / "mri_k1_k2.mgh"))
    native_joint = np.asarray(reference.dataobj, np.float32)[:, :, 0]
    native_widths = tuple(np.float32(value) for value in reference.header.get_zooms()[:2])
    result: dict[str, object] = {
        "scope": "first global topology curvature histogram only; source-order VNL float SVD inverse",
        "input_sha256": {"orig": _digest(orig), "qsphere": _digest(sphere)},
        "native_reference_sha256": {name: _digest(histograms / name)
                                    for name in ("k1.plt", "k2.plt", "mri_k1_k2.mgh")},
        "vertices": len(original), "defect_vertices": int(np.count_nonzero(fit["defect_labels"])),
        "invalid_fits": int(np.count_nonzero(fit["invalid"])),
        "ill_conditioned_fits": int(np.count_nonzero(fit["ill_conditioned"])),
        "python_fit_and_histogram_seconds_including_jit": seconds,
        "first_candidate_qcurv_ll": "not validated: candidate patch k1/k2 still require an independent fit",
    }
    if expected_reference_report is not None:
        frozen = json.loads(expected_reference_report.read_text())
        if result["input_sha256"] != frozen["input_sha256"]:
            raise ValueError("input surface SHA-256 differs from frozen report")
        if result["native_reference_sha256"] != frozen["native_reference_sha256"]:
            raise ValueError("histogram SHA-256 differs from frozen report")
        if native_curvature is not None and _digest(native_curvature) != frozen["native_first_curvature_sha256"]:
            raise ValueError("vertex curvature SHA-256 differs from frozen report")
        result["frozen_reference_sha256_match"] = True
        result["frozen_reference_report"] = expected_reference_report.name
    else:
        result["frozen_reference_sha256_match"] = None
    for position, name in enumerate(("k1", "k2")):
        plot = np.loadtxt(histograms / f"{name}.plt")
        computed_bins = predicted[f"{name}_bins"]
        computed_pdf = predicted[f"{name}_counts"]
        printed_bins = [f"{float(value):.6f}" for value in computed_bins]
        printed_pdf = [f"{float(value):.10f}" for value in computed_pdf]
        native_bins = [f"{value:.6f}" for value in plot[:, 0]]
        native_pdf = [f"{value:.10f}" for value in plot[:, 1]]
        bin_different = [index for index in range(100) if printed_bins[index] != native_bins[index]]
        pdf_different = [index for index in range(100) if printed_pdf[index] != native_pdf[index]]
        native_mean, native_start = _hist_mean(histograms / f"{name}.plt", native_widths[position])
        native_raw = np.rint(plot[:, 1] * len(original)).astype(np.int64)
        source_index = np.clip(np.trunc(np.float32(
            np.float32(fit[name] - predicted[f"{name}_start"]) / predicted[f"{name}_width"]
        )).astype(np.int32), 0, 99)
        predicted_raw = np.bincount(source_index, minlength=100)
        raw_different = np.flatnonzero(predicted_raw != native_raw)
        result[name] = {
            "raw_count_matching_bins": 100 - len(raw_different),
            "first_raw_count_mismatch": int(raw_different[0]) if len(raw_different) else None,
            "raw_count_l1_difference": int(np.abs(predicted_raw - native_raw).sum()),
            "predicted_min_max": [float(np.min(fit[name])), float(np.max(fit[name]))],
            "predicted_mean": float(predicted[f"{name}_mean"]),
            "native_mean_from_unrounded_bins": native_mean,
            "mean_float32_bits_equal": bool(np.float32(predicted[f"{name}_mean"]).view(np.uint32)
                                            == np.float32(native_mean).view(np.uint32)),
            "mean_absolute_difference": abs(float(predicted[f"{name}_mean"]) - native_mean),
            "native_first_bin": native_start,
            "bin_positions_matching_six_decimals": 100 - len(bin_different),
            "pdf_counts_matching_ten_decimals": 100 - len(pdf_different),
            "first_bin_position_mismatch": bin_different[0] if bin_different else None,
            "first_pdf_count_mismatch": pdf_different[0] if pdf_different else None,
            "largest_pdf_absolute_difference": float(np.max(np.abs(computed_pdf.astype(np.float64) - plot[:, 1]))),
        }
    native_norm = np.float32(np.float32(0.1) / np.min(native_joint))
    native_raw_joint = np.rint(native_joint.astype(np.float64) * float(native_norm)).astype(np.int64)
    predicted_raw_joint = np.rint(predicted["joint"].astype(np.float64)
                                   * float(predicted["joint_norm"])).astype(np.int64)
    raw_joint_different = np.argwhere(native_raw_joint != predicted_raw_joint)
    native_match = predicted["joint"].view(np.uint32) == native_joint.view(np.uint32)
    joint_delta = np.abs(predicted["joint"].astype(np.float64) - native_joint.astype(np.float64))
    joint_different = np.argwhere(~native_match)
    result["joint"] = {
        "bitwise_matching_cells": int(np.count_nonzero(native_match)),
        "cells": 10000,
        "first_bitwise_mismatch": joint_different[0].tolist() if len(joint_different) else None,
        "largest_absolute_difference": float(np.max(joint_delta)),
        "predicted_norm": float(predicted["joint_norm"]),
        "native_norm_inferred_from_pseudocount": float(native_norm),
        "raw_count_matching_cells": 10000 - len(raw_joint_different),
        "first_raw_count_mismatch": raw_joint_different[0].tolist() if len(raw_joint_different) else None,
        "raw_count_l1_difference": int(np.abs(native_raw_joint - predicted_raw_joint).sum()),
    }
    if native_curvature is not None:
        dumped = np.loadtxt(native_curvature, dtype=np.float32)
        if len(dumped) != len(original) or not np.array_equal(dumped[:, 0], np.arange(len(original))):
            raise ValueError("native first-curvature dump vertex order differs")
        result["native_first_curvature_sha256"] = _digest(native_curvature)
        for column, name in ((1, "k1"), (2, "k2")):
            delta = np.abs(fit[name].astype(np.float64) - dumped[:, column].astype(np.float64))
            above = np.flatnonzero(delta > 1e-3)
            printed_equal = np.fromiter((f"{float(a):.6f}" == f"{float(b):.6f}"
                                         for a, b in zip(fit[name], dumped[:, column])),
                                        np.bool_, len(original))
            first_printed = np.flatnonzero(~printed_equal)
            result[name]["native_printed_six_decimal_equal"] = int(np.count_nonzero(printed_equal))
            result[name]["first_native_printed_six_decimal_difference"] = (
                int(first_printed[0]) if len(first_printed) else None)
            result[name]["first_native_printed_difference_values"] = (
                [float(fit[name][first_printed[0]]), float(dumped[first_printed[0], column])]
                if len(first_printed) else None)
            result[name]["native_within_1e_3"] = int(np.count_nonzero(delta <= 1e-3))
            result[name]["native_median_absolute_difference"] = float(np.median(delta))
            result[name]["native_largest_absolute_difference"] = float(np.max(delta))
            result[name]["first_native_vertex_difference_over_1e_3"] = int(above[0]) if len(above) else None
    result["histogram_matches_native"] = all(
        result[name]["bin_positions_matching_six_decimals"] == 100
        and result[name]["pdf_counts_matching_ten_decimals"] == 100
        and result[name]["mean_float32_bits_equal"] for name in ("k1", "k2")) and result["joint"]["bitwise_matching_cells"] == 10000
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orig", type=Path, required=True)
    parser.add_argument("--sphere", type=Path, required=True)
    parser.add_argument("--histograms", type=Path, required=True)
    parser.add_argument("--native-curvature", type=Path)
    parser.add_argument("--predicted-curvature", type=Path)
    parser.add_argument("--expected-reference-report", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = validate(args.orig, args.sphere, args.histograms,
                      args.native_curvature, args.predicted_curvature,
                      args.expected_reference_report)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
