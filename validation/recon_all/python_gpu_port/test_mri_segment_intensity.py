"""Compare isolated Python mri_segment stages with native diagnostic MGZ files."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import gzip
import json
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import label

from fnit.recon_all.mri_segment import (
    detect_intensity_thresholds, histogram_segmentation, intensity_segmentation,
    mask_white_labels, median_curve_center, median_curve_segmentation,
    reclassify_border, recover_bright_white, remove_wrong_direction,
    filter_diagonal_morphology, remove_bright_nonwhite,
    remove_1d_structures, thin_strand_candidates, _strand_segments,
    _dilate_strand_segments, _closed_strand_volume, _thicken_strands_core,
    segment_white_matter, segment_white_matter_mgz)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("diagnostic_dir", type=Path)
    parser.add_argument("--histogram-limit", type=int, default=0)
    parser.add_argument("--median-limit", type=int, default=0)
    parser.add_argument("--curve-limit", type=int, default=0)
    parser.add_argument("--reclassify-limit", type=int, default=0)
    parser.add_argument("--bright", action="store_true")
    parser.add_argument("--direction", action="store_true")
    parser.add_argument("--rm1d", action="store_true")
    parser.add_argument("--bright-nonwm", action="store_true")
    parser.add_argument("--filter", action="store_true")
    parser.add_argument("--thin-candidates", action="store_true")
    parser.add_argument("--segments", action="store_true")
    parser.add_argument("--thicken-core", action="store_true")
    parser.add_argument("--thicken-full", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--full-output", type=Path)
    args = parser.parse_args()
    source_image = nib.load(str(args.source))
    source = np.asarray(source_image.dataobj).copy()
    tensor = torch.from_numpy(source)
    report = {"shape": list(source.shape), "dtype": str(source.dtype), "stages": {}}
    for stage, wm_low, gray_hi in ((1, 79.0, 99.0), (2, 77.5864, 87.7259)):
        start = time.perf_counter()
        actual = intensity_segmentation(tensor, wm_low=wm_low, wm_hi=125.0,
                                        gray_hi=gray_hi).numpy()
        elapsed = time.perf_counter() - start
        reference = np.asarray(nib.load(str(args.diagnostic_dir /
                                              f"wmseg.int.{stage}.mgz")).dataobj)
        values, counts = np.unique(actual, return_counts=True)
        report["stages"][str(stage)] = {
            "mismatched_voxels": int(np.count_nonzero(actual != reference)),
            "label_counts": dict(zip(map(str, values.tolist()), counts.tolist())),
            "seconds": elapsed,
        }
        if args.histogram_limit:
            points = np.argwhere(actual == 128)
            if args.histogram_limit > 0:
                points = points[:args.histogram_limit]
            start = time.perf_counter()
            histo = histogram_segmentation(tensor, torch.from_numpy(actual),
                                           wm_low=wm_low, wm_hi=125,
                                           gray_hi=gray_hi,
                                           limit=(args.histogram_limit if args.histogram_limit > 0
                                                  else None)).numpy()
            expected = np.asarray(nib.load(str(args.diagnostic_dir /
                                             f"wmseg.histo.{stage}.mgz")).dataobj)
            selected = tuple(points.T)
            pairs, pair_counts = np.unique(
                np.stack((histo[selected], expected[selected]), axis=1),
                axis=0, return_counts=True)
            mismatch = histo[selected] != expected[selected]
            report["stages"][str(stage)]["histogram_prefix"] = {
                "checked_voxels": len(points),
                "mismatched_voxels": int(np.count_nonzero(mismatch)),
                "prediction_reference_counts": {
                    f"{int(a)}->{int(b)}": int(count)
                    for (a, b), count in zip(pairs, pair_counts)},
                "first_mismatches": [
                    {"voxel": points[i].tolist(),
                     "intensity": int(source[tuple(points[i])]),
                     "predicted": int(histo[tuple(points[i])]),
                     "reference": int(expected[tuple(points[i])])}
                    for i in np.flatnonzero(mismatch)[:8]],
                "seconds": time.perf_counter() - start,
            }
    first_histo = np.asarray(nib.load(str(args.diagnostic_dir /
                                          "wmseg.histo.1.mgz")).dataobj).copy()
    report["detected_stats"] = asdict(detect_intensity_thresholds(
        tensor, torch.from_numpy(first_histo)))
    if args.median_limit:
        second_histo = np.asarray(nib.load(str(args.diagnostic_dir /
                                                   "wmseg.histo.2.mgz")).dataobj).copy()
        points = np.argwhere(second_histo == 128)
        if args.median_limit > 0:
            points = points[:args.median_limit]
        start = time.perf_counter()
        predicted = median_curve_center(tensor, torch.from_numpy(second_histo),
                                        gray_hi=87.7259, wm_low=77.5864,
                                        limit=(args.median_limit if args.median_limit > 0
                                               else None)).numpy()
        expected = np.asarray(nib.load(str(args.diagnostic_dir /
                                             "wmseg.medcurv.mgz")).dataobj)
        selected = tuple(points.T)
        decided = predicted[selected] != 128
        report["median_center"] = {
            "checked_voxels": len(points),
            "decided_voxels": int(decided.sum()),
            "mismatched_decided": int(np.count_nonzero(
                predicted[selected][decided] != expected[selected][decided])),
            "seconds": time.perf_counter() - start,
        }
    if args.curve_limit:
        second_histo = np.asarray(nib.load(str(args.diagnostic_dir /
                                                   "wmseg.histo.2.mgz")).dataobj).copy()
        points = np.argwhere(second_histo == 128)
        if args.curve_limit > 0:
            points = points[:args.curve_limit]
        start = time.perf_counter()
        predicted = median_curve_segmentation(tensor, torch.from_numpy(second_histo),
                                              gray_hi=87.7259, wm_low=77.5864,
                                              limit=(args.curve_limit if args.curve_limit > 0
                                                     else None)).numpy()
        expected = np.asarray(nib.load(str(args.diagnostic_dir /
                                             "wmseg.medcurv.mgz")).dataobj)
        selected = tuple(points.T)
        mismatch = predicted[selected] != expected[selected]
        report["median_curve"] = {
            "checked_voxels": len(points),
            "mismatched_voxels": int(mismatch.sum()),
            "first_mismatches": [
                {"voxel": points[i].tolist(),
                 "intensity": int(source[tuple(points[i])]),
                 "predicted": int(predicted[tuple(points[i])]),
                 "reference": int(expected[tuple(points[i])])}
                for i in np.flatnonzero(mismatch)[:8]],
            "seconds": time.perf_counter() - start,
        }
    if args.reclassify_limit:
        medcurv = np.asarray(nib.load(str(args.diagnostic_dir /
                                             "wmseg.medcurv.mgz")).dataobj).copy()
        points = np.argwhere((source > 72.58642578) & (source < 87.72589111))
        if args.reclassify_limit > 0:
            points = points[:args.reclassify_limit]
        start = time.perf_counter()
        predicted = reclassify_border(tensor, torch.from_numpy(medcurv),
                                      wm_low=72.58642578, gray_hi=87.72589111,
                                      limit=(args.reclassify_limit if args.reclassify_limit > 0
                                             else None)).numpy()
        expected = np.asarray(nib.load(str(args.diagnostic_dir /
                                             "wmseg.reclassified.mgz")).dataobj)
        selected = tuple(points.T)
        mismatch = predicted[selected] != expected[selected]
        report["reclassified"] = {
            "checked_voxels": len(points),
            "mismatched_voxels": int(mismatch.sum()),
            "first_mismatches": [
                {"voxel": points[i].tolist(),
                 "intensity": int(source[tuple(points[i])]),
                 "predicted": int(predicted[tuple(points[i])]),
                 "reference": int(expected[tuple(points[i])])}
                for i in np.flatnonzero(mismatch)[:8]],
            "seconds": time.perf_counter() - start,
        }
    if args.bright:
        reclassified = np.asarray(nib.load(str(args.diagnostic_dir /
                                                  "wmseg.reclassified.mgz")).dataobj).copy()
        start = time.perf_counter()
        masked = mask_white_labels(tensor, torch.from_numpy(reclassified))
        predicted = recover_bright_white(tensor, masked, wm_low=77.58642578,
                                         wm_hi=125, white_sigma=4.16194916).numpy()
        expected = np.asarray(nib.load(str(args.diagnostic_dir /
                                             "wmseg.recover-brightwm.mgz")).dataobj)
        report["recover_bright_white"] = {
            "mismatched_voxels": int(np.count_nonzero(predicted != expected)),
            "restored_voxels": int(np.count_nonzero(predicted != masked.numpy())),
            "seconds": time.perf_counter() - start,
        }
    if args.direction:
        recovered = np.asarray(nib.load(str(args.diagnostic_dir /
                                               "wmseg.recover-brightwm.mgz")).dataobj).copy()
        start = time.perf_counter()
        predicted = remove_wrong_direction(torch.from_numpy(recovered),
                                           low=72.58642578, high=87.72589111).numpy()
        expected = np.asarray(nib.load(str(args.diagnostic_dir /
                                             "wmseg.wrong-dir.mgz")).dataobj)
        report["wrong_direction"] = {
            "mismatched_voxels": int(np.count_nonzero(predicted != expected)),
            "removed_voxels": int(np.count_nonzero(recovered != predicted)),
            "seconds": time.perf_counter() - start,
        }
    if args.rm1d:
        wrong_direction = np.asarray(nib.load(str(args.diagnostic_dir /
                                                    "wmseg.wrong-dir.mgz")).dataobj).copy()
        start = time.perf_counter()
        predicted = remove_1d_structures(torch.from_numpy(wrong_direction)).numpy()
        expected = np.asarray(nib.load(str(args.diagnostic_dir /
                                             "wmseg.rm1d.mgz")).dataobj)
        report["remove_1d_structures"] = {
            "mismatched_voxels": int(np.count_nonzero(predicted != expected)),
            "removed_voxels": int(np.count_nonzero(wrong_direction != predicted)),
            "seconds": time.perf_counter() - start,
        }
    if args.bright_nonwm:
        thickened = np.asarray(nib.load(str(args.diagnostic_dir /
                                               "wmseg.thicken.mgz")).dataobj).copy()
        start = time.perf_counter()
        predicted = remove_bright_nonwhite(tensor, torch.from_numpy(thickened)).numpy()
        expected = np.asarray(nib.load(str(args.diagnostic_dir /
                                             "wmseg.bright-nonwm.mgz")).dataobj)
        report["bright_nonwm"] = {
            "mismatched_voxels": int(np.count_nonzero(predicted != expected)),
            "removed_voxels": int(np.count_nonzero(thickened != predicted)),
            "seconds": time.perf_counter() - start,
        }
    if args.filter:
        bright_nonwm = np.asarray(nib.load(str(args.diagnostic_dir /
                                                  "wmseg.bright-nonwm.mgz")).dataobj).copy()
        start = time.perf_counter()
        predicted = filter_diagonal_morphology(torch.from_numpy(bright_nonwm)).numpy()
        expected = np.asarray(nib.load(str(args.diagnostic_dir /
                                             "wmseg.filter.mgz")).dataobj)
        report["filter_morphology"] = {
            "mismatched_voxels": int(np.count_nonzero(predicted != expected)),
            "added_voxels": int(np.count_nonzero((bright_nonwm == 0) & (predicted != 0))),
            "seconds": time.perf_counter() - start,
        }
    if args.thin_candidates:
        rm1d = np.asarray(nib.load(str(args.diagnostic_dir /
                                           "wmseg.rm1d.mgz")).dataobj).copy()
        start = time.perf_counter()
        thin = thin_strand_candidates(torch.from_numpy(rm1d)).numpy()
        components, component_count = label(thin)
        sizes = np.bincount(components.ravel())
        largest = np.argmax(sizes[1:]) + 1
        expected = np.asarray(nib.load(str(args.diagnostic_dir /
                                             "seg845.mgh")).dataobj) > 0
        component = (components == largest) & (rm1d > 0)
        mismatches = np.argwhere(component != expected)
        report["thin_candidates"] = {
            "components": int(component_count),
            "largest_voxels": int(sizes[largest]),
            "reference_largest_voxels": int(expected.sum()),
            "largest_mismatched_voxels": len(mismatches),
            "seconds": time.perf_counter() - start,
        }
    if args.segments:
        rm1d = np.asarray(nib.load(str(args.diagnostic_dir /
                                           "wmseg.rm1d.mgz")).dataobj).copy()
        start = time.perf_counter()
        thin = thin_strand_candidates(torch.from_numpy(rm1d)).numpy()
        segments = _strand_segments(thin)
        largest = max(range(len(segments)), key=lambda i: len(segments[i]))
        segment_image = np.zeros(rm1d.shape, dtype=np.uint8)
        for point in segments[largest]:
            segment_image[point] = rm1d[point]
        initial = np.asarray(nib.load(str(args.diagnostic_dir /
                                            "seg845.mgh")).dataobj)
        report["strand_segments_initial"] = {
            "component_count": len(segments),
            "largest_index": largest,
            "largest_voxels": len(segments[largest]),
            "mismatched_voxels": int(np.count_nonzero(segment_image != initial)),
            "seconds": time.perf_counter() - start,
        }
        start = time.perf_counter()
        _dilate_strand_segments(segments, rm1d)
        largest = max(range(len(segments)), key=lambda i: len(segments[i]))
        segment_image.fill(0)
        for point in segments[largest]:
            segment_image[point] = rm1d[point]
        expected = np.asarray(nib.load(str(args.diagnostic_dir /
                                             "dilated_seg845.mgh")).dataobj)
        mismatch = np.argwhere(segment_image != expected)
        report["strand_segments_dilated"] = {
            "largest_index": largest,
            "largest_voxels": len(segments[largest]),
            "mismatched_voxels": len(mismatch),
            "first_mismatches": [{"voxel": point.tolist(),
                                  "predicted": int(segment_image[tuple(point)]),
                                  "reference": int(expected[tuple(point)])}
                                 for point in mismatch[:10]],
            "seconds": time.perf_counter() - start,
        }
    if args.thicken_core or args.thicken_full:
        rm1d = np.asarray(nib.load(str(args.diagnostic_dir /
                                           "wmseg.rm1d.mgz")).dataobj).copy()
        start = time.perf_counter()
        closed = _closed_strand_volume(rm1d)
        thin = thin_strand_candidates(torch.from_numpy(rm1d)).numpy()
        segments = _strand_segments(thin)
        _dilate_strand_segments(segments, rm1d)
        predicted, strand_images = _thicken_strands_core(
            source, rm1d, closed, segments, planar_holes=args.thicken_full)
        expected = np.asarray(nib.load(str(args.diagnostic_dir /
                                             "wmseg.thicken.mgz")).dataobj)
        report["thicken_full" if args.thicken_full else "thicken_core"] = {
            "mismatched_voxels": int(np.count_nonzero(predicted != expected)),
            "added_voxels": int(np.count_nonzero((rm1d == 0) & (predicted != 0))),
            "reference_added_voxels": int(np.count_nonzero((rm1d == 0) & (expected != 0))),
            "predicted_fill_counts": {str(value): int(np.count_nonzero(predicted == value))
                                      for value in (200, 210)},
            "reference_fill_counts": {str(value): int(np.count_nonzero(expected == value))
                                      for value in (200, 210)},
            "first_mismatches": [point.tolist()
                                 for point in np.argwhere(predicted != expected)[:8]],
            "seconds": time.perf_counter() - start,
        }
        final_strand = np.asarray(nib.load(str(args.diagnostic_dir / "thin.mgh")).dataobj)
        report["last_strand"] = {
            "mismatched_voxels": int(np.count_nonzero(strand_images[-1] != final_strand)),
            "predicted_voxels": int(np.count_nonzero(strand_images[-1])),
            "reference_voxels": int(np.count_nonzero(final_strand)),
        }
    if args.full:
        start = time.perf_counter()
        predicted = segment_white_matter(tensor).numpy()
        reference_image = nib.load(str(args.diagnostic_dir / "wm.seg.mgz"))
        expected = np.asarray(reference_image.dataobj)
        mismatches = np.argwhere(predicted != expected)
        report["full_pipeline"] = {
            "mismatched_voxels": len(mismatches),
            "first_mismatches": mismatches[:8].tolist(),
            "predicted_dtype": str(predicted.dtype),
            "reference_dtype": str(expected.dtype),
            "affine_max_abs_mm": float(np.max(np.abs(source_image.affine - reference_image.affine))),
            "header_bytes_equal_input_reference": source_image.header.binaryblock == reference_image.header.binaryblock,
            "seconds": time.perf_counter() - start,
        }
    if args.full_output:
        start = time.perf_counter()
        segment_white_matter_mgz(args.source, args.full_output)
        predicted_image = nib.load(str(args.full_output))
        reference_image = nib.load(str(args.diagnostic_dir / "wm.seg.mgz"))
        predicted = np.asarray(predicted_image.dataobj)
        expected = np.asarray(reference_image.dataobj)
        mismatches = np.argwhere(predicted != expected)
        with gzip.open(args.full_output, "rb") as stream:
            predicted_raw = stream.read()
        with gzip.open(args.diagnostic_dir / "wm.seg.mgz", "rb") as stream:
            reference_raw = stream.read()
        report["full_mgz"] = {
            "mismatched_voxels": len(mismatches),
            "first_mismatches": mismatches[:8].tolist(),
            "affine_max_abs_mm": float(np.max(np.abs(predicted_image.affine - reference_image.affine))),
            "header_284_bytes_equal": predicted_raw[:284] == reference_raw[:284],
            "voxel_bytes_equal": predicted_raw[284:284 + predicted.size] == reference_raw[284:284 + expected.size],
            "footer_bytes_equal": predicted_raw[284 + predicted.size:] == reference_raw[284 + expected.size:],
            "seconds": time.perf_counter() - start,
        }
    print(json.dumps(report, indent=2))
    exact = [stage["mismatched_voxels"] for stage in report["stages"].values()]
    exact += [stage["histogram_prefix"]["mismatched_voxels"]
              for stage in report["stages"].values() if "histogram_prefix" in stage]
    exact += [report[name]["mismatched_voxels"] for name in (
        "median_curve", "reclassified", "recover_bright_white", "wrong_direction",
        "remove_1d_structures", "bright_nonwm", "filter_morphology",
        "strand_segments_initial") if name in report]
    if "median_center" in report:
        exact.append(report["median_center"]["mismatched_decided"])
    if "thin_candidates" in report:
        exact.append(report["thin_candidates"]["largest_mismatched_voxels"])
    if "full_pipeline" in report:
        exact.append(report["full_pipeline"]["mismatched_voxels"])
    if "full_mgz" in report:
        exact.append(report["full_mgz"]["mismatched_voxels"])
        exact.append(not report["full_mgz"]["header_284_bytes_equal"])
        exact.append(not report["full_mgz"]["voxel_bytes_equal"])
    if any(exact):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
