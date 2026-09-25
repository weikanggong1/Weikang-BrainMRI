"""Bounded first-candidate MRI-match probe; not a validated topology replacement."""

from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import nibabel as nib
import nibabel.freesurfer.io as fsio
import numpy as np
from fnit.recon_all.topology_defect_mri_match import defect_mri_match


def _hist_mean(path):
    bins, counts = np.loadtxt(path).T
    bins = bins.astype(np.float32)
    counts = counts.astype(np.float32)
    maximum = np.max(counts)
    total = mean = np.float32(0)
    for value, count in zip(bins, counts):
        if count < np.float32(maximum / np.float32(100)):
            continue
        # This path leaves HISTOGRAM.bin_size at zero after HISTOalloc.
        mean = np.float32(mean + np.float32(value * count))
        total = np.float32(total + count)
    return float(np.float32(mean / total))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--patch-report', type=Path, required=True)
    ap.add_argument('--base', type=Path, required=True)
    ap.add_argument('--brain', type=Path, required=True)
    ap.add_argument('--histograms', type=Path, required=True)
    ap.add_argument('--report', type=Path, required=True)
    args = ap.parse_args()
    patch = json.loads(args.patch_report.read_text())
    inside = np.asarray(patch['first_candidate_inside_vertex_indices'], np.int32)
    rows = patch['first_candidate_inside_neighbor_rows']
    xyz, faces = fsio.read_geometry(str(args.base))
    native, native_faces = fsio.read_geometry(str(args.base) + 'm')
    assert np.array_equal(faces, native_faces)
    mri = nib.load(str(args.brain))
    volume = np.ascontiguousarray(mri.dataobj, np.uint8)
    inv = np.linalg.inv(mri.header.get_vox2ras_tkr())
    wm = _hist_mean(args.histograms / 'w.plt')
    gm = _hist_mean(args.histograms / 'g.plt')
    start = perf_counter()
    predicted = defect_mri_match(xyz, faces, inside, rows, volume, inv, wm, gm)
    seconds = perf_counter() - start
    delta = np.linalg.norm(predicted.astype(np.float64) - native.astype(np.float64), axis=1)
    bitwise = np.all(predicted.view(np.uint32) == native.astype(np.float32).view(np.uint32), axis=1)
    result = {'hemisphere': patch['hemisphere'], 'scope': 'experimental first-candidate MRI match from native select0s', 'neighbor_order': 'runtime_patch',
              'white_mean_from_native_histogram': wm, 'gray_mean_from_native_histogram': gm,
              'intensity_histogram_bin_size': 0.0,
              'input_sha256': {name: sha256(path.read_bytes()).hexdigest() for name, path in {
                  'patch_report': args.patch_report, 'select0s': args.base,
                  'select0sm': Path(str(args.base) + 'm'), 'brain': args.brain,
                  'white_histogram': args.histograms / 'w.plt',
                  'gray_histogram': args.histograms / 'g.plt',
              }.items()},
              'python_seconds_including_jit': seconds, 'inside': len(inside),
              'all_vertices_bitwise_equal': int(np.sum(bitwise)),
              'inside_bitwise_equal': int(np.sum(bitwise[inside])),
              'outside_bitwise_equal': int(np.sum(bitwise)) - int(np.sum(bitwise[inside])),
              'inside_max_distance_mm': float(np.max(delta[inside])),
              'inside_mean_distance_mm': float(np.mean(delta[inside])),
              'inside_first_mismatch': int(inside[np.flatnonzero(~bitwise[inside])[0]]) if np.any(~bitwise[inside]) else None}
    args.report.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
