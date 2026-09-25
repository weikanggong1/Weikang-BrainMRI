"""Run the connected Python N4 stage against preserved same-T1 subjects."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import time

import nibabel as nib
import numpy as np

from fnit.recon_all.n4_sitk import correct_volume
from fnit.recon_all.n4_wrapper import make_nu, normalize_n4_footer


WORK = Path('/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work')
BASE = WORK / 'reconall_python_gpu_20260925'
INPUT = BASE / 'connected_input_talairach_cpu_20260926'
OUTPUT = BASE / 'connected_n4_cpu_20260926'
REFERENCES = {
    'official': WORK / 'reconall_benchmark_pair_ac_20260924/official_subjects/a_official',
    'hybrid': WORK / 'reconall_main_20260924/single_subjects/fs_sub01',
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(candidate: Path, reference: Path) -> dict:
    c, r = nib.load(str(candidate)), nib.load(str(reference))
    cv, rv = np.asarray(c.dataobj), np.asarray(r.dataobj)
    diff = cv.astype(np.int16) - rv.astype(np.int16)
    c_raw, r_raw = gzip.decompress(candidate.read_bytes()), gzip.decompress(reference.read_bytes())
    end = 284 + cv.size * cv.dtype.itemsize
    return {
        'candidate_sha256': sha256(candidate), 'reference_sha256': sha256(reference),
        'dtype_equal': cv.dtype == rv.dtype, 'shape_equal': cv.shape == rv.shape,
        'voxel_mismatch_count': int(np.count_nonzero(diff)),
        'max_abs_voxel_difference': int(np.max(np.abs(diff))),
        'affine_max_abs_mm': float(np.max(np.abs(c.affine - r.affine))),
        'mgh_header_equal': c_raw[:284] == r_raw[:284],
        'mgh_header_and_payload_equal': c_raw[:end] == r_raw[:end],
        'decompressed_bytes_equal': c_raw == r_raw,
        'difference_histogram': {str(int(k)): int(v) for k, v in zip(*np.unique(diff[diff != 0], return_counts=True))},
    }


def main() -> None:
    OUTPUT.mkdir(exist_ok=True)
    orig = INPUT / 'mri/orig.mgz'
    tal = INPUT / 'mri/transforms/talairach.xfm'
    nu0, nu = OUTPUT / 'nu0.mgz', OUTPUT / 'nu.mgz'
    started = time.perf_counter()
    correct_volume(orig, nu0)
    n4_seconds = time.perf_counter() - started
    normalize_n4_footer(nu0)
    started = time.perf_counter()
    scale, bins = make_nu(orig, nu0, tal, nu)
    wrapper_seconds = time.perf_counter() - started
    comparisons = {name: compare(nu, root / "mri/nu.mgz") for name, root in REFERENCES.items()}
    fresh_native = Path("/tmp/reconall_n4_sub01_20260925/fresh_full/nu.mgz")
    if fresh_native.is_file():
        comparisons["fresh_native_same_host"] = compare(nu, fresh_native)
    report = {
        'candidate_source': 'fnit.recon_all.n4_sitk + n4_wrapper',
        'input_orig_sha256': sha256(orig), 'input_talairach_sha256': sha256(tal),
        'n4_seconds': n4_seconds, 'wrapper_seconds': wrapper_seconds,
        'scale': scale, 'histogram_bins': bins,
        'comparison': comparisons,
    }
    (OUTPUT / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
