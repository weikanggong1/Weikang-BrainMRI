"""Frozen fs_sub01 voxel gates for the Python WM aseg core and fixed options."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_dilation

from fnit.recon_all.edit_wm_aseg_core_python import (
    _add_aseg_wm_below_hippocampus,
    _edit_until_propagation,
    _post_spackle_early,
    _post_spackle_late,
    remove_paths_to_cortex,
    spackle_wm_superior_to_mtl,
)
from fnit.recon_all.edit_wm_aseg_late_python import apply_late_wm_edits


INPUT_SHA = {
    "wm.seg.mgz": "3cf79694f8e13889395e0e2a6e38508627ed1766bedecd751e76cece7884cd9f",
    "brain.mgz": "1b7360d069b76c8a5a63f93f3c296dddeb4db725c4e2625e11ebfc156279f401",
    "aseg.presurf.mgz": "263653b66a9aacba4d0d704781c6e67e581c86a9f99374e37d7e3f28d1674ce2",
    "entowm.mgz": "8bd292fc03e6a4a32ba9bedaa47f7e84e83e9594975df02f16f7fbd8cd9ca264",
}


def _read(path: Path) -> np.ndarray:
    return np.asarray(nib.load(str(path)).dataobj)


def _check(stage: str, actual: np.ndarray, expected: np.ndarray, results: dict) -> None:
    count = int(np.count_nonzero(actual != expected))
    results[stage] = {
        "different_voxels": count,
        "total_voxels": int(actual.size),
        "voxel_sha256": hashlib.sha256(actual.tobytes(order="F")).hexdigest(),
    }
    if count:
        raise AssertionError(f"{stage}: {count} voxels differ")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mri_dir", type=Path)
    parser.add_argument("reference_dir", type=Path)
    args = parser.parse_args()
    for name, sha in INPUT_SHA.items():
        actual = hashlib.sha256((args.mri_dir / name).read_bytes()).hexdigest()
        if actual != sha:
            raise ValueError(f"Input changed: {name}, SHA-256 {actual}")
    log = (args.reference_dir / "core.log").read_text()
    if "0 voxels added to wm to prevent paths from MTL structures to cortex" not in log:
        raise AssertionError("Native remove_paths_to_cortex did not report zero changes")

    wm = _read(args.mri_dir / "wm.seg.mgz").astype(np.uint8)
    brain = _read(args.mri_dir / "brain.mgz").astype(np.uint8)
    aseg = _read(args.mri_dir / "aseg.presurf.mgz").astype(np.int32)
    entowm = _read(args.mri_dir / "entowm.mgz")
    findings: dict = {"input_sha256": INPUT_SHA.copy(), "seconds": {}}
    start = time.monotonic()
    result, path_edits = remove_paths_to_cortex(wm, brain, aseg)
    findings["seconds"]["remove_paths_to_cortex"] = time.monotonic() - start
    findings["path_edits"] = path_edits
    _check("after_paths", result, wm, findings)
    for side, labels in (("lh", (17, 18, 5)), ("rh", (53, 54, 44))):
        diag_file = args.reference_dir / "path_diag" / f"{side}_roi.mgz"
        if diag_file.exists():
            roi = binary_dilation(np.isin(aseg, labels),
                                  structure=np.ones((3, 3, 3), bool), iterations=5)
            _check(f"{side}_native_roi", roi, _read(diag_file) > 0, findings)
    steps = (
        ("after_propagation", lambda: _edit_until_propagation(result, aseg)),
        ("after_early_mtl_spackle", lambda: _post_spackle_early(result, brain, aseg)),
        ("after_late_mtl_spackle", lambda: _post_spackle_late(result, aseg)),
        ("after_aseg_wm_below_hippocampus", lambda: _add_aseg_wm_below_hippocampus(result, aseg)),
    )
    for name, step in steps:
        start = time.monotonic()
        step()
        findings["seconds"][name] = time.monotonic() - start
        if name == "after_propagation":
            expected = np.fromfile(args.reference_dir / "after_propagation.raw", dtype=np.uint8)
            expected = expected.reshape(wm.shape, order="F")
            _check(name, result, expected, findings)
        elif name == "after_aseg_wm_below_hippocampus":
            expected = np.fromfile(args.reference_dir / "edit_only.raw", dtype=np.uint8)
            expected = expected.reshape(wm.shape, order="F")
            _check(name, result, expected, findings)
    start = time.monotonic()
    core = spackle_wm_superior_to_mtl(result, aseg)
    findings["seconds"]["final_spackle"] = time.monotonic() - start
    _check("core", core, _read(args.reference_dir / "core.mgz"), findings)
    start = time.monotonic()
    full = apply_late_wm_edits(core, aseg, entowm, wm, fill_seg_wm=True)
    findings["seconds"]["fixed_options"] = time.monotonic() - start
    _check("full", full, _read(args.reference_dir / "full.mgz"), findings)
    _check("original_recon_all", full, _read(args.mri_dir / "wm.asegedit.mgz"), findings)
    print(json.dumps(findings, indent=2))


if __name__ == "__main__":
    main()
