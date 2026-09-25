"""Check the first native ``mri_normalize -aseg`` control checkpoint."""

import argparse
from pathlib import Path
import sys
import time

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from fnit.recon_all.normalization.normalize_aseg_ridge import medial_ridge
from fnit.recon_all.normalization.normalize_aseg_source import (
    apply_initial_aseg_bias, filter_aseg_ridge, prepare_aseg_source,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mri_dir", type=Path)
    parser.add_argument("native_sc", type=Path)
    parser.add_argument("--native-dc", type=Path)
    parser.add_argument("--native-out", type=Path)
    parser.add_argument("--native-norm1", type=Path)
    parser.add_argument("--check-independent-ridge", action="store_true")
    args = parser.parse_args()
    images = [nib.load(str(args.mri_dir / name)) for name in
              ("norm.mgz", "brainmask.mgz", "aseg.presurf.mgz")]
    if any(image.shape != images[0].shape or
           not np.array_equal(image.affine, images[0].affine) for image in images[1:]):
        raise ValueError("the fixed-input identity-resampling gate requires equal geometry")
    norm, mask, aseg = (np.asarray(image.dataobj) for image in images)
    masked, controls = prepare_aseg_source(norm, mask, aseg)
    reference = np.asarray(nib.load(str(args.native_sc)).dataobj)
    mismatches = int(np.count_nonzero(controls != reference))
    report = {"shape": controls.shape, "seed_voxels": int(np.count_nonzero(controls)),
              "sc_voxel_mismatches": mismatches,
              "source_voxels_removed_by_mask": int(np.count_nonzero((norm != 0) & (mask == 0))),
              "masked_source_dtype": str(masked.dtype)}
    if args.native_dc or args.native_out:
        if not args.native_dc or not args.native_out:
            parser.error("--native-dc and --native-out must be supplied together")
        native_dc = np.asarray(nib.load(str(args.native_dc)).dataobj)
        native_out = np.asarray(nib.load(str(args.native_out)).dataobj)
        # The native outlier map marks exactly the controls removed from the ridge.
        native_ridge = (native_dc != 0) | (native_out != 0)
        if args.check_independent_ridge:
            start = time.monotonic()
            ridge, details = medial_ridge(aseg)
            ridge_mismatches = int(np.count_nonzero(ridge != native_ridge))
            independent_dc, independent_out, independent_peak = filter_aseg_ridge(masked, ridge)
            report.update({"ridge_seconds": round(time.monotonic() - start, 3),
                           "ridge_details": details,
                           "ridge_voxel_mismatches": ridge_mismatches,
                           "ridge_false_positive": int(np.count_nonzero((ridge != 0) & ~native_ridge)),
                           "ridge_false_negative": int(np.count_nonzero((ridge == 0) & native_ridge)),
                           "independent_dc_voxel_mismatches": int(np.count_nonzero(independent_dc != native_dc)),
                           "independent_out_voxel_mismatches": int(np.count_nonzero(independent_out != native_out)),
                           "independent_wm_peak": independent_peak})
            if args.native_norm1:
                native_norm1 = np.asarray(nib.load(str(args.native_norm1)).dataobj)
                independent_norm1 = apply_initial_aseg_bias(masked, independent_dc)
                report["independent_norm1_voxel_mismatches"] = int(np.count_nonzero(independent_norm1 != native_norm1))
            mismatches += ridge_mismatches
        got_dc, got_out, peak = filter_aseg_ridge(masked, native_ridge)
        report.update({"native_ridge_voxels": int(np.count_nonzero(native_ridge)),
                       "wm_peak": peak,
                       "dc_voxel_mismatches": int(np.count_nonzero(got_dc != native_dc)),
                       "out_voxel_mismatches": int(np.count_nonzero(got_out != native_out))})
        mismatches += report["dc_voxel_mismatches"] + report["out_voxel_mismatches"]
        if args.native_norm1:
            initial = apply_initial_aseg_bias(masked, got_dc)
            native_norm1 = np.asarray(nib.load(str(args.native_norm1)).dataobj)
            report["norm1_voxel_mismatches"] = int(np.count_nonzero(initial != native_norm1))
            mismatches += report["norm1_voxel_mismatches"]
    elif args.native_norm1:
        parser.error("--native-norm1 requires --native-dc and --native-out")
    print(report)
    return int(mismatches != 0)


if __name__ == "__main__":
    raise SystemExit(main())
