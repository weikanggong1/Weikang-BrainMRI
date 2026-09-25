"""Compare the Python aseg fill against an isolated reference volume."""

import argparse
import hashlib
import json
import time

import nibabel as nib
import numpy as np

from fnit.recon_all.fill_aseg_python import fill_with_aseg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("wm")
    parser.add_argument("aseg")
    parser.add_argument("reference")
    parser.add_argument("--cc-cut-mask")
    args = parser.parse_args()

    wm_image = nib.load(args.wm)
    wm = np.asarray(wm_image.dataobj)
    aseg = np.asarray(nib.load(args.aseg).dataobj)
    reference = np.asarray(nib.load(args.reference).dataobj)
    cut = np.asarray(nib.load(args.cc_cut_mask).dataobj) if args.cc_cut_mask else None
    start = time.perf_counter()
    filled = fill_with_aseg(wm, aseg, float(wm_image.header["delta"][0]), cut)
    seconds = time.perf_counter() - start
    if filled.shape != reference.shape:
        raise ValueError("reference shape differs from output")
    print(json.dumps({
        "seconds_algorithm": seconds,
        "mismatch_voxels": int(np.count_nonzero(filled != reference)),
        "total_voxels": int(filled.size),
        "voxel_sha256_fortran": hashlib.sha256(filled.tobytes(order="F")).hexdigest(),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
