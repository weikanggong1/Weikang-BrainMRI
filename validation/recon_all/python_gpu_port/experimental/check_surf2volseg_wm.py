"""Compare a Python wmparc output with fixed native mri_surf2volseg outputs."""

import argparse
import gzip
import hashlib
import json

import nibabel as nib
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("candidate")
    parser.add_argument("reference")
    parser.add_argument("rerun")
    args = parser.parse_args()
    source = np.asanyarray(nib.load(args.source).dataobj)
    image = nib.load(args.candidate)
    candidate = np.asanyarray(image.dataobj)
    expected_image = nib.load(args.reference)
    expected = np.asanyarray(expected_image.dataobj)
    rerun = np.asanyarray(nib.load(args.rerun).dataobj)
    raw = gzip.decompress(open(args.candidate, "rb").read())
    raw_ref = gzip.decompress(open(args.reference, "rb").read())
    diff = candidate != expected
    print(json.dumps({
        "total_voxels": int(candidate.size),
        "native_rerun_mismatches": int(np.count_nonzero(rerun != expected)),
        "python_mismatches": int(np.count_nonzero(diff)),
        "python_difference_by_input_label": {str(int(label)): int(np.count_nonzero(diff & (source == label)))
                                              for label in np.unique(source[diff])},
        "voxel_sha256_fortran": hashlib.sha256(candidate.tobytes(order="F")).hexdigest(),
        "decompressed_mgh_exact": raw == raw_ref,
        "mgh_header_exact": raw[:284] == raw_ref[:284],
        "affine_exact": bool(np.array_equal(image.affine, expected_image.affine)),
        "dtype_exact": image.get_data_dtype() == expected_image.get_data_dtype(),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
