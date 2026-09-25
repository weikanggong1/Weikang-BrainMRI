"""Compare an independent second mri_normalize output with a native reference."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np


def compare(candidate: Path, reference: Path) -> dict:
    images = [nib.load(str(path)) for path in (candidate, reference)]
    if images[0].shape != images[1].shape:
        raise ValueError("candidate and reference shapes differ")
    values = [np.asarray(image.dataobj) for image in images]
    raw = [gzip.decompress(path.read_bytes()) for path in (candidate, reference)]
    offset = int(images[0].header.get_data_offset())
    end = offset + values[0].size * values[0].dtype.itemsize
    differences = values[0].astype(np.int16) - values[1].astype(np.int16)
    return {
        "voxel_mismatches": int(np.count_nonzero(differences)),
        "voxel_count": int(values[0].size),
        "maximum_absolute_voxel_error": int(np.max(np.abs(differences))),
        "mgh_header_equal": raw[0][:offset] == raw[1][:offset],
        "voxel_payload_equal": raw[0][offset:end] == raw[1][offset:end],
        "decompressed_file_equal": raw[0] == raw[1],
        "candidate_trailer_bytes": len(raw[0]) - end,
        "reference_trailer_bytes": len(raw[1]) - end,
        "candidate_is_decompressed_prefix": raw[1].startswith(raw[0]),
        "candidate_decompressed_sha256": hashlib.sha256(raw[0]).hexdigest(),
        "reference_decompressed_sha256": hashlib.sha256(raw[1]).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = compare(args.candidate, args.reference)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    return int(result["voxel_mismatches"] != 0 or not result["mgh_header_equal"])


if __name__ == "__main__":
    raise SystemExit(main())
