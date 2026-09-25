"""Check whether float32 output storage resolves SynthSeg MGH metadata parity."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import surfa as sf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    volume = sf.load_volume(str(args.candidate))
    volume.new(volume.data.astype(np.float32)).save(str(args.output))
    ref, got = nib.load(str(args.reference)), nib.load(str(args.output))
    a, b = np.asarray(ref.dataobj), np.asarray(got.dataobj)
    raw_a = gzip.decompress(args.reference.read_bytes())
    raw_b = gzip.decompress(args.output.read_bytes())
    end = 284 + a.size * a.dtype.itemsize
    report = {
        "reference_dtype": str(a.dtype), "converted_dtype": str(b.dtype),
        "voxel_mismatches": int(np.count_nonzero(a != b)),
        "affine_max_abs": float(np.max(np.abs(ref.affine - got.affine))),
        "mgh_header_equal": raw_a[:284] == raw_b[:284],
        "mgh_header_and_payload_equal": raw_a[:end] == raw_b[:end],
        "trailer_equal": raw_a[end:] == raw_b[end:],
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
