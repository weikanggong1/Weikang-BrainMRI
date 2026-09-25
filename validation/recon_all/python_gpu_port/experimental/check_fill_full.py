"""Compare a complete Python mri_fill result with the fixed official reference."""

import argparse
import gzip
import hashlib
import json
import struct

import nibabel as nib
import numpy as np

from fnit.recon_all.fill_cutting_plane_python import _colortable_tag


def _tags(footer: bytes, ctab: bytes) -> list[tuple[int, bytes]]:
    tags = []
    cursor = 20
    while cursor < len(footer):
        tag = struct.unpack_from(">i", footer, cursor)[0]
        if tag == 1:
            end = cursor + len(ctab)
        else:
            length = struct.unpack_from(">q", footer, cursor + 4)[0]
            end = cursor + 12 + length
        tags.append((tag, footer[cursor:end]))
        cursor = end
    return tags


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate")
    parser.add_argument("reference")
    parser.add_argument("colortable")
    parser.add_argument("--cut-log")
    parser.add_argument("--reference-cut-log")
    args = parser.parse_args()
    candidate = nib.load(args.candidate)
    reference = nib.load(args.reference)
    values = np.asanyarray(candidate.dataobj)
    expected = np.asanyarray(reference.dataobj)
    raw = gzip.decompress(open(args.candidate, "rb").read())
    raw_ref = gzip.decompress(open(args.reference, "rb").read())
    footer = raw[284 + values.size:]
    footer_ref = raw_ref[284 + expected.size:]
    ctab = _colortable_tag(args.colortable)
    ctab_offset = footer_ref.find(b"\0\0\0\1\xff\xff\xff\xfe")
    candidate_tags = _tags(footer, ctab)
    reference_tags = _tags(footer_ref, ctab)
    result = {
        "voxel_mismatches": int(np.count_nonzero(values != expected)),
        "total_voxels": int(values.size),
        "voxel_sha256_fortran": hashlib.sha256(values.tobytes(order="F")).hexdigest(),
        "mgh_header_exact": raw[:284] == raw_ref[:284],
        "scan_parameters_exact": footer[:20] == footer_ref[:20],
        "ctab_tag_exact": ctab in footer and footer_ref[ctab_offset:ctab_offset + len(ctab)] == ctab,
        "ctab_tag_bytes": len(ctab),
        "affine_exact": bool(np.array_equal(candidate.affine, reference.affine)),
        "dtype_exact": candidate.get_data_dtype() == reference.get_data_dtype(),
        "non_command_tags_exact": ([block for tag, block in candidate_tags if tag != 3]
                                   == [block for tag, block in reference_tags if tag != 3]),
        "candidate_command_tags": sum(tag == 3 for tag, _ in candidate_tags),
        "reference_command_tags": sum(tag == 3 for tag, _ in reference_tags),
        "candidate_non_command_tags": [(tag, len(block), hashlib.sha256(block).hexdigest()[:12])
                                       for tag, block in candidate_tags if tag != 3],
        "reference_non_command_tags": [(tag, len(block), hashlib.sha256(block).hexdigest()[:12])
                                       for tag, block in reference_tags if tag != 3],
    }
    if args.cut_log:
        result["cut_log_exact"] = open(args.cut_log, "rb").read() == open(args.reference_cut_log, "rb").read()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
