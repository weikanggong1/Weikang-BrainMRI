"""Paired byte and vertex-ID validation of the fixed no-GA cortical label."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from fnit.recon_all.label_cortex_python import label_cortex


def label_ids(raw: bytes) -> np.ndarray:
    lines = raw.decode("ascii").splitlines()
    return np.fromiter((int(line.split()[0]) for line in lines[2:]), np.int32,
                       count=int(lines[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("surface", type=Path)
    parser.add_argument("aseg", type=Path)
    parser.add_argument("native", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    start = time.perf_counter()
    selected = label_cortex(args.surface, args.aseg, args.output)
    seconds = time.perf_counter() - start
    actual, native = args.output.read_bytes(), args.native.read_bytes()
    native_ids = label_ids(native)
    print(json.dumps({
        "surface": str(args.surface), "aseg": str(args.aseg),
        "native": str(args.native), "output": str(args.output),
        "python_seconds": seconds, "vertex_count": len(selected),
        "native_vertex_count": len(native_ids),
        "ordered_vertex_ids_equal": bool(np.array_equal(selected, native_ids)),
        "text_bytes_equal": actual == native,
        "python_sha256": hashlib.sha256(actual).hexdigest(),
        "native_sha256": hashlib.sha256(native).hexdigest(),
    }, indent=2))


if __name__ == "__main__":
    main()
