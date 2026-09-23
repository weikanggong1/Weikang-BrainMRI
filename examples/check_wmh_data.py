"""Verify the three public WMH-SynthSeg FLAIR examples."""

import gzip
import hashlib
import json
from pathlib import Path
import struct


root = Path(__file__).resolve().parent / "wmh_data"
records = json.loads((root / "SOURCES.json").read_text())
for case in records["cases"]:
    path = root / case["file"]
    assert path.stat().st_size == case["published_bytes"], path
    assert hashlib.sha256(path.read_bytes()).hexdigest() == case["published_sha256"], path
    with gzip.open(path, "rb") as volume:
        header = volume.read(348)
    assert struct.unpack_from("<i", header)[0] == 348, path
    assert list(struct.unpack_from("<3h", header, 42)) == case["shape"], path
    assert header[344:348] == b"n+1\x00", path
    print(f"verified {path.name}: {case['shape']}")
