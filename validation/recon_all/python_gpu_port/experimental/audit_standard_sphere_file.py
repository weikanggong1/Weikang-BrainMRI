"""Compare raw geometry payloads in isolated native and Python sphere files."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sections(path: Path) -> dict:
    with path.open("rb") as stream:
        if stream.read(3) != b"\xff\xff\xfe":
            raise ValueError(f"not a FreeSurfer triangle surface: {path}")
        stamp = stream.readline().decode(errors="replace").rstrip("\n")
        stream.readline()
        nvertices, nfaces = struct.unpack(">ii", stream.read(8))
        xyz = stream.read(12 * nvertices)
        faces = stream.read(12 * nfaces)
        footer = stream.read()
    if len(xyz) != 12 * nvertices or len(faces) != 12 * nfaces:
        raise ValueError(f"incomplete geometry payload: {path}")
    if footer:
        cras = footer.find(b"cras   =")
        if cras < 0:
            raise ValueError(f"incomplete volume geometry: {path}")
        volume_info = footer[:footer.index(b"\n", cras) + 1]
        tags = footer[len(volume_info):]
    else:
        volume_info, tags = b"", b""
    return {"file_sha256": _sha(path.read_bytes()), "stamp": stamp,
            "nvertices": nvertices, "nfaces": nfaces,
            "xyz_sha256": _sha(xyz), "faces_sha256": _sha(faces),
            "volume_info_sha256": _sha(volume_info),
            "provenance_tags_sha256": _sha(tags),
            "provenance_tags_bytes": len(tags)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inflated", type=Path)
    parser.add_argument("native", type=Path)
    parser.add_argument("python", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = {name: _sections(path) for name, path in
              (("inflated", args.inflated), ("native", args.native),
               ("python", args.python))}
    reference, predicted, source = result["native"], result["python"], result["inflated"]
    result["comparison"] = {
        "ordered_vertex_payload_bitwise": predicted["xyz_sha256"] == reference["xyz_sha256"],
        "ordered_face_payload_bitwise": predicted["faces_sha256"] == reference["faces_sha256"],
        "volume_geometry_bytes_bitwise": (predicted["volume_info_sha256"]
                                          == reference["volume_info_sha256"]
                                          == source["volume_info_sha256"]),
        "whole_file_bitwise": predicted["file_sha256"] == reference["file_sha256"],
    }
    output = json.dumps(result, indent=2) + "\n"
    if args.report:
        args.report.write_text(output)
    print(output)


if __name__ == "__main__":
    main()
