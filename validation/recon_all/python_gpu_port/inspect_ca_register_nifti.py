"""Inspect fixed FreeSurfer warp NIfTI header and extension metadata."""

from __future__ import annotations

import hashlib
import json
import struct
import sys

import nibabel as nib


def main() -> None:
    for path in sys.argv[1:]:
        image = nib.load(path)
        header = image.header
        print("PATH", path)
        print("SHAPE", image.shape)
        print("AFFINE", repr(image.affine))
        print("HEADER_SHA256", hashlib.sha256(header.binaryblock).hexdigest())
        print("HEADER_FIELDS", json.dumps({name: str(header[name]) for name in header.keys()}, indent=2))
        for extension in header.extensions:
            payload = extension.get_content()
            print("EXTENSION", extension.get_code(), len(payload), hashlib.sha256(payload).hexdigest())
            print("EXTENSION_START", payload[:128].hex())
            for position in (4, payload.find(struct.pack(">i", 15), 4, 4096)):
                if position < 0:
                    continue
                tag, length = struct.unpack_from(">iq", payload, position)
                print("TAG", position, tag, length)
            print("EXTENSION_END", payload[-64:].hex())


if __name__ == "__main__":
    main()
