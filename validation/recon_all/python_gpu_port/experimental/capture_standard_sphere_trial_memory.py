"""GDB script: dump one isolated mris_sphere logSSE trial from native memory.

Run with `gdb -batch -x THIS_FILE --args mris_sphere ...` and set
SPHERE_CAPTURE_DIR and SPHERE_LOGSSE_HIT in the environment.  Offsets refer to
FreeSurfer 8.2.0 d932c45's generated VERTEX/MRIS definitions and installed
`mris_sphere` SHA256 c34ca308a7fa03acdb3f689bf6125cf3d0198c37c3a62992a29f68e631c73612.
"""

import hashlib
import json
import os
from pathlib import Path
import struct
import time

import gdb


class TrialCapture(gdb.Breakpoint):
    def __init__(self):
        super().__init__("*0x4ac8f0")  # fprintf("logSSE:%d") entry
        self.hits = 0

    def stop(self):
        self.hits += 1
        wanted = int(os.environ["SPHERE_LOGSSE_HIT"])
        if self.hits != wanted:
            return False

        started = time.monotonic()
        out = Path(os.environ["SPHERE_CAPTURE_DIR"])
        out.mkdir(parents=True, exist_ok=True)
        inferior = gdb.selected_inferior()
        mris = int(gdb.parse_and_eval("$rbp"))
        header = bytes(inferior.read_memory(mris, 48))
        nvertices = struct.unpack_from("<i", header, 4)[0]
        topology_ptr = struct.unpack_from("<Q", header, 32)[0]
        vertices_ptr = struct.unpack_from("<Q", header, 40)[0]
        if not 100_000 < nvertices < 200_000:
            raise RuntimeError(f"unexpected MRIS vertex count: {nvertices}")
        vertices = bytes(inferior.read_memory(vertices_ptr, nvertices * 464))
        topology = bytes(inferior.read_memory(topology_ptr, nvertices * 48))

        xyz = bytearray(nvertices * 12)
        before = bytearray(nvertices * 12)
        gradient = bytearray(nvertices * 12)
        offsets = bytearray((nvertices + 1) * 8)
        total = 0
        with (out / "native_current.bin").open("wb") as current_file, \
             (out / "native_target.bin").open("wb") as target_file:
            for vno in range(nvertices):
                vertex_offset = vno * 464
                xyz[vno * 12:(vno + 1) * 12] = vertices[vertex_offset + 24:vertex_offset + 36]
                before[vno * 12:(vno + 1) * 12] = vertices[vertex_offset + 160:vertex_offset + 172]
                gradient[vno * 12:(vno + 1) * 12] = vertices[vertex_offset + 96:vertex_offset + 108]
                n = struct.unpack_from("<h", topology, vno * 48 + 38)[0]
                current_ptr, target_ptr = struct.unpack_from("<QQ", vertices, vertex_offset)
                if not 0 < n < 1024 or not current_ptr or not target_ptr:
                    raise RuntimeError(f"invalid metric row {vno}: {n}, {current_ptr:#x}, {target_ptr:#x}")
                current_file.write(inferior.read_memory(current_ptr, n * 4))
                target_file.write(inferior.read_memory(target_ptr, n * 4))
                total += n
                struct.pack_into("<Q", offsets, (vno + 1) * 8, total)

        (out / "native_xyz.bin").write_bytes(xyz)
        (out / "native_before.bin").write_bytes(before)
        (out / "native_gradient.bin").write_bytes(gradient)
        (out / "native_offsets.bin").write_bytes(offsets)
        names = ("native_xyz.bin", "native_before.bin", "native_gradient.bin",
                 "native_offsets.bin", "native_current.bin", "native_target.bin")
        summary = {
            "logSSE_hit": self.hits,
            "mris_address": hex(mris),
            "vertex_count": nvertices,
            "distance_count": total,
            "capture_seconds": time.monotonic() - started,
            "sha256": {name: hashlib.sha256((out / name).read_bytes()).hexdigest() for name in names},
        }
        (out / "native_capture.json").write_text(json.dumps(summary, indent=2) + "\n")
        print("NATIVE_TRIAL_CAPTURE " + json.dumps(summary))
        return True


gdb.execute("set pagination off")
gdb.execute("set confirm off")
TrialCapture()
gdb.execute("run")
