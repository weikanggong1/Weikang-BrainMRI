set pagination off
set confirm off
break *0x480afc
run
python
import gdb, hashlib, json, os, struct
from pathlib import Path

inferior = gdb.selected_inferior()
surface = int(gdb.parse_and_eval("$rbp"))
count = struct.unpack("<i", bytes(inferior.read_memory(surface + 4, 4)))[0]
assert count == int(os.environ["REGISTER_EXPECTED_VERTICES"])
vertices = struct.unpack("<Q", bytes(inferior.read_memory(surface + 0x28, 8)))[0]
raw = bytes(inferior.read_memory(vertices, count * 0x1d0))
fields = (("positions", 0x18, 3), ("normals", 0x30, 3),
          ("curvature", 0x84, 1), ("e1", 0x118, 3), ("e2", 0x124, 3),
          ("after_correlation", 0x60, 3))
result = {"vertices": count,
          "avg_vertex_dist": struct.unpack("<d", bytes(inferior.read_memory(surface + 0xd8, 8)))[0]}
for name, offset, width in fields:
    values = bytearray(count * width * 4)
    for index in range(count):
        struct.pack_into("<" + "f" * width, values, index * width * 4,
                         *struct.unpack_from("<" + "f" * width, raw,
                                             index * 0x1d0 + offset))
    Path("correlation_" + name + ".bin").write_bytes(values)
    result[name + "_sha256"] = hashlib.sha256(values).hexdigest()
    result["first_" + name] = struct.unpack_from("<" + "f" * width, values, 0)
print("CORRELATION_INPUTS", json.dumps(result))
end
quit
