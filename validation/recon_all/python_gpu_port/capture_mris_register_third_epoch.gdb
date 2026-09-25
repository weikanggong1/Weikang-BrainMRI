set pagination off
set confirm off
python
import gdb, hashlib, json, os, struct
from pathlib import Path

def capture(name):
    inferior = gdb.selected_inferior()
    surface = int(gdb.parse_and_eval("$rbp"))
    count = struct.unpack("<i", bytes(inferior.read_memory(surface + 4, 4)))[0]
    assert count == int(os.environ["REGISTER_EXPECTED_VERTICES"])
    vertices = struct.unpack("<Q", bytes(inferior.read_memory(surface + 0x28, 8)))[0]
    raw = bytes(inferior.read_memory(vertices, count * 0x1d0))
    result = {"checkpoint": name, "vertices": count}
    for field, offset, width in (("positions", 0x18, 3), ("normals", 0x30, 3),
                                 ("curvature", 0x84, 1), ("e1", 0x118, 3),
                                 ("e2", 0x124, 3), ("gradient", 0x60, 3)):
        values = bytearray(count * width * 4)
        for index in range(count):
            struct.pack_into("<" + "f" * width, values, index * width * 4,
                             *struct.unpack_from("<" + "f" * width, raw,
                                                 index * 0x1d0 + offset))
        Path(name + "_" + field + ".bin").write_bytes(values)
        result[field + "_sha256"] = hashlib.sha256(values).hexdigest()
        result["first_" + field] = struct.unpack_from("<" + "f" * width, values, 0)
    result["avg_vertex_dist"] = struct.unpack("<d", bytes(inferior.read_memory(surface + 0xd8, 8)))[0]
    print("THIRD_EPOCH_CAPTURE", json.dumps(result))
end
break *0x480abe
ignore 1 2
run
python capture("before_distance")
break *0x480ac3
continue
python capture("after_distance")
break *0x480af1
continue
python capture("after_area")
break *0x480afc
continue
python capture("after_correlation")
break *0x480b46
continue
python capture("before_average")
break *0x480b4b
continue
python capture("after_average")
quit
