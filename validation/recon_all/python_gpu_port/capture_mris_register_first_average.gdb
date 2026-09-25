set pagination off
set confirm off
python
import gdb, hashlib, json, os, struct
from pathlib import Path

def capture_gradient(name):
    inferior = gdb.selected_inferior()
    surface = int(gdb.parse_and_eval("$rbp"))
    count = struct.unpack("<i", bytes(inferior.read_memory(surface + 4, 4)))[0]
    assert count == int(os.environ["REGISTER_EXPECTED_VERTICES"])
    vertices = struct.unpack("<Q", bytes(inferior.read_memory(surface + 0x28, 8)))[0]
    raw = bytes(inferior.read_memory(vertices, count * 0x1d0))
    gradients = bytearray(count * 12)
    for index in range(count):
        struct.pack_into("<3f", gradients, index * 12,
                         *struct.unpack_from("<3f", raw, index * 0x1d0 + 0x60))
    Path(name + ".bin").write_bytes(gradients)
    print("CAPTURE", json.dumps({"checkpoint": name, "vertices": count,
          "first_gradient": struct.unpack_from("<3f", gradients, 0),
          "sha256": hashlib.sha256(gradients).hexdigest()}))
end
break *0x480b46
run
python capture_gradient("gradient_before_average")
break *0x480b4b
continue
python capture_gradient("gradient_after_average")
quit
