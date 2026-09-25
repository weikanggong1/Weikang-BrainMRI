set pagination off
set confirm off
python
import gdb, hashlib, json, os, struct
from pathlib import Path

def read_surface():
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
    return inferior, surface, count, raw, gradients

def capture_gradient(name):
    _, _, count, _, gradients = read_surface()
    Path(name + ".bin").write_bytes(gradients)
    print("CAPTURE", json.dumps({"checkpoint": name, "vertices": count,
          "first_gradient": struct.unpack_from("<3f", gradients, 0),
          "sha256": hashlib.sha256(gradients).hexdigest()}))

def capture_topology():
    inferior, surface, count, raw, _ = read_surface()
    topology = struct.unpack("<Q", bytes(inferior.read_memory(surface + 0x20, 8)))[0]
    topo_raw = bytes(inferior.read_memory(topology, count * 48))
    offsets = [0]
    neighbors = bytearray()
    for index in range(count):
        vertex_topology = index * 48
        degree = struct.unpack_from("<h", topo_raw, vertex_topology + 32)[0]
        assert 0 <= degree <= 20
        if degree:
            address = struct.unpack_from("<Q", topo_raw, vertex_topology + 24)[0]
            neighbors.extend(bytes(inferior.read_memory(address, degree * 4)))
        offsets.append(offsets[-1] + degree)
    offset_bytes = struct.pack("<" + "i" * (count + 1), *offsets)
    ripflags = bytes(raw[index * 0x1d0 + 0x1cf] for index in range(count))
    Path("neighbors_offsets.bin").write_bytes(offset_bytes)
    Path("neighbors_flat.bin").write_bytes(neighbors)
    Path("ripflags.bin").write_bytes(ripflags)
    print("TOPOLOGY", json.dumps({"vertices": count, "neighbor_entries": offsets[-1],
          "first_neighbors": struct.unpack_from("<" + "i" * offsets[1], neighbors, 0),
          "ripped_vertices": sum(bool(value) for value in ripflags),
          "offsets_sha256": hashlib.sha256(offset_bytes).hexdigest(),
          "neighbors_sha256": hashlib.sha256(neighbors).hexdigest()}))
end
break *0x480b46
run
python print("ORIGINAL_AVERAGES", int(gdb.parse_and_eval("$esi")))
python capture_topology()
python capture_gradient("gradient_before_one_average")
set $esi = 1
break *0x480b4b
continue
python capture_gradient("gradient_after_one_average")
quit
