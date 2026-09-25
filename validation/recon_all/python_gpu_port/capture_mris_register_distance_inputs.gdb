set pagination off
set confirm off
break *0x480abe
run
python
import gdb, hashlib, json, os, struct
from pathlib import Path
inferior = gdb.selected_inferior()
surface = int(gdb.parse_and_eval("$rbp"))
parms = int(gdb.parse_and_eval("$rbx"))
count = struct.unpack("<i", bytes(inferior.read_memory(surface + 4, 4)))[0]
assert count == int(os.environ["REGISTER_EXPECTED_VERTICES"])
vertices = struct.unpack("<Q", bytes(inferior.read_memory(surface + 0x28, 8)))[0]
topology = struct.unpack("<Q", bytes(inferior.read_memory(surface + 0x20, 8)))[0]
raw = bytes(inferior.read_memory(vertices, count * 0x1d0))
topo_raw = bytes(inferior.read_memory(topology, count * 48))
positions = bytearray(count * 12)
original_positions = bytearray(count * 12)
normals = bytearray(count * 12)
current_distances = bytearray()
original_distances = bytearray()
for index in range(count):
    vertex = index * 0x1d0
    topo = index * 48
    degree = struct.unpack_from("<h", topo_raw, topo + 32)[0]
    total = struct.unpack_from("<h", topo_raw, topo + 38)[0]
    assert degree == total
    struct.pack_into("<3f", positions, index * 12, *struct.unpack_from("<3f", raw, vertex + 0x18))
    struct.pack_into("<3f", original_positions, index * 12, *struct.unpack_from("<3f", raw, vertex + 0x24))
    struct.pack_into("<3f", normals, index * 12, *struct.unpack_from("<3f", raw, vertex + 0x30))
    current_pointer = struct.unpack_from("<Q", raw, vertex)[0]
    original_pointer = struct.unpack_from("<Q", raw, vertex + 8)[0]
    if total:
        current_distances.extend(bytes(inferior.read_memory(current_pointer, total * 4)))
        original_distances.extend(bytes(inferior.read_memory(original_pointer, total * 4)))
fields = {"avg_nbrs": struct.unpack("<f", bytes(inferior.read_memory(surface + 0x570, 4)))[0],
          "total_area": struct.unpack("<f", bytes(inferior.read_memory(surface + 0xc8, 4)))[0],
          "orig_area": struct.unpack("<f", bytes(inferior.read_memory(surface + 0xe8, 4)))[0],
          "status": struct.unpack("<i", bytes(inferior.read_memory(surface + 0x550, 4)))[0],
          "patch": struct.unpack("<i", bytes(inferior.read_memory(surface + 0x558, 4)))[0],
          "l_dist": struct.unpack("<f", bytes(inferior.read_memory(parms + 0x84, 4)))[0]}
for name, data in (("positions", positions), ("original_positions", original_positions),
                   ("normals", normals), ("current_distances", current_distances),
                   ("original_distances", original_distances)):
    Path("distance_" + name + ".bin").write_bytes(data)
    fields[name + "_sha256"] = hashlib.sha256(data).hexdigest()
fields["vertices"] = count
fields["neighbor_entries"] = len(current_distances) // 4
fields["first_position"] = struct.unpack_from("<3f", positions, 0)
fields["first_original_position"] = struct.unpack_from("<3f", original_positions, 0)
fields["first_normal"] = struct.unpack_from("<3f", normals, 0)
fields["first_current_distance"] = struct.unpack_from("<f", current_distances, 0)[0]
fields["first_original_distance"] = struct.unpack_from("<f", original_distances, 0)[0]
print("DISTANCE_INPUTS", json.dumps(fields))
end
quit
