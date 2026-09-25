set pagination off
set confirm off
break *0x480aec
run
python
import gdb, hashlib, json, os, struct
from pathlib import Path
inferior = gdb.selected_inferior()
surface = int(gdb.parse_and_eval("$rbp"))
number = struct.unpack("<i", bytes(inferior.read_memory(surface + 8, 4)))[0]
assert number == int(os.environ["REGISTER_EXPECTED_FACES"])
faces = struct.unpack("<Q", bytes(inferior.read_memory(surface + 0x48, 8)))[0]
cache = struct.unpack("<Q", bytes(inferior.read_memory(surface + 0x60, 8)))[0]
face_raw = bytes(inferior.read_memory(faces, number * 80))
cache_raw = bytes(inferior.read_memory(cache, number * 16))
vertex_indices = bytearray(number * 12)
current_areas = bytearray(number * 4)
original_normals = bytearray(number * 12)
original_areas = bytearray(number * 4)
ripped = bytearray(number)
for index in range(number):
    face = index * 80
    entry = index * 16
    struct.pack_into("<3i", vertex_indices, index * 12, *struct.unpack_from("<3i", face_raw, face))
    struct.pack_into("<f", current_areas, index * 4, struct.unpack_from("<f", face_raw, face + 12)[0])
    struct.pack_into("<3f", original_normals, index * 12, *struct.unpack_from("<3f", cache_raw, entry))
    struct.pack_into("<f", original_areas, index * 4, struct.unpack_from("<f", cache_raw, entry + 12)[0])
    ripped[index] = face_raw[face + 40]
assert not any(ripped)
fields = {"faces": number,
          "first_vertices": struct.unpack_from("<3i", vertex_indices, 0),
          "first_current_area": struct.unpack_from("<f", current_areas, 0)[0],
          "first_original_normal": struct.unpack_from("<3f", original_normals, 0),
          "first_original_area": struct.unpack_from("<f", original_areas, 0)[0],
          "ripped_faces": sum(bool(x) for x in ripped)}
for name, data in (("vertices", vertex_indices), ("current_areas", current_areas),
                   ("original_normals", original_normals), ("original_areas", original_areas)):
    Path("area_" + name + ".bin").write_bytes(data)
    fields[name + "_sha256"] = hashlib.sha256(data).hexdigest()
print("AREA_INPUTS", json.dumps(fields))
end
quit
