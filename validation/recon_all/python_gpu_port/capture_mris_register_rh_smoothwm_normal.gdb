set pagination off
set confirm off
python
import gdb, hashlib, json, struct
from pathlib import Path

def capture():
    mean = float(gdb.parse_and_eval('$xmm3.v2_double[0]'))
    std = float(gdb.parse_and_eval('$xmm4.v2_double[0]'))
    if not (-0.025 < mean < -0.015 and 0.240 < std < 0.270):
        return
    inferior = gdb.selected_inferior()
    surface = int(gdb.parse_and_eval('$rbx'))
    count = struct.unpack('<i', bytes(inferior.read_memory(surface + 4, 4)))[0]
    assert count == 105541
    vertices = struct.unpack('<Q', bytes(inferior.read_memory(surface + 0x28, 8)))[0]
    raw = bytes(inferior.read_memory(vertices, count * 0x1d0))
    report = {'vertices': count, 'mean': mean, 'std': std, 'selected': {}}
    for field, offset, width in (('positions', 0x18, 3),
                                 ('normals', 0x30, 3), ('curvature', 0x84, 1)):
        values = bytearray(count * width * 4)
        for index in range(count):
            struct.pack_into('<' + 'f' * width, values, index * width * 4,
                             *struct.unpack_from('<' + 'f' * width, raw,
                                                 index * 0x1d0 + offset))
        Path('native_smoothwm_' + field + '.bin').write_bytes(values)
        report[field + '_sha256'] = hashlib.sha256(values).hexdigest()
        for vertex in (0, 57378):
            report['selected'].setdefault(str(vertex), {})[field] = struct.unpack_from(
                '<' + 'f' * width, values, vertex * width * 4)
    Path('native_smoothwm_capture.json').write_text(json.dumps(report, indent=2) + '\n')
    print('RH_SMOOTHWM_CAPTURE', json.dumps(report))
    gdb.execute('quit')
end
break *0x4a6796
commands
silent
python capture()
continue
end
run
