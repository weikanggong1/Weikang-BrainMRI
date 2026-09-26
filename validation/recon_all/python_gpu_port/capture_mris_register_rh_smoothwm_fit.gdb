set pagination off
set confirm off
python
import gdb, json, struct
from pathlib import Path

records = {}
inferior = None

def unpack(address, fmt):
    return struct.unpack(fmt, bytes(inferior.read_memory(address, struct.calcsize(fmt))))

def matrix(address):
    kind, rows, cols = unpack(address, '<hxxii')
    assert kind == 1 and 0 < rows <= 256 and 0 < cols <= 256
    row_pointers = unpack(address + 16, '<Q')[0]
    result = []
    for row in range(1, rows + 1):
        pointer = unpack(row_pointers + row * 8, '<Q')[0]
        result.append(unpack(pointer + 4, '<' + 'f' * cols))
    return result

def capture():
    global inferior
    inferior = gdb.selected_inferior()
    stack = int(gdb.parse_and_eval('$rsp'))
    vertex = unpack(stack + 0x6c, '<i')[0]
    if vertex not in (0, 57378):
        return
    address = int(gdb.parse_and_eval('$rbx'))
    position = unpack(address + 0x18, '<3f')
    expected_x = {0: 22.442964553833008, 57378: 13.276896476745605}[vertex]
    if position[0] != expected_x:
        return
    surface = unpack(stack + 0x50, '<Q')[0]
    topology_base = unpack(surface + 0x20, '<Q')[0]
    topology = topology_base + vertex * 0x30
    neighbor_count = unpack(topology + 0x26, '<h')[0]
    assert 0 < neighbor_count <= 256
    neighbor_pointer = unpack(topology + 0x18, '<Q')[0]
    record = {
        'vertex': vertex,
        'position': position,
        'normal': unpack(address + 0x30, '<3f'),
        'tangent_e1': unpack(address + 0x118, '<3f'),
        'tangent_e2': unpack(address + 0x124, '<3f'),
        'neighbors': unpack(neighbor_pointer, '<' + 'i' * neighbor_count),
        'design': matrix(unpack(stack + 0xb0, '<Q')[0]),
        'height': matrix(unpack(stack + 0x130, '<Q')[0]),
        'gram': matrix(unpack(stack + 0xe0, '<Q')[0]),
        'rhs': matrix(unpack(stack + 0xd0, '<Q')[0]),
        'inverse': matrix(unpack(stack + 0xf0, '<Q')[0]),
        'coefficients': matrix(unpack(stack + 0x120, '<Q')[0]),
        'condition_number': unpack(stack, '<f')[0],
    }
    assert len(record['neighbors']) == len(record['design']) == len(record['height'])
    records[str(vertex)] = record
    print('RH_SMOOTHWM_FIT_CAPTURE', vertex, neighbor_count, record['condition_number'])
    if len(records) == 2:
        Path('native_smoothwm_fit.json').write_text(json.dumps(records, indent=2) + '\n')
        gdb.execute('quit')
end
break *0x4a81d5
commands
silent
python capture()
continue
end
run
