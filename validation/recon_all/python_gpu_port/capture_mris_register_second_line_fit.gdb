set pagination off
set confirm off
set environment FREESURFER_logSSE 1
break *0x47edb7
ignore 1 1
run
python
import gdb, json, struct
inferior = gdb.selected_inferior()
rsp = int(gdb.parse_and_eval('$rsp'))
row = int(gdb.parse_and_eval('$rax'))
def read(address, fmt):
    return struct.unpack(fmt, bytes(inferior.read_memory(address, struct.calcsize(fmt))))
dt = read(rsp + 0xe0, '<3d')
sse = read(rsp + 0x400, '<3d')
a, b, c = read(row + 4, '<3f')
def matrix(offset, rows, cols):
    pointer = read(rsp + offset, '<Q')[0]
    row_pointers = read(pointer + 0x10, '<Q')[0]
    result = []
    for index in range(1, rows + 1):
        row_pointer = read(row_pointers + 8 * index, '<Q')[0]
        result.append(read(row_pointer + 4, '<' + str(cols) + 'f'))
    return result
print('SECOND_LINE_FIT', json.dumps({'dt_in': dt, 'sse_out': sse,
                                    'normal_matrix': matrix(0xa0, 3, 3),
                                    'normal_inverse': matrix(0xb0, 3, 3),
                                    'right_hand_side': matrix(0xc0, 3, 1),
                                    'a': a, 'b': b, 'c': c,
                                    'predicted_float32_dt': struct.unpack('<f', struct.pack('<f', -b/a))[0]}))
end
call (int) fflush(0)
quit
