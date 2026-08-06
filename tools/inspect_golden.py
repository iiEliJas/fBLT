import struct
from pathlib import Path

for path in [
    'data/golden_matmul_a.bin',
    'data/golden_matmul_b.bin',
    'data/golden_matmul_out.bin',
    'data/golden_softmax_in.bin',
    'data/golden_softmax_out.bin',
]:
    with open(path, 'rb') as fh:
        data = fh.read()
    print(path, 'size', len(data))
    ndim = struct.unpack_from('<I', data, 0)[0]
    offset = 4
    shape = []
    for _ in range(ndim):
        shape.append(struct.unpack_from('<I', data, offset)[0])
        offset += 4
    numel = 1
    for dim in shape:
        numel *= dim
    values = list(struct.unpack_from('<%df' % numel, data, offset))
    print(' ndim', ndim, 'shape', shape, 'values', values)
