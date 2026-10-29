import json
import struct

import torch


def _read_tensor(fh):
    (name_len,) = struct.unpack("<H", fh.read(2))
    name = fh.read(name_len).decode("utf-8")
    (ndim,) = struct.unpack("<I", fh.read(4))
    dims = [struct.unpack("<Q" if False else "<I", fh.read(4))[0] for _ in range(ndim)]
    # dims are size_t in C; file uses 4 bytes? C writes sizeof(size_t) (8 on 64-bit). Handle 8.
    # Rewind: above assumed 4; redo properly
    return name, dims


def read_checkpoint(path):
    with open(path, "rb") as fh:
        magic = fh.read(4)
        assert magic == b"FBLT"
        (version,) = struct.unpack("<I", fh.read(4))
        meta = {}
        if version == 2:
            (meta_len,) = struct.unpack("<I", fh.read(4))
            meta = json.loads(fh.read(meta_len).decode("utf-8") or "{}")
        (num_tensors,) = struct.unpack("<I", fh.read(4))
        tensors = {}
        for _ in range(num_tensors):
            (name_len,) = struct.unpack("<H", fh.read(2))
            name = fh.read(name_len).decode("utf-8")
            (ndim,) = struct.unpack("<I", fh.read(4))
            dims = list(struct.unpack("<" + "Q" * ndim, fh.read(8 * ndim)))
            numel = 1
            for d in dims:
                numel *= d
            data = torch.frombuffer(fh.read(numel * 4), dtype=torch.float32).clone().reshape(dims)
            tensors[name] = data
        return version, meta, tensors


def write_checkpoint(path, version, meta, tensors):
    with open(path, "wb") as fh:
        fh.write(b"FBLT")
        fh.write(struct.pack("<I", version))
        if version == 2:
            meta_bytes = json.dumps(meta, separators=(",", ":")).encode("utf-8")
            fh.write(struct.pack("<I", len(meta_bytes)))
            fh.write(meta_bytes)
        fh.write(struct.pack("<I", len(tensors)))
        for name, t in tensors.items():
            nb = name.encode("utf-8")
            fh.write(struct.pack("<H", len(nb)))
            fh.write(nb)
            fh.write(struct.pack("<I", t.dim()))
            fh.write(struct.pack("<" + "Q" * t.dim(), *t.shape))
            fh.write(t.contiguous().view(-1).numpy().astype("float32").tobytes())
