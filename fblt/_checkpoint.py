import json
import struct
import typing


def read_meta(path: str) -> typing.Optional[dict]:
    with open(path, "rb") as fh:
        magic = fh.read(4)
        if magic != b"FBLT":
            return None
        (version,) = struct.unpack("<I", fh.read(4))
        if version == 1:
            return None
        if version != 2:
            raise ValueError(f"unsupported checkpoint version {version}")
        (meta_len,) = struct.unpack("<I", fh.read(4))
        if meta_len > 1024 * 1024:
            raise ValueError("meta_len too large")
        meta_bytes = fh.read(meta_len)
        if len(meta_bytes) != meta_len:
            raise ValueError("truncated metadata")
        text = meta_bytes.decode("utf-8")
        return json.loads(text) if text else {}
