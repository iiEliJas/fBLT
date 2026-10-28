import os
import sys
import tempfile

import torch

sys.path.insert(0, "pysrc")
from fblt_torch.checkpoint import read_checkpoint, write_checkpoint


def test_roundtrip():
    tensors = {"a": torch.randn(2, 3), "b": torch.arange(6).reshape(2, 3).float()}
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ckpt.fblt")
        write_checkpoint(path, 2, {"embed": 32}, tensors)
        v, meta, out = read_checkpoint(path)
        assert v == 2
        assert meta == {"embed": 32}
        for k, val in tensors.items():
            assert torch.equal(val, out[k])
