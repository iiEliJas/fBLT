import os
import sys
import tempfile

sys.path.insert(0, "pysrc")
import torch

from fblt_torch.checkpoint import export_c_checkpoint, load_c_checkpoint
from fblt_torch.models.blt import BLT


def test_export_roundtrip():
    m = BLT(embed_dim=8, hidden_dim=16, enc_layers=1, glob_layers=1, dec_layers=1, max_seq_len=32)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "x.fblt")
        export_c_checkpoint(m, path, {"config": {"embed": 8}})
        m2 = BLT(
            embed_dim=8, hidden_dim=16, enc_layers=1, glob_layers=1, dec_layers=1, max_seq_len=32
        )
        load_c_checkpoint(path, m2)
        for a, b in zip(m.parameters(), m2.parameters()):
            assert torch.allclose(a, b)
