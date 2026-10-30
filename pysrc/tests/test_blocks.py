import sys

sys.path.insert(0, "pysrc")
from fblt_torch.blocks import build_block_batch


def test_build_block_batch():
    bytes_seq = list(range(24))
    patches = [(0, 4), (4, 5), (9, 4), (13, 5), (18, 6)]
    b = build_block_batch(bytes_seq, patches, 4, 7, t_in=0.5)
    assert b["num_clean"] == 24
    assert b["n_block_rows"] == 16
    assert b["tokens"].shape[0] == 16
    assert b["cell_masked"].sum().item() >= 0
