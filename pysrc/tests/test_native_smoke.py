import sys

sys.path.insert(0, "pysrc")
from fblt._config import TrainConfig

from fblt_torch.train import run_training


def test_native_smoke():
    cfg = TrainConfig(
        corpus="data/train.bin", steps=1, window=8, embed=16, hidden=32, layers=1, backend="torch"
    )
    m = run_training(cfg)
    assert sum(p.numel() for p in m.parameters()) > 0
