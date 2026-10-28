from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class ModelConfig:
    embed: int = 64
    hidden: int = 128
    enc_layers: int = 2
    glob_layers: int = 2
    dec_layers: int = 2
    cross_attn: str = "all"
    d0_mode: str = "learned"
    block_size: int = 4
    window: int = 48
    max_seq_len: int = 1024
    rope_theta_enc: float = 500000.0
    rope_theta_dec: float = 500000.0
    local_window: int = 0
    num_heads: int = 4
    cross_heads: int = 4
    ngram_sizes: List[int] = field(default_factory=lambda: [3, 4])
    ngram_buckets: int = 50
    ngram_prime: int = 1000000007
    patch_dim: int = 0
