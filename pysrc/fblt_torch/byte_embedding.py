import torch
import torch.nn as nn


class ByteEmbedding(nn.Module):
    """Byte embedding table. Shape: (vocab_size, embed_dim)."""

    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(vocab_size, embed_dim))

    def forward(self, byte_ids: torch.Tensor) -> torch.Tensor:
        # byte_ids: (...,) -> (..., embed_dim)
        return self.weight[byte_ids]
