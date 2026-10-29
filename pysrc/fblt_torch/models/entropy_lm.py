import torch
import torch.nn as nn
import torch.nn.functional as F

from ..ops import rope_precompute
from ..transformer import TransformerLayer


class EntropyLM(nn.Module):
    def __init__(
        self,
        embed_dim=32,
        hidden_dim=64,
        num_layers=1,
        num_heads=2,
        max_seq_len=512,
        rope_theta=10000.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        cos, sin = rope_precompute(
            max_seq_len, embed_dim // num_heads, rope_theta, "cpu", torch.float32
        )
        self.byte_emb = nn.Embedding(256, embed_dim)
        self.layers = nn.ModuleList(
            [
                TransformerLayer(embed_dim, hidden_dim, num_heads, cos, sin)
                for _ in range(num_layers)
            ]
        )
        self.lm_head = nn.Linear(embed_dim, 256, bias=False)
        self.rope_cos = cos
        self.rope_sin = sin

    def forward(self, byte_ids):
        x = self.byte_emb(byte_ids.long())
        for layer in self.layers:
            x = layer(x, window=0)
        logits = self.lm_head(x)
        return logits

    def loss(self, byte_ids):
        logits = self.forward(byte_ids)
        targets = byte_ids[1:]
        return F.cross_entropy(logits[:-1], targets.long())
