import torch.nn as nn

from .transformer import TransformerLayer


class GlobalTransformer(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_layers, num_heads, rope_cos, rope_sin):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerLayer(embed_dim, hidden_dim, num_heads, rope_cos, rope_sin)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x, window=0)
        return x
