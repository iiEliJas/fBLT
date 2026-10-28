import torch
import torch.nn as nn

from .byte_embedding import ByteEmbedding
from .ops import rmsnorm
from .transformer import TransformerLayer


class LocalEncoder(nn.Module):
    def __init__(
        self,
        embed_dim,
        hidden_dim,
        num_layers,
        num_heads,
        cross_heads,
        rope_cos,
        rope_sin,
        ngram,
        local_window=0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.cross_heads = cross_heads
        self.head_dim = embed_dim // cross_heads
        self.byte_emb = ByteEmbedding(256, embed_dim)
        self.ngram = ngram
        self.layers = nn.ModuleList(
            [
                TransformerLayer(embed_dim, hidden_dim, num_heads, rope_cos, rope_sin)
                for _ in range(num_layers)
            ]
        )
        self.cross_norms = nn.ParameterList(
            [nn.Parameter(torch.ones(embed_dim)) for _ in range(num_layers)]
        )
        self.cross_qs = nn.ModuleList(
            [nn.Linear(embed_dim, embed_dim, bias=False) for _ in range(num_layers)]
        )
        self.cross_ks = nn.ModuleList(
            [nn.Linear(embed_dim, embed_dim, bias=False) for _ in range(num_layers)]
        )
        self.cross_vs = nn.ModuleList(
            [nn.Linear(embed_dim, embed_dim, bias=False) for _ in range(num_layers)]
        )
        self.cross_projs = nn.ModuleList(
            [nn.Linear(embed_dim, embed_dim, bias=False) for _ in range(num_layers)]
        )
        self.local_window = local_window

    def _pool(self, e, patches):
        out = []
        for s, plen in patches:
            out.append(e[s : s + plen].mean(0))
        return torch.stack(out, 0)

    def forward(self, byte_ids, patches):
        e = self.byte_emb(byte_ids).clone()
        K = len(self.ngram.ngram_sizes)
        for ti, n in enumerate(self.ngram.ngram_sizes):
            hashes = self.ngram._hash(byte_ids, n)
            for i, h in hashes:
                e[i] = e[i] + self.ngram.tables[ti][h]
        if self.ngram.normalize:
            e = e / (K + 1)
        h = e.clone()
        patch_tokens = self._pool(e, patches)
        P = patch_tokens.clone()
        for li, layer in enumerate(self.layers):
            h = layer(h, window=self.local_window)
            Pn = rmsnorm(P, self.cross_norms[li])
            # multi-head cross attention, block-diagonal by patch spans
            Q = (
                (Pn @ self.cross_qs[li].weight.T)
                .view(P.shape[0], self.cross_heads, self.head_dim)
                .transpose(0, 1)
            )
            Kmat = (
                (h @ self.cross_ks[li].weight.T)
                .view(h.shape[0], self.cross_heads, self.head_dim)
                .transpose(0, 1)
            )
            Vmat = (
                (h @ self.cross_vs[li].weight.T)
                .view(h.shape[0], self.cross_heads, self.head_dim)
                .transpose(0, 1)
            )
            mask = torch.full((P.shape[0], h.shape[0]), float("-inf"))
            for j, (s, plen) in enumerate(patches):
                mask[j, s : s + plen] = 0.0
            scores = (Q @ Kmat.transpose(-2, -1)) / (self.head_dim**0.5)
            scores = scores + mask.unsqueeze(0)
            probs = torch.softmax(scores, dim=-1)
            out = (probs @ Vmat).transpose(0, 1).reshape(P.shape[0], -1)
            P = P + self.cross_projs[li](out)
        return P, h
