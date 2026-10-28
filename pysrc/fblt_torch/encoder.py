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
        self.byte_emb = ByteEmbedding(256, embed_dim)
        self.ngram = ngram
        self.layers = nn.ModuleList(
            [
                TransformerLayer(embed_dim, hidden_dim, num_heads, rope_cos, rope_sin)
                for _ in range(num_layers)
            ]
        )
        self.cross_norm = nn.Parameter(torch.ones(embed_dim))
        self.cross_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.cross_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.cross_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.cross_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.local_window = local_window

    def _pool(self, e, patches):
        # patches: list of (start,length); return (num_patches, E)
        out = []
        for s, plen in patches:
            out.append(e[s : s + plen].mean(0))
        return torch.stack(out, 0)

    def forward(self, byte_ids, patches):
        e = self.byte_emb(byte_ids) + self.ngram(byte_ids)
        # byte hidden via causal transformer
        h = e
        for layer in self.layers:
            h = layer(h, window=self.local_window)
        patch_tokens = self._pool(e, patches)  # base P0 from e_0? C uses P_0 = pool(e_0)
        # P starts as pool(e_0); cross-attn added below
        # We'll approximate: update P per layer
        P = patch_tokens.clone()
        for layer in self.layers:
            pass
        # cross-attn using final h
        Pn = rmsnorm(P, self.cross_norm)
        # block-diagonal: patch i attends bytes in its span
        # Build Q,K,V
        Q = self.cross_q(Pn)  # (M,E)
        K = self.cross_k(h)  # (N,E)
        V = self.cross_v(h)
        M = Q.shape[0]
        out = torch.zeros(M, self.embed_dim, device=Q.device, dtype=Q.dtype)
        for i, (s, plen) in enumerate(patches):
            ks = K[s : s + plen]
            vs = V[s : s + plen]
            scores = (Q[i] @ ks.T) / (self.embed_dim**0.5)
            w = torch.softmax(scores, dim=-1)
            out[i] = w @ vs
        P = P + self.cross_proj(out)
        return P, h
