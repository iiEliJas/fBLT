import torch
import torch.nn as nn

from .ops import rmsnorm
from .transformer import SelfAttention


class InferDecoder(nn.Module):
    def __init__(
        self,
        embed_dim,
        hidden_dim,
        num_layers,
        num_heads,
        cross_heads,
        rope_cos,
        rope_sin,
        vocab_size=256,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "norm1": nn.Parameter(torch.ones(embed_dim)),
                        "self_attn": SelfAttention(embed_dim, num_heads, rope_cos, rope_sin),
                        "norm2": nn.Parameter(torch.ones(embed_dim)),
                        "ffn_up": nn.Linear(embed_dim, hidden_dim, bias=False),
                        "ffn_gate": nn.Linear(embed_dim, hidden_dim, bias=False),
                        "ffn_down": nn.Linear(hidden_dim, embed_dim, bias=False),
                        "cross_norm": nn.Parameter(torch.ones(embed_dim)),
                        "cross_q": nn.Linear(embed_dim, embed_dim, bias=False),
                        "cross_k": nn.Linear(embed_dim, embed_dim, bias=False),
                        "cross_v": nn.Linear(embed_dim, embed_dim, bias=False),
                        "cross_proj": nn.Linear(embed_dim, embed_dim, bias=False),
                    }
                )
            )
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, byte_hidden, patch_in, block_tokens, block_positions, block_groups):
        # byte_hidden: (N,E); block tokens: (B,) ints (256 mask); positions: (B,) future positions
        N = byte_hidden.shape[0]
        B = block_tokens.shape[0]
        S = N + B
        # d0: clean rows byte_hidden; block rows zeros (or learned not handled)
        x = torch.cat([byte_hidden, torch.zeros(B, self.embed_dim)], dim=0)
        # rope positions: clean 0..N-1, block positions given
        # self-attention infer mask: clean causal; block rows attend all clean + all block rows
        mask = torch.full((S, S), float("-inf"))
        for i in range(N):
            for j in range(i + 1):
                mask[i, j] = 0.0
        for i in range(N, S):
            for j in range(N):
                mask[i, j] = 0.0
            for j in range(N, S):
                mask[i, j] = 0.0
        # cross-attn: all rows attend last patch (simplified)
        for li, layer in enumerate(self.layers):
            # cross for all rows (clean use paper rule? skip; use group last patch for all)
            xn = rmsnorm(x, layer["cross_norm"])
            M = patch_in.shape[0]
            Q = (xn @ layer["cross_q"].weight.T).view(S, 4, self.embed_dim // 4).transpose(0, 1)
            K = (
                (patch_in @ layer["cross_k"].weight.T)
                .view(M, 4, self.embed_dim // 4)
                .transpose(0, 1)
            )
            V = (
                (patch_in @ layer["cross_v"].weight.T)
                .view(M, 4, self.embed_dim // 4)
                .transpose(0, 1)
            )
            # mask: all rows attend last patch only (simplified)
            cmask = torch.full((S, M), float("-inf"))
            for i in range(S):
                cmask[i, M - 1] = 0.0
            scores = (Q @ K.transpose(-2, -1)) / ((self.embed_dim // 4) ** 0.5) + cmask.unsqueeze(0)
            probs = torch.softmax(scores, dim=-1)
            out = (probs @ V).transpose(0, 1).reshape(S, -1)
            x = x + layer["cross_proj"](out)
            # self-attn
            normed = rmsnorm(x, layer["norm1"])
            # manual self-attn with infer mask
            qkv = layer["self_attn"].qkv(normed).view(S, 3, 4, self.embed_dim // 4)
            q = qkv[:, 0].transpose(0, 1)
            k = qkv[:, 1].transpose(0, 1)
            v = qkv[:, 2].transpose(0, 1)
            scores2 = (q @ k.transpose(-2, -1)) / ((self.embed_dim // 4) ** 0.5) + mask.unsqueeze(0)
            probs2 = torch.softmax(scores2, dim=-1)
            attn_out = (probs2 @ v).transpose(0, 1).reshape(S, -1)
            attn_out = layer["self_attn"].proj(attn_out)
            x = x + attn_out
            # ffn
            normed2 = rmsnorm(x, layer["norm2"])
            gate = layer["ffn_gate"](normed2)
            up = layer["ffn_up"](normed2)
            x = x + layer["ffn_down"](torch.nn.functional.silu(gate) * up)
        logits = self.lm_head(x)
        return logits
