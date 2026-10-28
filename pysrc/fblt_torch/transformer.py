import torch
import torch.nn as nn

from .ops import apply_rope, rmsnorm, swiglu


class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, rope_cos, rope_sin):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.rope_cos = rope_cos
        self.rope_sin = rope_sin

    def forward(self, x, window=0):
        S, D = x.shape
        qkv = self.qkv(x).view(S, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        q = apply_rope(q, self.rope_cos[:S], self.rope_sin[:S])
        k = apply_rope(k, self.rope_cos[:S], self.rope_sin[:S])
        # causal mask
        i = torch.arange(S, device=x.device)[:, None]
        j = torch.arange(S, device=x.device)[None, :]
        mask = j <= i
        if window and window > 0:
            mask = mask & ((i - j) < window)
        scores = (q.transpose(0, 1) @ k.transpose(0, 1).transpose(-2, -1)) / (self.head_dim**0.5)
        scores = scores.masked_fill(~mask[None, None, :, :], float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        out = (probs @ v.transpose(0, 1)).transpose(0, 1).reshape(S, D)
        return self.proj(out)


class TransformerLayer(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads, rope_cos, rope_sin):
        super().__init__()
        self.norm1 = nn.Parameter(torch.ones(embed_dim))
        self.attn = SelfAttention(embed_dim, num_heads, rope_cos, rope_sin)
        self.norm2 = nn.Parameter(torch.ones(embed_dim))
        self.ffn_up = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.ffn_gate = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.ffn_down = nn.Linear(hidden_dim, embed_dim, bias=False)

    def forward(self, x, window=0):
        h = rmsnorm(x, self.norm1)
        a = self.attn(h, window)
        r = x + a
        h2 = rmsnorm(r, self.norm2)
        gate = self.ffn_gate(h2)
        up = self.ffn_up(h2)
        ffn = self.ffn_down(swiglu(gate, up))
        return r + ffn
