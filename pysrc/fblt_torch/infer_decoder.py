import torch
import torch.nn as nn

from .ops import apply_rope, rmsnorm, rope_precompute


def c_decoder_latent_at(patches, pos, trailing_closed=False):
    if pos < patches[0][0]:
        return 0
    M = len(patches)
    for j, (start, plen) in enumerate(patches):
        if start > pos:
            break
        if pos < start + plen:
            if trailing_closed and pos == (patches[-1][0] + patches[-1][1] - 1):
                return M - 1
            is_final = (pos + 1 == start + plen) and (j + 1 < M)
            if is_final:
                return j
            return 0 if j == 0 else j - 1
    return M - 1


class InferLayer(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads, rope_cos, rope_sin):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.norm1 = nn.Parameter(torch.ones(embed_dim))
        self.self_attn_qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.self_attn_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.norm2 = nn.Parameter(torch.ones(embed_dim))
        self.ffn_up = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.ffn_gate = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.ffn_down = nn.Linear(hidden_dim, embed_dim, bias=False)
        self.cross_norm = nn.Parameter(torch.ones(embed_dim))
        self.cross_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.cross_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.cross_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.cross_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.rope_cos = rope_cos
        self.rope_sin = rope_sin

    def forward(self, x, patch_in, mask, cos, sin, embed_dim, q_groups=None):
        S = x.shape[0]
        M = patch_in.shape[0]
        # cross
        xn = rmsnorm(x, self.cross_norm)
        Q = (xn @ self.cross_q.weight.T).view(S, 4, embed_dim // 4).transpose(0, 1)
        K = (patch_in @ self.cross_k.weight.T).view(M, 4, embed_dim // 4).transpose(0, 1)
        V = (patch_in @ self.cross_v.weight.T).view(M, 4, embed_dim // 4).transpose(0, 1)
        cmask = torch.full((S, M), float("-inf"))
        for i in range(S):
            gi = q_groups[i] if i < len(q_groups) else M - 1
            if 0 <= gi < M:
                cmask[i, gi] = 0.0
        scores = (Q @ K.transpose(-2, -1)) / ((embed_dim // 4) ** 0.5) + cmask.unsqueeze(0)
        probs = torch.softmax(scores, dim=-1)
        out = (probs @ V).transpose(0, 1).reshape(S, -1)
        x = x + self.cross_proj(out)
        # self
        normed = rmsnorm(x, self.norm1)
        qkv = self.self_attn_qkv(normed).view(S, 3, 4, embed_dim // 4)
        q = qkv[:, 0].transpose(0, 1)
        k = qkv[:, 1].transpose(0, 1)
        v = qkv[:, 2].transpose(0, 1)
        q = q.permute(1, 0, 2).contiguous().view(S, 4, embed_dim // 4)
        q = apply_rope(q, cos, sin)
        q = q.permute(1, 0, 2).contiguous().view(4, S, embed_dim // 4)
        k = k.permute(1, 0, 2).contiguous().view(S, 4, embed_dim // 4)
        k = apply_rope(k, cos, sin)
        k = k.permute(1, 0, 2).contiguous().view(4, S, embed_dim // 4)
        scores2 = (q @ k.transpose(-2, -1)) / ((embed_dim // 4) ** 0.5) + mask.unsqueeze(0)
        probs2 = torch.softmax(scores2, dim=-1)
        attn_out = (probs2 @ v).transpose(0, 1).reshape(S, -1)
        attn_out = self.self_attn_proj(attn_out)
        x = x + attn_out
        # ffn
        normed2 = rmsnorm(x, self.norm2)
        gate = self.ffn_gate(normed2)
        up = self.ffn_up(normed2)
        x = x + self.ffn_down(torch.nn.functional.silu(gate) * up)
        return x


class InferDecoder(nn.Module):
    def __init__(
        self,
        embed_dim,
        hidden_dim,
        num_layers,
        num_heads,
        cross_heads,
        rope_theta,
        vocab_size=256,
        max_seq_len=4096,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.layers = nn.ModuleList(
            [InferLayer(embed_dim, hidden_dim, num_heads, None, None) for _ in range(num_layers)]
        )
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        cos, sin = rope_precompute(
            max_seq_len, embed_dim // num_heads, rope_theta, "cpu", torch.float32
        )
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)

    def forward(
        self,
        byte_hidden,
        patch_in,
        block_tokens,
        block_positions,
        patches,
        trailing_closed=True,
    ):
        N = byte_hidden.shape[0]
        B = block_tokens.shape[0]
        S = N + B
        M = patch_in.shape[0]
        q_groups = [c_decoder_latent_at(patches, i, True) for i in range(N)] + [M - 1] * B
        x = torch.cat([byte_hidden, torch.zeros(B, self.embed_dim)], dim=0)
        positions = list(range(N)) + [int(p) for p in block_positions.tolist()]
        cos = self.rope_cos[positions]
        sin = self.rope_sin[positions]
        mask = torch.full((S, S), float("-inf"))
        for i in range(N):
            for j in range(i + 1):
                mask[i, j] = 0.0
        for i in range(N, S):
            for j in range(S):
                mask[i, j] = 0.0
        for layer in self.layers:
            x = layer(x, patch_in, mask, cos, sin, self.embed_dim, q_groups)
        return self.lm_head(x)
