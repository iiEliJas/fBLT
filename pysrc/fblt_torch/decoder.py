import torch
import torch.nn as nn

from .ops import rmsnorm
from .transformer import TransformerLayer


def paper_decoder_latent_at(patches, pos):
    if pos < patches[0][0]:
        return 0
    for j, (start, plen) in enumerate(patches):
        if start > pos:
            break
        if pos < start + plen:
            if pos + 1 == start + plen:
                return j
            return 0 if j == 0 else j - 1
    return len(patches) - 1


class LocalDecoder(nn.Module):
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
        local_window=0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.cross_heads = cross_heads
        self.head_dim = embed_dim // cross_heads
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
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.local_window = local_window

    def forward(self, byte_hidden, patch_in, patches):
        P = patch_in
        if P.shape[1] != self.embed_dim:
            # k-split: mean over sub-tokens
            P = P.view(P.shape[0], -1, self.embed_dim).mean(1)
        x = byte_hidden.clone()
        for li, layer in enumerate(self.layers):
            # cross-attn: bytes query patches
            xn = rmsnorm(x, self.cross_norms[li])
            # group id per byte
            M = P.shape[0]
            Q = (
                (xn @ self.cross_qs[li].weight.T)
                .view(x.shape[0], self.cross_heads, self.head_dim)
                .transpose(0, 1)
            )
            Kmat = (
                (P @ self.cross_ks[li].weight.T)
                .view(M, self.cross_heads, self.head_dim)
                .transpose(0, 1)
            )
            Vmat = (
                (P @ self.cross_vs[li].weight.T)
                .view(M, self.cross_heads, self.head_dim)
                .transpose(0, 1)
            )
            # build mask: byte i attends latent paper_decoder_latent_at(patches,i)
            mask = torch.full((x.shape[0], M), float("-inf"))
            for i in range(x.shape[0]):
                gi = paper_decoder_latent_at(patches, i)
                mask[i, gi] = 0.0
            scores = (Q @ Kmat.transpose(-2, -1)) / (self.head_dim**0.5)
            # mask shape (N,M) -> (1,N? need broadcast to (heads,N,M))
            scores = scores + mask.unsqueeze(0).unsqueeze(0) if False else scores
            scores = scores + mask[None, None, :, :]
            probs = torch.softmax(scores, dim=-1)
            out = (probs @ Vmat).transpose(0, 1).reshape(x.shape[0], -1)
            x = x + self.cross_projs[li](out)
            x = layer(x, window=self.local_window)
        logits = self.lm_head(x)
        return logits
