import torch
import torch.nn as nn
import torch.nn.functional as F

from ..decoder import DiffusionDecoder, LocalDecoder
from ..encoder import LocalEncoder
from ..global_model import GlobalTransformer
from ..hash_ngram import HashNgram
from ..ops import rope_precompute


class BLT(nn.Module):
    def __init__(
        self,
        embed_dim=64,
        hidden_dim=128,
        enc_layers=2,
        glob_layers=2,
        dec_layers=2,
        max_seq_len=1024,
        rope_theta=500000.0,
        ngram_sizes=None,
        ngram_buckets=50,
        ngram_prime=1000000007,
        cross_attn="all",
        d0_mode="learned",
        num_heads=4,
        cross_heads=4,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.enc_layers = enc_layers
        self.glob_layers = glob_layers
        self.dec_layers = dec_layers
        self.max_seq_len = max_seq_len
        self.cross_attn = cross_attn
        self.d0_mode = d0_mode
        cos, sin = rope_precompute(
            max_seq_len, embed_dim // num_heads, rope_theta, "cpu", torch.float32
        )
        if ngram_sizes is None:
            ngram_sizes = [3, 4]
        ngram = HashNgram(ngram_sizes, ngram_buckets, ngram_prime, True, embed_dim)
        self.encoder = LocalEncoder(
            embed_dim,
            hidden_dim,
            enc_layers,
            num_heads,
            cross_heads,
            cos,
            sin,
            ngram,
            0,
        )
        self.global_model = GlobalTransformer(
            embed_dim,
            hidden_dim,
            glob_layers,
            num_heads,
            cos[:max_seq_len],
            sin[:max_seq_len],
        )
        self.decoder = LocalDecoder(
            embed_dim,
            hidden_dim,
            dec_layers,
            num_heads,
            cross_heads,
            cos,
            sin,
            256,
            0,
        )
        self.diffusion_decoder = DiffusionDecoder(
            embed_dim,
            hidden_dim,
            dec_layers,
            num_heads,
            cross_heads,
            cos,
            sin,
            256,
            max_seq_len,
            d0_mode,
        )
        # Share decoder weights with the clean decoder for checkpoint interop.
        for i in range(dec_layers):
            dl = self.diffusion_decoder.layers[i]
            cl = self.decoder.layers[i]
            dl.norm1 = cl.norm1
            dl.self_qkv = cl.attn.qkv
            dl.self_proj = cl.attn.proj
            dl.norm2 = cl.norm2
            dl.ffn_up = cl.ffn_up
            dl.ffn_gate = cl.ffn_gate
            dl.ffn_down = cl.ffn_down
            dl.cross_norm = self.decoder.cross_norms[i]
            dl.cross_q = self.decoder.cross_qs[i]
            dl.cross_k = self.decoder.cross_ks[i]
            dl.cross_v = self.decoder.cross_vs[i]
            dl.cross_proj = self.decoder.cross_projs[i]
        self.diffusion_decoder.lm_head = self.decoder.lm_head

    def encode(self, byte_ids, patches):
        return self.encoder(byte_ids, patches)

    def forward(self, byte_ids, patches, targets=None):
        P, h = self.encode(byte_ids, patches)
        g = self.global_model(P)
        logits = self.decoder(h, g, patches)
        loss = None
        if targets is not None:
            t = targets[1 : 1 + logits.shape[0]]
            loss = F.cross_entropy(logits, t.long())
        return logits, loss

    def forward_diffusion(
        self,
        byte_hidden,
        patch_in,
        patches,
        batch,
        clean_bytes=None,
        targets=None,
        trailing_closed=False,
    ):
        return self.diffusion_decoder(
            byte_hidden, patch_in, patches, batch, clean_bytes, targets, trailing_closed
        )


class BLTD(BLT):
    pass
