import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from pathlib import Path
from golden import write_tensor_fp32, write_tensor_uint8, create_parser

EMBED_DIM = 32
NUM_LAYERS = 3
HIDDEN_DIM = 64
NUM_HEADS = 4
HEAD_DIM = EMBED_DIM // NUM_HEADS
CROSS_HEADS = 4
LOCAL_WINDOW = 8
ROPE_THETA = 500000.0
NGRAM_SIZES = [3, 4, 5]
NGRAM_VOCAB = 512
HASH_PRIME = 1000000007
PATCH_STARTS = [0, 4, 9, 13, 18]
SEQ_LEN = 24


def patch_spans():
    spans = []
    for i, start in enumerate(PATCH_STARTS):
        length = (PATCH_STARTS[i + 1] - start) if (i + 1 < len(PATCH_STARTS)) else (SEQ_LEN - start)
        spans.append((start, length))
    return spans


def rolling_poly_hash(gram_bytes, prime, modulus):
    n = len(gram_bytes)
    h = 0
    for j, b in enumerate(gram_bytes):
        power = n - 1 - j
        h = (h + int(b) * pow(prime, power, modulus)) % modulus
    return h


def rope_cos_sin(seq_len, head_dim, theta):
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    pos = torch.arange(seq_len).float()
    angles = torch.outer(pos, freqs)  # [seq_len, head_dim/2]
    return torch.cos(angles), torch.sin(angles)


def apply_rope(x, cos, sin):
    # x: [seq_len, num_heads, head_dim]
    x1, x2 = x[..., ::2], x[..., 1::2]
    cos = cos.unsqueeze(1)  # [seq_len, 1, head_dim/2]
    sin = sin.unsqueeze(1)
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    out = torch.stack([out1, out2], dim=-1).flatten(-2)
    return out


class ReferenceNgramEmbedding(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM, ngram_sizes=NGRAM_SIZES,
                 per_ngram_vocab=NGRAM_VOCAB, hash_prime=HASH_PRIME):
        super().__init__()
        self.ngram_sizes = ngram_sizes
        self.hash_prime = hash_prime
        self.per_ngram_vocab = per_ngram_vocab
        self.tables = nn.ParameterList([
            nn.Parameter(torch.empty(per_ngram_vocab, embed_dim)) for _ in ngram_sizes
        ])

    def forward(self, bytes_in):
        seq_len = bytes_in.shape[0]
        out = torch.zeros(seq_len, self.tables[0].shape[1])
        for t_idx, n in enumerate(self.ngram_sizes):
            table = self.tables[t_idx]
            for i in range(seq_len):
                if i < n - 1:
                    continue
                gram = bytes_in[i - n + 1:i + 1]
                idx = rolling_poly_hash(gram, self.hash_prime, self.per_ngram_vocab)
                out[i] += table[idx]
        return out


class ReferenceSelfAttention(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM, num_heads=NUM_HEADS, head_dim=HEAD_DIM):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.qkv_w = nn.Parameter(torch.empty(embed_dim, 3 * embed_dim))
        self.proj_w = nn.Parameter(torch.empty(embed_dim, embed_dim))

    def forward(self, x, cos, sin, window):
        seq_len = x.shape[0]
        qkv = x @ self.qkv_w  # [seq_len, 3*embed_dim]
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(seq_len, self.num_heads, self.head_dim)
        k = k.view(seq_len, self.num_heads, self.head_dim)
        v = v.view(seq_len, self.num_heads, self.head_dim)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        i_idx = torch.arange(seq_len).unsqueeze(1)
        j_idx = torch.arange(seq_len).unsqueeze(0)
        causal = j_idx <= i_idx
        in_window = (window == 0) | ((i_idx - j_idx) < window)
        allowed = causal & in_window
        mask = torch.zeros(seq_len, seq_len)
        mask.masked_fill_(~allowed, float("-inf"))

        q = q.transpose(0, 1)  # [heads, seq, head_dim]
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores + mask.unsqueeze(0)
        probs = F.softmax(scores, dim=-1)
        out = probs @ v  # [heads, seq, head_dim]
        out = out.transpose(0, 1).reshape(seq_len, -1)
        return out @ self.proj_w


class ReferenceCrossAttention(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM, num_heads=CROSS_HEADS):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.weight_q = nn.Parameter(torch.empty(embed_dim, embed_dim))
        self.weight_k = nn.Parameter(torch.empty(embed_dim, embed_dim))
        self.weight_v = nn.Parameter(torch.empty(embed_dim, embed_dim))
        self.weight_proj = nn.Parameter(torch.empty(embed_dim, embed_dim))

    def forward(self, patch_query, byte_kv, spans):
        num_patches = patch_query.shape[0]
        seq_len = byte_kv.shape[0]

        q = (patch_query @ self.weight_q).view(num_patches, self.num_heads, self.head_dim).transpose(0, 1)
        k = (byte_kv @ self.weight_k).view(seq_len, self.num_heads, self.head_dim).transpose(0, 1)
        v = (byte_kv @ self.weight_v).view(seq_len, self.num_heads, self.head_dim).transpose(0, 1)

        # block-diagonal mask
        mask = torch.full((num_patches, seq_len), float("-inf"))
        for j, (start, length) in enumerate(spans):
            mask[j, start:start + length] = 0.0

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores + mask.unsqueeze(0)
        probs = F.softmax(scores, dim=-1)
        out = (probs @ v).transpose(0, 1).reshape(num_patches, -1)
        return out @ self.weight_proj


class ReferenceLocalEncoderLayer(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, num_heads=NUM_HEADS):
        super().__init__()
        self.norm1_weight = nn.Parameter(torch.ones(embed_dim))
        self.self_attn = ReferenceSelfAttention(embed_dim, num_heads)
        self.norm2_weight = nn.Parameter(torch.ones(embed_dim))
        self.ffn_up_w = nn.Parameter(torch.empty(embed_dim, hidden_dim))
        self.ffn_gate_w = nn.Parameter(torch.empty(embed_dim, hidden_dim))
        self.ffn_down_w = nn.Parameter(torch.empty(hidden_dim, embed_dim))

        self.cross_norm_weight = nn.Parameter(torch.ones(embed_dim))
        self.cross_attn = ReferenceCrossAttention(embed_dim, CROSS_HEADS)

    @staticmethod
    def rmsnorm(x, weight, eps=1e-6):
        rms = torch.sqrt((x * x).mean(dim=-1, keepdim=True) + eps)
        return (x / rms) * weight

    def forward(self, x, cos, sin, window):
        normed = self.rmsnorm(x, self.norm1_weight)
        x = x + self.self_attn(normed, cos, sin, window)

        normed2 = self.rmsnorm(x, self.norm2_weight)
        gate = normed2 @ self.ffn_gate_w
        up = normed2 @ self.ffn_up_w
        ffn_out = (F.silu(gate) * up) @ self.ffn_down_w
        x = x + ffn_out
        return x

    def apply_cross_attn(self, patch_repr, byte_hidden, spans):
        normed_patch = self.rmsnorm(patch_repr, self.cross_norm_weight)
        return patch_repr + self.cross_attn(normed_patch, byte_hidden, spans)


class ReferenceLocalEncoder(nn.Module):
    def __init__(self, cross_attn_all_layers=True):
        super().__init__()
        self.embed_dim = EMBED_DIM
        self.num_layers = NUM_LAYERS
        self.cross_attn_all_layers = cross_attn_all_layers

        self.byte_embedding_weight = nn.Parameter(torch.empty(256, EMBED_DIM))
        self.ngram_embedding = ReferenceNgramEmbedding()

        self.layers = nn.ModuleList([
            ReferenceLocalEncoderLayer() for _ in range(self.num_layers)
        ])

        cos, sin = rope_cos_sin(SEQ_LEN, HEAD_DIM, ROPE_THETA)
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)

    def forward(self, bytes_in):
        spans = patch_spans()

        emb = self.byte_embedding_weight[bytes_in.long()]
        ngram_sum = self.ngram_embedding(bytes_in)
        x = (emb + ngram_sum) / (len(NGRAM_SIZES) + 1)  # normalize

        # P_0 via mean pooling of initial byte representations
        patch_repr = torch.stack(
            [x[start:start + length].mean(dim=0) for (start, length) in spans], dim=0
        )

        for layer in self.layers:
            x = layer(x, self.rope_cos, self.rope_sin, LOCAL_WINDOW)
            if self.cross_attn_all_layers:
                patch_repr = layer.apply_cross_attn(patch_repr, x, spans)

        if not self.cross_attn_all_layers:
            patch_repr = self.layers[-1].apply_cross_attn(patch_repr, x, spans)

        byte_hidden_out = x
        patch_out = patch_repr
        return patch_out, byte_hidden_out


def init_weights(model, seed=1234):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            if p.dim() >= 2:
                nn.init.uniform_(p, -0.05, 0.05, generator=g)
            else:
                p.fill_(1.0)  # norm weights start at 1.0


def export_reference_weights(model, weights_dir):
    out_path = Path(weights_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    write_tensor_fp32(out_path / "byte_embedding_weight.bin", model.byte_embedding_weight.detach())

    for t_idx, table in enumerate(model.ngram_embedding.tables):
        write_tensor_fp32(out_path / f"ngram_table_{t_idx}.bin", table.detach())

    for l, layer in enumerate(model.layers):
        write_tensor_fp32(out_path / f"layer_{l}_norm1_weight.bin", layer.norm1_weight.detach())
        write_tensor_fp32(out_path / f"layer_{l}_attn_qkv_w.bin", layer.self_attn.qkv_w.detach())
        write_tensor_fp32(out_path / f"layer_{l}_attn_proj_w.bin", layer.self_attn.proj_w.detach())
        write_tensor_fp32(out_path / f"layer_{l}_norm2_weight.bin", layer.norm2_weight.detach())
        write_tensor_fp32(out_path / f"layer_{l}_ffn_up_w.bin", layer.ffn_up_w.detach())
        write_tensor_fp32(out_path / f"layer_{l}_ffn_gate_w.bin", layer.ffn_gate_w.detach())
        write_tensor_fp32(out_path / f"layer_{l}_ffn_down_w.bin", layer.ffn_down_w.detach())
        write_tensor_fp32(out_path / f"layer_{l}_cross_norm_weight.bin", layer.cross_norm_weight.detach())
        write_tensor_fp32(out_path / f"layer_{l}_cross_weight_q.bin", layer.cross_attn.weight_q.detach())
        write_tensor_fp32(out_path / f"layer_{l}_cross_weight_k.bin", layer.cross_attn.weight_k.detach())
        write_tensor_fp32(out_path / f"layer_{l}_cross_weight_v.bin", layer.cross_attn.weight_v.detach())
        write_tensor_fp32(out_path / f"layer_{l}_cross_weight_proj.bin", layer.cross_attn.weight_proj.detach())


def generate_data(output_dir="data", cross_attn_all_layers=True):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    suffix = "all_layers" if cross_attn_all_layers else "final_layer"

    torch.manual_seed(42)
    bytes_in = torch.randint(0, 256, (SEQ_LEN,), dtype=torch.uint8)

    model = ReferenceLocalEncoder(cross_attn_all_layers=cross_attn_all_layers)
    init_weights(model, seed=1234)  # same seed for both configs
    model.eval()

    with torch.no_grad():
        patch_out, byte_hidden_out = model(bytes_in)

    write_tensor_uint8(out_path / "local_encoder_bytes_in.bin", bytes_in)
    write_tensor_fp32(out_path / f"local_encoder_patch_out_{suffix}.bin", patch_out)
    write_tensor_fp32(out_path / f"local_encoder_byte_hidden_out_{suffix}.bin", byte_hidden_out)

    export_reference_weights(model, out_path / f"local_encoder_weights_{suffix}")


if __name__ == "__main__":
    parser = create_parser("Generate local encoder golden files and weights")
    args = parser.parse_args()

    generate_data(args.output_dir, cross_attn_all_layers=True)
    generate_data(args.output_dir, cross_attn_all_layers=False)
    print("Successfully generated golden files and reference weights.")