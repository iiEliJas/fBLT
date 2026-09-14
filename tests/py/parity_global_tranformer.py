import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Assume these are provided by your testing framework as in the local encoder test
from golden import create_parser, write_tensor_fp32

# Configuration matching the C test
NUM_LAYERS = 3
EMBED_DIM = 32
HIDDEN_DIM = 64
NUM_HEADS = 4
HEAD_DIM = EMBED_DIM // NUM_HEADS
MAX_SEQ_LEN = 64
ROPE_THETA = 500000.0
NUM_PATCHES = 16  # Arbitrary sequence length <= MAX_SEQ_LEN


def rope_cos_sin(seq_len, head_dim, theta):
    """Precomputes RoPE frequencies."""
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    pos = torch.arange(seq_len).float()
    angles = torch.outer(pos, freqs)  # [seq_len, head_dim/2]
    return torch.cos(angles), torch.sin(angles)


def apply_rope(x, cos, sin):
    """Applies RoPE to Q and K tensors."""
    # x: [seq_len, num_heads, head_dim]
    x1, x2 = x[..., ::2], x[..., 1::2]
    cos = cos.unsqueeze(1)  # [seq_len, 1, head_dim/2]
    sin = sin.unsqueeze(1)
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    out = torch.stack([out1, out2], dim=-1).flatten(-2)
    return out


class ReferenceGlobalAttention(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM, num_heads=NUM_HEADS, head_dim=HEAD_DIM):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.qkv_w = nn.Parameter(torch.empty(embed_dim, 3 * embed_dim))
        self.proj_w = nn.Parameter(torch.empty(embed_dim, embed_dim))

    def forward(self, x, cos, sin):
        seq_len = x.shape[0]
        qkv = x @ self.qkv_w  # [seq_len, 3*embed_dim]
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(seq_len, self.num_heads, self.head_dim)
        k = k.view(seq_len, self.num_heads, self.head_dim)
        v = v.view(seq_len, self.num_heads, self.head_dim)

        # Apply Rotary Position Embeddings
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # Full causal mask across all patches (single document)
        i_idx = torch.arange(seq_len).unsqueeze(1)
        j_idx = torch.arange(seq_len).unsqueeze(0)
        causal = j_idx <= i_idx
        mask = torch.zeros(seq_len, seq_len)
        mask.masked_fill_(~causal, float("-inf"))

        q = q.transpose(0, 1)  # [heads, seq_len, head_dim]
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores + mask.unsqueeze(0)
        probs = F.softmax(scores, dim=-1)
        out = probs @ v  # [heads, seq_len, head_dim]
        out = out.transpose(0, 1).reshape(seq_len, -1)
        return out @ self.proj_w


class ReferenceGlobalEncoderLayer(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, num_heads=NUM_HEADS):
        super().__init__()
        self.norm1_weight = nn.Parameter(torch.ones(embed_dim))
        self.attn = ReferenceGlobalAttention(embed_dim, num_heads)
        self.norm2_weight = nn.Parameter(torch.ones(embed_dim))

        # SwiGLU FFN
        self.ffn_up_w = nn.Parameter(torch.empty(embed_dim, hidden_dim))
        self.ffn_gate_w = nn.Parameter(torch.empty(embed_dim, hidden_dim))
        self.ffn_down_w = nn.Parameter(torch.empty(hidden_dim, embed_dim))

    @staticmethod
    def rmsnorm(x, weight, eps=1e-6):
        rms = torch.sqrt((x * x).mean(dim=-1, keepdim=True) + eps)
        return (x / rms) * weight

    def forward(self, x, cos, sin):
        # Pre-norm self attention
        normed = self.rmsnorm(x, self.norm1_weight)
        x = x + self.attn(normed, cos, sin)

        # Pre-norm SwiGLU
        normed2 = self.rmsnorm(x, self.norm2_weight)
        gate = normed2 @ self.ffn_gate_w
        up = normed2 @ self.ffn_up_w
        ffn_out = (F.silu(gate) * up) @ self.ffn_down_w
        x = x + ffn_out

        return x


class ReferenceGlobalTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([ReferenceGlobalEncoderLayer() for _ in range(NUM_LAYERS)])

        cos, sin = rope_cos_sin(MAX_SEQ_LEN, HEAD_DIM, ROPE_THETA)
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)

    def forward(self, x):
        seq_len = x.shape[0]
        # Only use the sequence length we currently need
        cos = self.rope_cos[:seq_len]
        sin = self.rope_sin[:seq_len]

        for layer in self.layers:
            x = layer(x, cos, sin)
        return x


def init_weights(model, seed=1234):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            if p.dim() >= 2:
                nn.init.uniform_(p, -0.05, 0.05, generator=g)
            else:
                p.fill_(1.0)  # RMSNorm weights


def generate_data(output_dir="data/tests"):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    weights_dir = out_path / "global_transformer_weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    # 1. Initialize random seeds and inputs
    torch.manual_seed(42)
    patch_in = torch.randn(NUM_PATCHES, EMBED_DIM, requires_grad=True)
    grad_patch_out = torch.randn(NUM_PATCHES, EMBED_DIM)

    # 2. Build model and apply initialization
    model = ReferenceGlobalTransformer()
    init_weights(model, seed=1234)
    model.train()  # Keep track of gradients

    # 3. Forward Pass
    patch_out = model(patch_in)

    # 4. Backward Pass
    patch_out.backward(grad_patch_out)

    # 5. Export I/O Tensors
    write_tensor_fp32(out_path / "global_transformer_patch_in.bin", patch_in)
    write_tensor_fp32(out_path / "global_transformer_patch_out.bin", patch_out)
    write_tensor_fp32(out_path / "global_transformer_grad_patch_out.bin", grad_patch_out)
    write_tensor_fp32(out_path / "global_transformer_grad_patch_in.bin", patch_in.grad)

    # 6. Export Weights and Gradients
    for li, layer in enumerate(model.layers):
        # Forward Weights
        write_tensor_fp32(weights_dir / f"layer_{li}_norm1_weight.bin", layer.norm1_weight)
        write_tensor_fp32(weights_dir / f"layer_{li}_attn_qkv_w.bin", layer.attn.qkv_w)
        write_tensor_fp32(weights_dir / f"layer_{li}_attn_proj_w.bin", layer.attn.proj_w)
        write_tensor_fp32(weights_dir / f"layer_{li}_norm2_weight.bin", layer.norm2_weight)
        write_tensor_fp32(weights_dir / f"layer_{li}_ffn_up_w.bin", layer.ffn_up_w)
        write_tensor_fp32(weights_dir / f"layer_{li}_ffn_gate_w.bin", layer.ffn_gate_w)
        write_tensor_fp32(weights_dir / f"layer_{li}_ffn_down_w.bin", layer.ffn_down_w)

        # Gradients
        write_tensor_fp32(
            weights_dir / f"grad_layer_{li}_norm1_weight.bin", layer.norm1_weight.grad
        )
        write_tensor_fp32(weights_dir / f"grad_layer_{li}_attn_qkv_w.bin", layer.attn.qkv_w.grad)
        write_tensor_fp32(weights_dir / f"grad_layer_{li}_attn_proj_w.bin", layer.attn.proj_w.grad)
        write_tensor_fp32(
            weights_dir / f"grad_layer_{li}_norm2_weight.bin", layer.norm2_weight.grad
        )
        write_tensor_fp32(weights_dir / f"grad_layer_{li}_ffn_up_w.bin", layer.ffn_up_w.grad)
        write_tensor_fp32(weights_dir / f"grad_layer_{li}_ffn_gate_w.bin", layer.ffn_gate_w.grad)
        write_tensor_fp32(weights_dir / f"grad_layer_{li}_ffn_down_w.bin", layer.ffn_down_w.grad)


def main_global_transformer():
    parser = create_parser("Generate global transformer golden files and weights")
    args = parser.parse_args()
    generate_data(args.output_dir)
    print("Successfully generated global transformer golden files and weights.")


if __name__ == "__main__":
    main_global_transformer()
