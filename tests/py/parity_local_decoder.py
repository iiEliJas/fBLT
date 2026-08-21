import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from pathlib import Path
from golden import write_tensor_fp32, write_tensor_uint8, create_parser

# Configuration mirroring the C test
EMBED_DIM = 32
NUM_LAYERS = 3
HIDDEN_DIM = 64
NUM_HEADS = 4
CROSS_HEADS = 4
HEAD_DIM = EMBED_DIM // NUM_HEADS
LOCAL_WINDOW = 8
ROPE_THETA = 500000.0
SEQ_LEN = 24
PATCH_STARTS = [0, 4, 9, 13, 18]


def patch_spans():
    spans = []
    for i, start in enumerate(PATCH_STARTS):
        length = (PATCH_STARTS[i + 1] - start) if (i + 1 < len(PATCH_STARTS)) else (SEQ_LEN - start)
        spans.append((start, length))
    return spans


def rope_cos_sin(seq_len, head_dim, theta):
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    pos = torch.arange(seq_len).float()
    angles = torch.outer(pos, freqs)  
    return torch.cos(angles), torch.sin(angles)


def apply_rope(x, cos, sin):
    x1, x2 = x[..., ::2], x[..., 1::2]
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    return torch.stack([out1, out2], dim=-1).flatten(-2)


class ReferenceSelfAttention(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM, num_heads=NUM_HEADS, head_dim=HEAD_DIM):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.qkv_w = nn.Parameter(torch.empty(embed_dim, 3 * embed_dim))
        self.proj_w = nn.Parameter(torch.empty(embed_dim, embed_dim))

    def forward(self, x, cos, sin, window):
        seq_len = x.shape[0]
        q, k, v = (x @ self.qkv_w).chunk(3, dim=-1)
        q = q.view(seq_len, self.num_heads, self.head_dim)
        k = k.view(seq_len, self.num_heads, self.head_dim)
        v = v.view(seq_len, self.num_heads, self.head_dim)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        i_idx = torch.arange(seq_len).unsqueeze(1)
        j_idx = torch.arange(seq_len).unsqueeze(0)
        causal = j_idx <= i_idx
        in_window = (window == 0) | ((i_idx - j_idx) < window)
        mask = torch.zeros(seq_len, seq_len)
        mask.masked_fill_(~(causal & in_window), float("-inf"))

        scores = (q.transpose(0, 1) @ k.transpose(0, 1).transpose(-2, -1)) / math.sqrt(self.head_dim)
        probs = F.softmax(scores + mask.unsqueeze(0), dim=-1)
        return (probs @ v.transpose(0, 1)).transpose(0, 1).reshape(seq_len, -1) @ self.proj_w


class ReferenceCrossAttentionDecoder(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM, num_heads=CROSS_HEADS):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.weight_q = nn.Parameter(torch.empty(embed_dim, embed_dim))
        self.weight_k = nn.Parameter(torch.empty(embed_dim, embed_dim))
        self.weight_v = nn.Parameter(torch.empty(embed_dim, embed_dim))
        self.weight_proj = nn.Parameter(torch.empty(embed_dim, embed_dim))

    def forward(self, byte_query, patch_kv, spans):
        seq_len, num_patches = byte_query.shape[0], patch_kv.shape[0]
        q = (byte_query @ self.weight_q).view(seq_len, self.num_heads, self.head_dim).transpose(0, 1)
        k = (patch_kv @ self.weight_k).view(num_patches, self.num_heads, self.head_dim).transpose(0, 1)
        v = (patch_kv @ self.weight_v).view(num_patches, self.num_heads, self.head_dim).transpose(0, 1)

        # STRICTLY BLOCK-DIAGONAL: Bytes only attend to their parent patch.
        mask = torch.full((seq_len, num_patches), float("-inf"))
        for j, (start, length) in enumerate(spans):
            mask[start:start + length, j] = 0.0

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        probs = F.softmax(scores + mask.unsqueeze(0), dim=-1)
        return (probs @ v).transpose(0, 1).reshape(seq_len, -1) @ self.weight_proj


class ReferenceLocalDecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_norm_weight = nn.Parameter(torch.ones(EMBED_DIM))
        self.cross_attn = ReferenceCrossAttentionDecoder()
        
        self.norm1_weight = nn.Parameter(torch.ones(EMBED_DIM))
        self.self_attn = ReferenceSelfAttention()
        
        self.norm2_weight = nn.Parameter(torch.ones(EMBED_DIM))
        self.ffn_up_w = nn.Parameter(torch.empty(EMBED_DIM, HIDDEN_DIM))
        self.ffn_gate_w = nn.Parameter(torch.empty(EMBED_DIM, HIDDEN_DIM))
        self.ffn_down_w = nn.Parameter(torch.empty(HIDDEN_DIM, EMBED_DIM))

    @staticmethod
    def rmsnorm(x, weight, eps=1e-6):
        return (x / torch.sqrt((x * x).mean(dim=-1, keepdim=True) + eps)) * weight

    def forward(self, x, patch_repr, cos, sin, window, spans, apply_cross=True):
        # 1. Cross-Attention First
        if apply_cross:
            x = x + self.cross_attn(self.rmsnorm(x, self.cross_norm_weight), patch_repr, spans)
            
        # 2. Causal Self-Attention
        x = x + self.self_attn(self.rmsnorm(x, self.norm1_weight), cos, sin, window)
        
        # 3. SwiGLU FFN
        normed2 = self.rmsnorm(x, self.norm2_weight)
        x = x + ((F.silu(normed2 @ self.ffn_gate_w) * (normed2 @ self.ffn_up_w)) @ self.ffn_down_w)
        
        return x

    def apply_cross_only(self, x, patch_repr, spans):
        return x + self.cross_attn(self.rmsnorm(x, self.cross_norm_weight), patch_repr, spans)


class ReferenceLocalDecoder(nn.Module):
    def __init__(self, cross_attn_all_layers=True):
        super().__init__()
        self.cross_attn_all_layers = cross_attn_all_layers
        self.layers = nn.ModuleList([ReferenceLocalDecoderLayer() for _ in range(NUM_LAYERS)])
        self.lm_head_weight = nn.Parameter(torch.empty(EMBED_DIM, 256))
        cos, sin = rope_cos_sin(SEQ_LEN, HEAD_DIM, ROPE_THETA)
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)

    def forward(self, byte_hidden_in, patch_in, bytes_in):
        spans = patch_spans()
        x = byte_hidden_in
        for layer in self.layers:
            x = layer(x, patch_in, self.rope_cos, self.rope_sin, LOCAL_WINDOW, spans, self.cross_attn_all_layers)
        if not self.cross_attn_all_layers:
            x = self.layers[-1].apply_cross_only(x, patch_in, spans)
            
        logits = x @ self.lm_head_weight
        loss = F.cross_entropy(logits[:-1, :].contiguous(), bytes_in[1:].contiguous().long())
        return logits, loss


def generate_data(output_dir="data", cross_attn_all_layers=True):
    out_path = Path(output_dir)
    suffix = "all_layers" if cross_attn_all_layers else "final_layer"
    weights_dir = out_path / f"local_decoder_weights_{suffix}"
    weights_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(42)
    bytes_in = torch.randint(0, 256, (SEQ_LEN,), dtype=torch.uint8)
    
    # CRITICAL FIX: Scale initial inputs to prevent accumulation drift in C
    byte_hidden_in = torch.empty(SEQ_LEN, EMBED_DIM, requires_grad=True)
    patch_in = torch.empty(len(PATCH_STARTS), EMBED_DIM, requires_grad=True)
    nn.init.uniform_(byte_hidden_in, -0.05, 0.05)
    nn.init.uniform_(patch_in, -0.05, 0.05)

    model = ReferenceLocalDecoder(cross_attn_all_layers)
    g = torch.Generator().manual_seed(1234)
    with torch.no_grad():
        for p in model.parameters():
            if p.dim() >= 2: nn.init.uniform_(p, -0.05, 0.05, generator=g)
            else: p.fill_(1.0)
    
    model.train()
    logits, loss = model(byte_hidden_in, patch_in, bytes_in)
    loss.backward()

    if cross_attn_all_layers:
        write_tensor_uint8(out_path / "local_decoder_bytes_in.bin", bytes_in)
        write_tensor_fp32(out_path / "local_decoder_byte_hidden_in.bin", byte_hidden_in.detach())
        write_tensor_fp32(out_path / "local_decoder_patch_in.bin", patch_in.detach())

    # 2. Outputs
    write_tensor_fp32(out_path / f"local_decoder_logits_out_{suffix}.bin", logits.detach())
    # Reshape scalar to 1D tensor of size [1]
    write_tensor_fp32(out_path / f"local_decoder_loss_out_{suffix}.bin", loss.detach().view(1))
    write_tensor_fp32(out_path / f"local_decoder_grad_byte_hidden_in_{suffix}.bin", byte_hidden_in.grad)
    write_tensor_fp32(out_path / f"local_decoder_grad_patch_in_{suffix}.bin", patch_in.grad)
    write_tensor_fp32(weights_dir / "lm_head_weight.bin", model.lm_head_weight)
    
    for l, layer in enumerate(model.layers):
        write_tensor_fp32(weights_dir / f"layer_{l}_norm1_weight.bin", layer.norm1_weight)
        write_tensor_fp32(weights_dir / f"layer_{l}_attn_qkv_w.bin", layer.self_attn.qkv_w)
        write_tensor_fp32(weights_dir / f"layer_{l}_attn_proj_w.bin", layer.self_attn.proj_w)
        write_tensor_fp32(weights_dir / f"layer_{l}_norm2_weight.bin", layer.norm2_weight)
        write_tensor_fp32(weights_dir / f"layer_{l}_ffn_up_w.bin", layer.ffn_up_w)
        write_tensor_fp32(weights_dir / f"layer_{l}_ffn_gate_w.bin", layer.ffn_gate_w)
        write_tensor_fp32(weights_dir / f"layer_{l}_ffn_down_w.bin", layer.ffn_down_w)
        write_tensor_fp32(weights_dir / f"layer_{l}_cross_norm_weight.bin", layer.cross_norm_weight)
        write_tensor_fp32(weights_dir / f"layer_{l}_cross_weight_q.bin", layer.cross_attn.weight_q)
        write_tensor_fp32(weights_dir / f"layer_{l}_cross_weight_k.bin", layer.cross_attn.weight_k)
        write_tensor_fp32(weights_dir / f"layer_{l}_cross_weight_v.bin", layer.cross_attn.weight_v)
        write_tensor_fp32(weights_dir / f"layer_{l}_cross_weight_proj.bin", layer.cross_attn.weight_proj)

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    generate_data(args.output_dir, True)
    generate_data(args.output_dir, False)