import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from golden import create_parser as create_parser_trans
from golden import write_tensor as write_tensor_transformer

SEED = 0
EMBED_DIM = 16
NUM_HEADS = 4
HEAD_DIM = EMBED_DIM // NUM_HEADS
SEQ_LEN = 6
HIDDEN_DIM = 32
LAYER_NORM_EPS = 1e-5


def gelu_tanh(x: np.ndarray) -> np.ndarray:
    x3 = x * x * x
    inner = 0.7978845608 * (x + 0.044715 * x3)
    return 0.5 * x * (1.0 + np.tanh(inner))


def softmax_last_axis(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def split_heads(t: np.ndarray, seq_len: int, num_heads: int, head_dim: int) -> np.ndarray:
    return t.reshape(seq_len, num_heads, head_dim).transpose(1, 0, 2)


def merge_heads(t: np.ndarray, seq_len: int, embed_dim: int) -> np.ndarray:
    return t.transpose(1, 0, 2).reshape(seq_len, embed_dim)


def compute_rope_cos_sin(seq_len: int, head_dim: int, theta: float):
    half = head_dim // 2
    positions = np.arange(seq_len)[:, None]
    freqs = 1.0 / (theta ** ((2 * np.arange(half))[None, :] / head_dim))
    angles = positions * freqs
    return np.cos(angles), np.sin(angles)


def apply_rope_forward(xh: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    x_even = xh[:, :, 0::2]
    x_odd = xh[:, :, 1::2]
    rotated_even = x_even * cos[None, :, :] - x_odd * sin[None, :, :]
    rotated_odd = x_odd * cos[None, :, :] + x_even * sin[None, :, :]

    rotated = np.empty_like(xh)
    rotated[:, :, 0::2] = rotated_even
    rotated[:, :, 1::2] = rotated_odd
    return rotated


def apply_rope_backward(grad_xh: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    g_even = grad_xh[:, :, 0::2]
    g_odd = grad_xh[:, :, 1::2]
    grad_even = g_even * cos[None, :, :] + g_odd * sin[None, :, :]
    grad_odd = g_odd * cos[None, :, :] - g_even * sin[None, :, :]

    grad_x = np.empty_like(grad_xh)
    grad_x[:, :, 0::2] = grad_even
    grad_x[:, :, 1::2] = grad_odd
    return grad_x


def apply_rope(xh, seq_len, head_dim, theta=10000.0):
    cos, sin = compute_rope_cos_sin(seq_len, head_dim, theta)
    return apply_rope_forward(xh, cos, sin)


def multihead_attention(x, w_qkv, w_proj, num_heads, is_causal, use_rope=False, rope_theta=10000.0):
    seq_len, embed_dim = x.shape
    head_dim = embed_dim // num_heads

    qkv = x @ w_qkv
    q, k, v = np.split(qkv, 3, axis=-1)

    qh = split_heads(q, seq_len, num_heads, head_dim)
    kh = split_heads(k, seq_len, num_heads, head_dim)
    vh = split_heads(v, seq_len, num_heads, head_dim)

    if use_rope:
        qh = apply_rope(qh, seq_len, head_dim, rope_theta)
        kh = apply_rope(kh, seq_len, head_dim, rope_theta)

    scale = 1.0 / np.sqrt(head_dim)
    scores = np.einsum("hqd,hkd->hqk", qh, kh) * scale

    if is_causal:
        mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
        scores = np.where(mask[None, :, :], -np.inf, scores)

    weights = softmax_last_axis(scores)
    attn_out = np.einsum("hqk,hkd->hqd", weights, vh)
    attn_out = merge_heads(attn_out, seq_len, embed_dim)

    return attn_out @ w_proj


def multihead_attention_backward(
    x, w_qkv, w_proj, grad_out, num_heads, is_causal, use_rope=False, rope_theta=10000.0
):
    seq_len, embed_dim = x.shape
    head_dim = embed_dim // num_heads
    scale = 1.0 / np.sqrt(head_dim)

    qkv = x @ w_qkv
    q, k, v = np.split(qkv, 3, axis=-1)

    qh = split_heads(q, seq_len, num_heads, head_dim)
    kh = split_heads(k, seq_len, num_heads, head_dim)
    vh = split_heads(v, seq_len, num_heads, head_dim)

    rope_cos, rope_sin = (None, None)
    qh_rot, kh_rot = qh, kh
    if use_rope:
        rope_cos, rope_sin = compute_rope_cos_sin(seq_len, head_dim, rope_theta)
        qh_rot = apply_rope_forward(qh, rope_cos, rope_sin)
        kh_rot = apply_rope_forward(kh, rope_cos, rope_sin)

    scores = np.einsum("hqd,hkd->hqk", qh_rot, kh_rot) * scale
    if is_causal:
        mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
        scores = np.where(mask[None, :, :], -np.inf, scores)
    weights = softmax_last_axis(scores)

    attn_out_heads = np.einsum("hqk,hkd->hqd", weights, vh)
    combined = merge_heads(attn_out_heads, seq_len, embed_dim)

    grad_combined = grad_out @ w_proj.T
    grad_w_proj = combined.T @ grad_out

    grad_combined_h = split_heads(grad_combined, seq_len, num_heads, head_dim)

    grad_weights = np.einsum("hqd,hkd->hqk", grad_combined_h, vh)
    grad_vh = np.einsum("hqk,hqd->hkd", weights, grad_combined_h)

    dot = np.sum(grad_weights * weights, axis=-1, keepdims=True)
    grad_scores = weights * (grad_weights - dot)

    grad_raw = grad_scores * scale
    grad_qh_rot = np.einsum("hqk,hkd->hqd", grad_raw, kh_rot)
    grad_kh_rot = np.einsum("hqk,hqd->hkd", grad_raw, qh_rot)

    if use_rope:
        grad_qh = apply_rope_backward(grad_qh_rot, rope_cos, rope_sin)
        grad_kh = apply_rope_backward(grad_kh_rot, rope_cos, rope_sin)
    else:
        grad_qh = grad_qh_rot
        grad_kh = grad_kh_rot

    grad_q = merge_heads(grad_qh, seq_len, embed_dim)
    grad_k = merge_heads(grad_kh, seq_len, embed_dim)
    grad_v = merge_heads(grad_vh, seq_len, embed_dim)

    grad_qkv = np.concatenate([grad_q, grad_k, grad_v], axis=-1)

    grad_x = grad_qkv @ w_qkv.T
    grad_w_qkv = x.T @ grad_qkv

    return grad_x, grad_w_qkv, grad_w_proj


def layer_norm(x, weight, bias, eps):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    normed = (x - mean) / np.sqrt(var + eps)
    return normed * weight + bias


def transformer_block(x, weights, num_heads, is_causal, eps, use_rope=False, rope_theta=10000.0):
    norm1 = layer_norm(x, weights["norm1_weight"], weights["norm1_bias"], eps)
    attn_out = multihead_attention(
        norm1,
        weights["attn_qkv_w"],
        weights["attn_proj_w"],
        num_heads,
        is_causal,
        use_rope=use_rope,
        rope_theta=rope_theta,
    )
    attn_residual = x + attn_out

    norm2 = layer_norm(attn_residual, weights["norm2_weight"], weights["norm2_bias"], eps)
    ffn_hidden = norm2 @ weights["ffn_up_w"]
    ffn_activated = gelu_tanh(ffn_hidden)
    ffn_out = ffn_activated @ weights["ffn_down_w"]

    return attn_residual + ffn_out


def generate_golden_transformer_files(outdir: str = "data/tests", use_rope: bool = False) -> None:
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(SEED)

    def randn(*shape):
        return rng.standard_normal(shape).astype(np.float32) * 0.1

    attn_input = randn(SEQ_LEN, EMBED_DIM)
    attn_qkv_w = randn(EMBED_DIM, 3 * EMBED_DIM)
    attn_proj_w = randn(EMBED_DIM, EMBED_DIM)

    attn_expected = multihead_attention(
        attn_input, attn_qkv_w, attn_proj_w, NUM_HEADS, is_causal=True, use_rope=use_rope
    )

    write_tensor_transformer(os.path.join(outdir, "attn_input.bin"), attn_input)
    write_tensor_transformer(os.path.join(outdir, "attn_qkv_w.bin"), attn_qkv_w)
    write_tensor_transformer(os.path.join(outdir, "attn_proj_w.bin"), attn_proj_w)
    write_tensor_transformer(os.path.join(outdir, "attn_expected_out.bin"), attn_expected)

    attn_grad_out = randn(SEQ_LEN, EMBED_DIM)

    attn_grad_input, attn_grad_qkv_w, attn_grad_proj_w = multihead_attention_backward(
        attn_input,
        attn_qkv_w,
        attn_proj_w,
        attn_grad_out,
        NUM_HEADS,
        is_causal=True,
        use_rope=use_rope,
    )

    write_tensor_transformer(os.path.join(outdir, "attn_grad_out.bin"), attn_grad_out)
    write_tensor_transformer(os.path.join(outdir, "attn_expected_grad_input.bin"), attn_grad_input)
    write_tensor_transformer(os.path.join(outdir, "attn_expected_grad_qkv_w.bin"), attn_grad_qkv_w)
    write_tensor_transformer(
        os.path.join(outdir, "attn_expected_grad_proj_w.bin"), attn_grad_proj_w
    )

    block_input = randn(SEQ_LEN, EMBED_DIM)
    block_weights = {
        "norm1_weight": np.ones(EMBED_DIM, dtype=np.float32),
        "norm1_bias": np.zeros(EMBED_DIM, dtype=np.float32),
        "attn_qkv_w": randn(EMBED_DIM, 3 * EMBED_DIM),
        "attn_proj_w": randn(EMBED_DIM, EMBED_DIM),
        "norm2_weight": np.ones(EMBED_DIM, dtype=np.float32),
        "norm2_bias": np.zeros(EMBED_DIM, dtype=np.float32),
        "ffn_up_w": randn(EMBED_DIM, HIDDEN_DIM),
        "ffn_down_w": randn(HIDDEN_DIM, EMBED_DIM),
    }

    block_expected = transformer_block(
        block_input,
        block_weights,
        NUM_HEADS,
        is_causal=True,
        eps=LAYER_NORM_EPS,
        use_rope=True,
        rope_theta=10000.0,
    )

    write_tensor_transformer(os.path.join(outdir, "transformer_input.bin"), block_input)
    write_tensor_transformer(
        os.path.join(outdir, "transformer_norm1_weight.bin"), block_weights["norm1_weight"]
    )
    write_tensor_transformer(
        os.path.join(outdir, "transformer_norm1_bias.bin"), block_weights["norm1_bias"]
    )
    write_tensor_transformer(
        os.path.join(outdir, "transformer_attn_qkv_w.bin"), block_weights["attn_qkv_w"]
    )
    write_tensor_transformer(
        os.path.join(outdir, "transformer_attn_proj_w.bin"), block_weights["attn_proj_w"]
    )
    write_tensor_transformer(
        os.path.join(outdir, "transformer_norm2_weight.bin"), block_weights["norm2_weight"]
    )
    write_tensor_transformer(
        os.path.join(outdir, "transformer_norm2_bias.bin"), block_weights["norm2_bias"]
    )
    write_tensor_transformer(
        os.path.join(outdir, "transformer_ffn_up_w.bin"), block_weights["ffn_up_w"]
    )
    write_tensor_transformer(
        os.path.join(outdir, "transformer_ffn_down_w.bin"), block_weights["ffn_down_w"]
    )
    write_tensor_transformer(os.path.join(outdir, "transformer_expected_out.bin"), block_expected)


def main_transformer():
    parser = create_parser_trans("Generate golden files for transformer")
    parser.add_argument("--use-rope", action="store_true", help="Enable RoPE in reference math")
    args = parser.parse_args()
    generate_golden_transformer_files(str(args.output_dir), use_rope=args.use_rope)


if __name__ == "__main__":
    main_transformer()
