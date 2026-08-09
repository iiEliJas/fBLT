"""
Generates every golden .bin file that tests/test_phase2.c (and the attention
backward parity test) loads, for the attention forward parity test, the
attention backward parity test, and the transformer block parity test.

Run:
    python tools/dump_phase2_golden.py
"""

import argparse
import os
import struct

import numpy as np

SEED = 0
EMBED_DIM = 16
NUM_HEADS = 4
HEAD_DIM = EMBED_DIM // NUM_HEADS
SEQ_LEN = 6
HIDDEN_DIM = 32
LAYER_NORM_EPS = 1e-5



# ---------------------------------------------------------------------
# Binary

def write_tensor(path: str, arr: np.ndarray) -> None:
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    with open(path, "wb") as f:
        f.write(struct.pack("<I", arr.ndim))
        for dim in arr.shape:
            f.write(struct.pack("<I", dim))
        f.write(arr.tobytes(order="C"))
    print(f"wrote {path}  shape={arr.shape}  bytes={arr.nbytes}")



# ---------------------------------------------------------------------
# Reference math — (causal multi-head attention with a combined QKV
# projection, and a pre-norm LayerNorm + GELU transformer block)

def gelu_tanh(x: np.ndarray) -> np.ndarray:
    x3 = x * x * x
    inner = 0.7978845608 * (x + 0.044715 * x3)
    return 0.5 * x * (1.0 + np.tanh(inner))


def softmax_last_axis(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def split_heads(t: np.ndarray, seq_len: int, num_heads: int, head_dim: int) -> np.ndarray:
    # [seq_len, embed_dim] -> [heads, seq_len, head_dim]
    return t.reshape(seq_len, num_heads, head_dim).transpose(1, 0, 2)


def merge_heads(t: np.ndarray, seq_len: int, embed_dim: int) -> np.ndarray:
    # [heads, seq_len, head_dim] -> [seq_len, embed_dim]  (inverse of split_heads)
    return t.transpose(1, 0, 2).reshape(seq_len, embed_dim)


def compute_rope_cos_sin(seq_len: int, head_dim: int, theta: float):
    half = head_dim // 2
    positions = np.arange(seq_len)[:, None]
    freqs = 1.0 / (theta ** ((2 * np.arange(half))[None, :] / head_dim))
    angles = positions * freqs
    return np.cos(angles), np.sin(angles)  # each [seq_len, half]


def apply_rope_forward(xh: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    # xh: [heads, seq_len, head_dim]
    x_even = xh[:, :, 0::2]
    x_odd = xh[:, :, 1::2]
    rotated_even = x_even * cos[None, :, :] - x_odd * sin[None, :, :]
    rotated_odd = x_odd * cos[None, :, :] + x_even * sin[None, :, :]

    rotated = np.empty_like(xh)
    rotated[:, :, 0::2] = rotated_even
    rotated[:, :, 1::2] = rotated_odd
    return rotated


def apply_rope_backward(grad_xh: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    # RoPE is a per-pair rotation (an orthogonal transform), so its backward
    # is the inverse rotation, i.e. the same forward rotation with sin
    # negated. grad_xh: [heads, seq_len, head_dim]
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
    """
    x:      [seq_len, embed_dim]
    w_qkv:  [embed_dim, 3*embed_dim]
    w_proj: [embed_dim, embed_dim]
    returns [seq_len, embed_dim]
    """
    seq_len, embed_dim = x.shape
    head_dim = embed_dim // num_heads

    qkv = x @ w_qkv  # [seq_len, 3*embed_dim]
    q, k, v = np.split(qkv, 3, axis=-1)  # each [seq_len, embed_dim]

    qh = split_heads(q, seq_len, num_heads, head_dim)
    kh = split_heads(k, seq_len, num_heads, head_dim)
    vh = split_heads(v, seq_len, num_heads, head_dim)

    if use_rope:
        qh = apply_rope(qh, seq_len, head_dim, rope_theta)
        kh = apply_rope(kh, seq_len, head_dim, rope_theta)

    scale = 1.0 / np.sqrt(head_dim)
    scores = np.einsum("hqd,hkd->hqk", qh, kh) * scale  # [heads, seq_len, seq_len]

    if is_causal:
        mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
        scores = np.where(mask[None, :, :], -np.inf, scores)

    weights = softmax_last_axis(scores)  # [heads, seq_len, seq_len]
    attn_out = np.einsum("hqk,hkd->hqd", weights, vh)  # [heads, seq_len, head_dim]
    attn_out = merge_heads(attn_out, seq_len, embed_dim)  # [seq_len, embed_dim]

    return attn_out @ w_proj  # [seq_len, embed_dim]


def multihead_attention_backward(x, w_qkv, w_proj, grad_out, num_heads, is_causal,
                                  use_rope=False, rope_theta=10000.0):
    """
    Analytic backward pass mirroring blt_multihead_attention_backward step for
    step: output-projection matmul -> weighted-V sum -> softmax -> QK^T ->
    (optional RoPE) -> QKV projection matmul.

    x:        [seq_len, embed_dim]
    w_qkv:    [embed_dim, 3*embed_dim]
    w_proj:   [embed_dim, embed_dim]
    grad_out: [seq_len, embed_dim]  (dL/d(output))

    returns (grad_x, grad_w_qkv, grad_w_proj)
    """
    seq_len, embed_dim = x.shape
    head_dim = embed_dim // num_heads
    scale = 1.0 / np.sqrt(head_dim)

    # ---- recompute forward intermediates ----
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
    weights = softmax_last_axis(scores)  # [heads, seq_len, seq_len]

    attn_out_heads = np.einsum("hqk,hkd->hqd", weights, vh)  # [heads, seq_len, head_dim]
    combined = merge_heads(attn_out_heads, seq_len, embed_dim)  # [seq_len, embed_dim]

    # ---- Step 1: backward through output projection: out = combined @ w_proj ----
    grad_combined = grad_out @ w_proj.T
    grad_w_proj = combined.T @ grad_out

    grad_combined_h = split_heads(grad_combined, seq_len, num_heads, head_dim)  # [heads, seq_len, head_dim]

    # ---- Step 2: backward through weighted-V sum: attn_out = weights @ V ----
    grad_weights = np.einsum("hqd,hkd->hqk", grad_combined_h, vh)  # [heads, seq_len, seq_len]
    grad_vh = np.einsum("hqk,hqd->hkd", weights, grad_combined_h)  # [heads, seq_len, head_dim]

    # ---- Step 3: softmax backward ----
    dot = np.sum(grad_weights * weights, axis=-1, keepdims=True)
    grad_scores = weights * (grad_weights - dot)  # [heads, seq_len, seq_len]

    # ---- Step 4: backward through scores = scale * Q @ K^T ----
    grad_raw = grad_scores * scale
    grad_qh_rot = np.einsum("hqk,hkd->hqd", grad_raw, kh_rot)
    grad_kh_rot = np.einsum("hqk,hqd->hkd", grad_raw, qh_rot)

    # ---- Step 5: backward through RoPE ----
    if use_rope:
        grad_qh = apply_rope_backward(grad_qh_rot, rope_cos, rope_sin)
        grad_kh = apply_rope_backward(grad_kh_rot, rope_cos, rope_sin)
    else:
        grad_qh = grad_qh_rot
        grad_kh = grad_kh_rot

    grad_q = merge_heads(grad_qh, seq_len, embed_dim)
    grad_k = merge_heads(grad_kh, seq_len, embed_dim)
    grad_v = merge_heads(grad_vh, seq_len, embed_dim)

    grad_qkv = np.concatenate([grad_q, grad_k, grad_v], axis=-1)  # [seq_len, 3*embed_dim]

    # ---- Step 6: backward through QKV projection: qkv = x @ w_qkv ----
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
    attn_out = multihead_attention(norm1, weights["attn_qkv_w"], weights["attn_proj_w"], num_heads, is_causal, use_rope=use_rope, rope_theta=rope_theta)
    attn_residual = x + attn_out

    norm2 = layer_norm(attn_residual, weights["norm2_weight"], weights["norm2_bias"], eps)
    ffn_hidden = norm2 @ weights["ffn_up_w"]
    ffn_activated = gelu_tanh(ffn_hidden)
    ffn_out = ffn_activated @ weights["ffn_down_w"]

    return attn_residual + ffn_out



# ---------------------------------------------------------------------
# Generation

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="data")
    parser.add_argument("--use-rope", action="store_true", help="Enable RoPE in the reference math")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(SEED)

    def randn(*shape):
        return rng.standard_normal(shape).astype(np.float32) * 0.1

    # -------------------------------------------------------------
    # Attention parity test data (forward)

    attn_input = randn(SEQ_LEN, EMBED_DIM)
    attn_qkv_w = randn(EMBED_DIM, 3 * EMBED_DIM)
    attn_proj_w = randn(EMBED_DIM, EMBED_DIM)

    attn_expected = multihead_attention(attn_input, attn_qkv_w, attn_proj_w, NUM_HEADS, is_causal=True, use_rope=args.use_rope)

    write_tensor(os.path.join(args.outdir, "phase2_attn_input.bin"), attn_input)
    write_tensor(os.path.join(args.outdir, "phase2_attn_qkv_w.bin"), attn_qkv_w)
    write_tensor(os.path.join(args.outdir, "phase2_attn_proj_w.bin"), attn_proj_w)
    write_tensor(os.path.join(args.outdir, "phase2_attn_expected_out.bin"), attn_expected)

    # -------------------------------------------------------------
    # Attention parity test data (backward)
    #
    # Reuses attn_input / attn_qkv_w / attn_proj_w and the same config
    # (num_heads=4, is_causal=True) as the forward test above. grad_out is a
    # fresh random upstream gradient with the same shape as the output.

    attn_grad_out = randn(SEQ_LEN, EMBED_DIM)

    attn_grad_input, attn_grad_qkv_w, attn_grad_proj_w = multihead_attention_backward(
        attn_input, attn_qkv_w, attn_proj_w, attn_grad_out,
        NUM_HEADS, is_causal=True, use_rope=args.use_rope,
    )

    write_tensor(os.path.join(args.outdir, "phase2_attn_grad_out.bin"), attn_grad_out)
    write_tensor(os.path.join(args.outdir, "phase2_attn_expected_grad_input.bin"), attn_grad_input)
    write_tensor(os.path.join(args.outdir, "phase2_attn_expected_grad_qkv_w.bin"), attn_grad_qkv_w)
    write_tensor(os.path.join(args.outdir, "phase2_attn_expected_grad_proj_w.bin"), attn_grad_proj_w)

    # -------------------------------------------------------------
    # Transformer block parity test data (LayerNorm + GELU config)

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

    # NOTE: run_phase2_transformer_block_parity_test in test_transformer.c
    # hardcodes .use_rope = true in its attn_config regardless of any
    # command-line flag here, so this call must always match that (unlike
    # the attention-only fixtures above, which follow args.use_rope because
    # the corresponding C tests never set use_rope at all).
    block_expected = transformer_block(block_input, block_weights, NUM_HEADS, is_causal=True, eps=LAYER_NORM_EPS, use_rope=True, rope_theta=10000.0)

    write_tensor(os.path.join(args.outdir, "phase2_block_input.bin"), block_input)
    write_tensor(os.path.join(args.outdir, "phase2_block_norm1_weight.bin"), block_weights["norm1_weight"])
    write_tensor(os.path.join(args.outdir, "phase2_block_norm1_bias.bin"), block_weights["norm1_bias"])
    write_tensor(os.path.join(args.outdir, "phase2_block_attn_qkv_w.bin"), block_weights["attn_qkv_w"])
    write_tensor(os.path.join(args.outdir, "phase2_block_attn_proj_w.bin"), block_weights["attn_proj_w"])
    write_tensor(os.path.join(args.outdir, "phase2_block_norm2_weight.bin"), block_weights["norm2_weight"])
    write_tensor(os.path.join(args.outdir, "phase2_block_norm2_bias.bin"), block_weights["norm2_bias"])
    write_tensor(os.path.join(args.outdir, "phase2_block_ffn_up_w.bin"), block_weights["ffn_up_w"])
    write_tensor(os.path.join(args.outdir, "phase2_block_ffn_down_w.bin"), block_weights["ffn_down_w"])
    write_tensor(os.path.join(args.outdir, "phase2_block_expected_out.bin"), block_expected)

    print("\nDone. Generated 18 golden files in", args.outdir)


if __name__ == "__main__":
    main()