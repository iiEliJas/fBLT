"""
Generates every golden .bin file that tests/test_phase2.c loads, 
for both the attention parity test and the transformer block parity test

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


def multihead_attention(x, w_qkv, w_proj, num_heads, is_causal):
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

    def split_heads(t):
        return t.reshape(seq_len, num_heads, head_dim).transpose(1, 0, 2)  # [heads, seq_len, head_dim]

    qh, kh, vh = split_heads(q), split_heads(k), split_heads(v)

    scale = 1.0 / np.sqrt(head_dim)
    scores = np.einsum("hqd,hkd->hqk", qh, kh) * scale  # [heads, seq_len, seq_len]

    if is_causal:
        mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
        scores = np.where(mask[None, :, :], -np.inf, scores)

    weights = softmax_last_axis(scores)  # [heads, seq_len, seq_len]
    attn_out = np.einsum("hqk,hkd->hqd", weights, vh)  # [heads, seq_len, head_dim]
    attn_out = attn_out.transpose(1, 0, 2).reshape(seq_len, embed_dim)  # [seq_len, embed_dim]

    return attn_out @ w_proj  # [seq_len, embed_dim]


def layer_norm(x, weight, bias, eps):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    normed = (x - mean) / np.sqrt(var + eps)
    return normed * weight + bias


def transformer_block(x, weights, num_heads, is_causal, eps):
    norm1 = layer_norm(x, weights["norm1_weight"], weights["norm1_bias"], eps)
    attn_out = multihead_attention(norm1, weights["attn_qkv_w"], weights["attn_proj_w"], num_heads, is_causal)
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
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(SEED)

    def randn(*shape):
        return rng.standard_normal(shape).astype(np.float32) * 0.1

    # -------------------------------------------------------------
    # Attention parity test data
    
    attn_input = randn(SEQ_LEN, EMBED_DIM)
    attn_qkv_w = randn(EMBED_DIM, 3 * EMBED_DIM)
    attn_proj_w = randn(EMBED_DIM, EMBED_DIM)

    attn_expected = multihead_attention(attn_input, attn_qkv_w, attn_proj_w, NUM_HEADS, is_causal=True)

    write_tensor(os.path.join(args.outdir, "phase2_attn_input.bin"), attn_input)
    write_tensor(os.path.join(args.outdir, "phase2_attn_qkv_w.bin"), attn_qkv_w)
    write_tensor(os.path.join(args.outdir, "phase2_attn_proj_w.bin"), attn_proj_w)
    write_tensor(os.path.join(args.outdir, "phase2_attn_expected_out.bin"), attn_expected)

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

    block_expected = transformer_block(block_input, block_weights, NUM_HEADS, is_causal=True, eps=LAYER_NORM_EPS)

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

    print("\nDone. Generated 14 golden files in", args.outdir)


if __name__ == "__main__":
    main()