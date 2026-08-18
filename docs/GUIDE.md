# Transformer Block Guide

This document explains how `blt_transformer_forward` and its supporting ops work and how to call them correctly. It's a companion to `API_REFERENCE.md` — that file tells you *what functions exist*; this one explains *how they fit together*.

Covers: `blt/models/transformer.h`, `blt/models/attention.h`, and the norm/activation ops that back them (`layernorm.h`, `rmsnorm.h`, `gelu.h`, `swiglu.h`).

---

## 1. What this is

`blt_transformer_forward` is a single pre-norm Transformer block:

```
input ──┬─────────────────────────────────────────────┐
        │                                              │
        ▼                                              │
      norm1 ──▶ attention ──▶ (+) ◀───────────────────┘
                                │
        ┌───────────────────────┘
        │
        ▼
      norm2 ──▶ FFN ──▶ (+) ──▶ output
        ▲                │
        └────────────────┘
     (attn_residual feeds both norm2 and the final add)
```

Concretely, in order:

1. `norm1(input)` — pre-attention normalization
2. `attention(norm1_out)` — causal or non-causal multi-head self-attention
3. `input + attention_out` — first residual connection → `attn_residual`
4. `norm2(attn_residual)` — pre-FFN normalization
5. `FFN(norm2_out)` — up-projection → activation → down-projection
6. `attn_residual + FFN_out` — second residual connection → `output`

This is the same block shape used by essentially every modern decoder-style Transformer (GPT-style, LLaMA-style, etc.) — what differs between models is usually just **which norm** and **which activation** are used, which is exactly what this implementation makes configurable instead of hardcoding.

### Why one function instead of two

A "classic" Transformer block (GPT-2 style: LayerNorm + bias + GELU) and a "BLT-style" block (RMSNorm, no bias, SwiGLU) are architecturally the same six steps above — they only differ in which norm and which activation get plugged in at steps 1/2/4/5. So `blt_transformer_forward` takes that choice as **config**, not as two separate code paths. This means:

- The local encoder, local decoder, and patch transformer (all BLT-style) and any classic baseline block you want to compare against are *the same function*.
- Every ablation in `configs/ablations/` (per the project's Phase 5 design) can flip `norm_type`/`activation_type` without touching model code.

---

## 2. Configuration: `blt_transformer_config`

```c
typedef enum {
    BLT_NORM_LAYERNORM = 0,   // classic: mean/variance norm + weight + bias
    BLT_NORM_RMSNORM   = 1,   // BLT-style: RMS norm + weight only, no bias
} blt_norm_type;

typedef enum {
    BLT_ACTIVATION_GELU   = 0,   // out = down(gelu(up(x)))
    BLT_ACTIVATION_SWIGLU = 1,   // out = down(silu(gate(x)) * up(x))
} blt_activation_type;

typedef struct {
    blt_attention_config attn_config;
    size_t hidden_dim;
    float layer_norm_eps;
    blt_norm_type norm_type;
    blt_activation_type activation_type;
} blt_transformer_config;
```

| Field | Meaning | Notes |
|---|---|---|
| `attn_config` | Passed straight through to `blt_multihead_attention` | See §4 |
| `hidden_dim` | Width of the FFN's intermediate layer | Must equal `ffn_up_w->shape[1]` — checked at call time |
| `layer_norm_eps` | Numerical stability epsilon | **Only used** when `norm_type == BLT_NORM_LAYERNORM`. RMSNorm uses its own fixed internal epsilon (not configurable — it isn't a BLT ablation knob) |
| `norm_type` | Which normalization both `norm1` and `norm2` use | Same choice applies to both norms in the block |
| `activation_type` | Which FFN activation is used | Determines whether `ffn_gate_w` is required |

**Rule of thumb for picking values:**
- Reproducing a classic GPT-style block → `BLT_NORM_LAYERNORM` + `BLT_ACTIVATION_GELU`
- Building the actual BLT local encoder/decoder/patch transformer → `BLT_NORM_RMSNORM` + `BLT_ACTIVATION_SWIGLU`

There's nothing stopping you from mixing (e.g. RMSNorm + GELU) — the function doesn't enforce that the two choices go together, since that's exactly the kind of thing an ablation sweep might want to test.

---

## 3. Weights: `blt_transformer_weights`

```c
typedef struct {
    const blt_tensor* norm1_weight;
    const blt_tensor* norm1_bias;   // NULL when norm_type == BLT_NORM_RMSNORM
    const blt_tensor* attn_qkv_w;
    const blt_tensor* attn_proj_w;
    const blt_tensor* norm2_weight;
    const blt_tensor* norm2_bias;   // NULL when norm_type == BLT_NORM_RMSNORM
    const blt_tensor* ffn_up_w;
    const blt_tensor* ffn_gate_w;   // required (non-NULL) when activation_type == BLT_ACTIVATION_SWIGLU
    const blt_tensor* ffn_down_w;
} blt_transformer_weights;
```

### Shape reference

Given `embed_dim` (inferred from `input->shape[1]`) and `hidden_dim` (= `config->hidden_dim`):

| Weight | Shape | Required when |
|---|---|---|
| `norm1_weight` | `[embed_dim]` | always |
| `norm1_bias` | `[embed_dim]` | `norm_type == BLT_NORM_LAYERNORM` only |
| `attn_qkv_w` | passed to attention, see §4 | always |
| `attn_proj_w` | passed to attention, see §4 | always |
| `norm2_weight` | `[embed_dim]` | always |
| `norm2_bias` | `[embed_dim]` | `norm_type == BLT_NORM_LAYERNORM` only |
| `ffn_up_w` | `[embed_dim, hidden_dim]` | always |
| `ffn_gate_w` | `[embed_dim, hidden_dim]` | `activation_type == BLT_ACTIVATION_SWIGLU` only |
| `ffn_down_w` | `[hidden_dim, embed_dim]` | always |

**Important:** fields marked "NULL when..." aren't optional in the sense of "pass whatever" — passing a non-NULL bias with `BLT_NORM_RMSNORM` is harmless (it's just ignored), but passing `NULL` for `ffn_gate_w` while `activation_type == BLT_ACTIVATION_SWIGLU` will hit a validation failure, since the FFN literally can't compute the gate projection without it.

---

## 4. Attention: `blt_multihead_attention`

```c
typedef struct {
    size_t embed_dim;
    size_t num_heads;
    size_t head_dim;   // 0 → inferred as embed_dim / num_heads
    bool is_causal;
} blt_attention_config;

void blt_multihead_attention(input, weight_qkv, weight_proj, output, config, arena);
```

- `input`: `[seq_len, embed_dim]`
- `weight_qkv`: combined QKV projection, `[embed_dim, 3*embed_dim]`
- `weight_proj`: output projection, `[embed_dim, embed_dim]`
- `is_causal = true` means position *i* can only attend to positions `<= i` — this is what you want for autoregressive generation (byte-level decoding, code completion). Set `false` only for a bidirectional/encoder-style use case.
- `head_dim` can usually be left at `0`; it's only needed explicitly if `embed_dim` doesn't divide evenly by `num_heads`.

Inside `blt_transformer_forward`, `attn_config.embed_dim` **must** match `input->shape[1]` — this is checked as part of the block's own validation, not just the attention call's.

---

## 5. The norm/activation ops directly

You generally won't call these yourself if you're using `blt_transformer_forward`, but they're independently usable (e.g. inside the entropy model, which per `FBLT.md` may use a different, smaller block shape).

### LayerNorm — `blt_layernorm_forward(x, weight, bias, out, eps)`
Row-wise over the last dimension of a `[seq_len, embed_dim]` tensor: subtracts the row mean, divides by `sqrt(variance + eps)`, then applies a learned per-channel `weight` and `bias`. Needs both weight *and* bias — this is the "classic" norm.

### RMSNorm — `blt_rmsnorm_forward(x, weight, out)`
Also row-wise, but **no mean subtraction and no bias**: `out = x / sqrt(mean(x²) + eps) * weight`. Cheaper than LayerNorm and what most modern LLMs (and BLT itself) actually use. Note there's no `eps` parameter here — unlike LayerNorm's epsilon, this one isn't exposed as a tunable, since it's a fixed internal constant rather than a BLT ablation knob.

### GELU — `blt_gelu_forward(x, out)`
Elementwise, tanh approximation. Any rank — `x` and `out` just need matching element counts. Used as the FFN activation in classic blocks.

### SwiGLU — `blt_swiglu_forward(gate, up, out)`
Elementwise `silu(gate) * up`, where `silu(z) = z * sigmoid(z)`. This is *not* a drop-in replacement for GELU in the same way RMSNorm replaces LayerNorm — it needs **two** input tensors (`gate` and `up`), because SwiGLU's whole design is a learned gate multiplying a learned value. This is why `BLT_ACTIVATION_SWIGLU` needs the extra `ffn_gate_w` weight that `BLT_ACTIVATION_GELU` doesn't.

All four ops dispatch through `src/core/backend.c` to a CPU (and eventually CUDA) implementation, following the project's standard "one header, one implementation per backend" pattern — calling code never needs to know which backend it's running on.

---

## 6. Memory: the `arena` parameter

```c
void blt_transformer_forward(input, weights, output, config, arena);
```

`arena` is a bump allocator (`blt_arena`) that every intermediate tensor inside the block (normed input, attention output, residual, normed-attention, FFN hidden, FFN activated) is allocated from — nothing inside the function uses `malloc`/`free`.

- The function **does not reset the arena**. Intermediate lifetime is the caller's responsibility.
- If you're calling this in a loop (e.g. multiple layers, or a training step), decide up front whether you want:
  - **one arena reset per layer** (`blt_arena_reset(arena)` between layers) — lower peak memory, but earlier layers' intermediates become invalid/overwritten
  - **one arena for the whole forward pass** — needed if you'll want those intermediates later (e.g. for a backward pass), but costs more memory
- Size the arena for the worst case: `input`, `norm1_out`, `attn_out`, `attn_residual`, `norm2_out`, `ffn_up_out`, `ffn_activated`, plus (if SwiGLU) `ffn_gate_out` — all at `seq_len × max(embed_dim, hidden_dim)` FP32 elements, plus whatever `blt_multihead_attention` itself allocates internally for QKV splits and attention scores.

---

## 7. Minimal usage example

```c
#include "blt/core/allocator.h"
#include "blt/models/transformer.h"

// Arena sized generously for one block's intermediates.
blt_arena* arena = blt_arena_create(16 * 1024 * 1024, BLT_BACKEND_CPU);

size_t seq_len = 64, embed_dim = 256, hidden_dim = 683; // ~2.67x, typical for SwiGLU

// ... load or create input, weights (norm1_weight, attn_qkv_w, etc.) ...

blt_transformer_config config = {
    .attn_config = {
        .embed_dim = embed_dim,
        .num_heads = 8,
        .head_dim = 0,        // inferred
        .is_causal = true,    // autoregressive byte-level decoding
    },
    .hidden_dim = hidden_dim,
    .layer_norm_eps = 1e-5f,       // ignored for RMSNorm, but harmless to set
    .norm_type = BLT_NORM_RMSNORM,
    .activation_type = BLT_ACTIVATION_SWIGLU,
};

blt_transformer_weights weights = {
    .norm1_weight = &norm1_w,
    .norm1_bias   = NULL,          // unused: RMSNorm has no bias
    .attn_qkv_w   = &qkv_w,
    .attn_proj_w  = &proj_w,
    .norm2_weight = &norm2_w,
    .norm2_bias   = NULL,          // unused: RMSNorm has no bias
    .ffn_up_w     = &ffn_up_w,
    .ffn_gate_w   = &ffn_gate_w,   // required: SwiGLU
    .ffn_down_w   = &ffn_down_w,
};

size_t out_shape[2] = { seq_len, embed_dim };
blt_tensor output = blt_tensor_create(arena, out_shape, 2, BLT_DTYPE_FP32);

blt_transformer_forward(&input, &weights, &output, &config, arena);

blt_arena_destroy(arena);
```

---

## 8. Validation behavior (what fails, and why)

`blt_transformer_forward` validates before doing any work, and fails via `BLT_FATAL` (prints to stderr, exits) rather than returning an error code. Checked, in order:

1. `weights` and `config` aren't `NULL`
2. `input` is 2D FP32
3. `output` shape matches `input`'s `[seq_len, embed_dim]`
4. `norm1_weight`/`norm2_weight` are `[embed_dim]` FP32
5. If `norm_type == BLT_NORM_LAYERNORM`: `norm1_bias`/`norm2_bias` are `[embed_dim]` FP32, and `layer_norm_eps >= 1e-12f`
6. `attn_qkv_w`/`attn_proj_w` aren't `NULL`
7. `ffn_up_w` is `[embed_dim, hidden_dim]`, and that `hidden_dim` matches `config->hidden_dim`
8. If `activation_type == BLT_ACTIVATION_SWIGLU`: `ffn_gate_w` is `[embed_dim, hidden_dim]` FP32
9. `ffn_down_w` is `[hidden_dim, embed_dim]` FP32
10. `arena` isn't `NULL`

If you hit a `BLT_FATAL`, the message tells you which weight/shape was wrong — the most common mistake in practice is forgetting to set `ffn_gate_w` after switching `activation_type` to `BLT_ACTIVATION_SWIGLU`, or leaving `hidden_dim` in the config out of sync with the actual weight shapes.

---

## 9. Where this fits in the bigger picture

Per the project's phase plan, `blt_transformer_forward` is the shared building block for:

- **Local encoder** (`model/local_encoder.*`, Phase 3) — small block-causal local attention window, compresses bytes into a patch embedding
- **Patch transformer** (`model/patch_transformer.*`, Phase 4) — the large block-causal "global" model, the expensive part
- **Local decoder** (`model/local_decoder.*`, Phase 4) — expands patch context back into bytes

All three are expected to use `BLT_NORM_RMSNORM` + `BLT_ACTIVATION_SWIGLU` per the architecture notes; the entropy model (Phase 1) is a separate, smaller model and may reasonably use the classic `BLT_NORM_LAYERNORM` + `BLT_ACTIVATION_GELU` config if that's what it was trained with. Phase 5's ablation sweeps (n-gram sizes, cross-attention placement, encoder/decoder depth split, etc.) all reuse this same function — none of them require touching `transformer.c` itself, only the `blt_transformer_config`/`blt_transformer_weights` values passed in.