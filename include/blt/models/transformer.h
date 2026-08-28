#ifndef BLT_MODELS_TRANSFORMER_H
#define BLT_MODELS_TRANSFORMER_H

#ifdef __cplusplus
extern "C" {
#endif

 
#include "blt/core/tensor.h"
#include "blt/core/allocator.h"
#include "blt/models/attention.h"
 
// normalization the block uses. RMSNorm has no bias term, so
// norm1_bias/norm2_bias in blt_transformer_weights are ignored (may be
// NULL) when BLT_NORM_RMSNORM is set.
typedef enum {
    BLT_NORM_LAYERNORM = 0,   // classic pre-norm block: mean/var norm + weight + bias
    BLT_NORM_RMSNORM   = 1,   // BLT-style block: RMS norm + weight only
} blt_norm_type;
 
// FFN activation the block uses. SwiGLU needs a gate projection in
// addition to up/down, so ffn_gate_w in blt_transformer_weights is
// required (non-NULL) when BLT_ACTIVATION_SWIGLU is set, and ignored
// otherwise.
typedef enum {
    BLT_ACTIVATION_GELU   = 0,   // out = down(gelu(up(x)))
    BLT_ACTIVATION_SWIGLU = 1,   // out = down(silu(gate(x)) * up(x))
} blt_activation_type;
 
// Everything a config can sweep for a single block: which
// norm, which activation, attention shape, FFN width. Nothing about the
// block's behavior is hardcoded — a classic transformer block and a BLT
// local-encoder/decoder/patch-transformer block are the same function
// with a different config.
typedef struct {
    blt_attention_config attn_config;
    size_t hidden_dim;        // FFN hidden width
    float layer_norm_eps;     // used only when norm_type == BLT_NORM_LAYERNORM
    blt_norm_type norm_type;
    blt_activation_type activation_type;
    int use_bf16;             // 1 = use bf16 weight copies for matmul (mixed precision)
} blt_transformer_config;
 
// All learned weights for one block. Fields whose relevance depends on
// config are documented above (norm1_bias/norm2_bias, ffn_gate_w).
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


// Optional bf16 weight copies for mixed-precision matmul.
// NULL when use_bf16 is disabled. Only the matmul-participating
// weights have bf16 copies; norm weights stay fp32.
typedef struct {
    const blt_tensor* attn_qkv_w;
    const blt_tensor* attn_proj_w;
    const blt_tensor* ffn_up_w;
    const blt_tensor* ffn_gate_w;
    const blt_tensor* ffn_down_w;
} blt_transformer_weights_bf16;


// Owned storage for one transformer layer's weights. A parallel
// blt_transformer_weights entry (of const pointers into this struct) is
// what actually gets passed to blt_transformer_forward.
typedef struct {
    blt_tensor norm1_weight;
    blt_tensor attn_qkv_w;
    blt_tensor attn_proj_w;
    blt_tensor norm2_weight;
    blt_tensor ffn_up_w;
    blt_tensor ffn_gate_w;
    blt_tensor ffn_down_w;
    // bf16 copies for mixed-precision matmul (allocated only when use_bf16=1)
    blt_tensor attn_qkv_w_bf16;
    blt_tensor attn_proj_w_bf16;
    blt_tensor ffn_up_w_bf16;
    blt_tensor ffn_gate_w_bf16;
    blt_tensor ffn_down_w_bf16;
} blt_transformer_layer_storage;


// Gradient part of blt_transformer_layer_storage
typedef struct {
    blt_tensor norm1_weight;
    blt_tensor attn_qkv_w;
    blt_tensor attn_proj_w;
    blt_tensor norm2_weight;
    blt_tensor ffn_up_w;
    blt_tensor ffn_gate_w;
    blt_tensor ffn_down_w;
} blt_transformer_layer_grad;

 
// Executes a single Transformer block forward pass: pre-norm attention with
// residual, then pre-norm FFN with residual. Norm and activation are chosen
// via config, so this one function serves the classic transformer block and
// BLT's local encoder/decoder/patch-transformer blocks alike.
// `arena` provides scratch storage for intermediates; it is not reset by
// this function, so the caller controls intermediate lifetime.
// `w_bf16` may be NULL (fp32-only path) or non-NULL (mixed-precision path).
void blt_transformer_forward(
    const blt_tensor* input,
    const blt_transformer_weights* weights,
    blt_tensor* output,
    const blt_transformer_config* config,
    blt_arena* arena
);

// Builds a blt_transformer_weights_bf16 view from a layer storage.
// Returns a zeroed struct when use_bf16 is disabled.
blt_transformer_weights_bf16 blt_transformer_layer_bf16_view(
    const blt_transformer_layer_storage* s, int use_bf16);

// Refresh bf16 weight copies from their fp32 masters.
// Must be called once per training step after the optimizer update
// and before the next forward pass.
void blt_mixed_weight_refresh(blt_tensor* bf16_copy, const blt_tensor* fp32_master);

#ifdef __cplusplus
}
#endif

#endif // BLT_MODELS_TRANSFORMER_H