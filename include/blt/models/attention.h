#ifndef BLT_MODELS_ATTENTION_H
#define BLT_MODELS_ATTENTION_H

#ifdef __cplusplus
extern "C" {
#endif

#include "blt/core/tensor.h"
#include "blt/ops/mask_builder.h"
#include "blt/core/allocator.h"

typedef struct {
    size_t embed_dim;
    size_t num_heads;
    size_t head_dim;     // embed_dim / num_heads
    bool is_causal;      // apply masking matrix
    bool use_rope;
    float rope_theta;    // 500000.0f default
    const blt_tensor* rope_cos_cache;   // optional, precomputed
    const blt_tensor* rope_sin_cache;   // optional, precomputed
    const blt_mask_config* mask_config; // overrides is_causal when non-NULL
} blt_attention_config;

// RoPE tables are precomputed internally when use_rope is set.
void blt_multihead_attention(
    const blt_tensor* input,
    const blt_tensor* weight_qkv,
    const blt_tensor* weight_proj,
    blt_tensor* output,
    const blt_attention_config* config,
    blt_arena* arena
);

void blt_multihead_attention_backward(const blt_tensor* input, const blt_tensor* weight_qkv,
    const blt_tensor* weight_proj, const blt_tensor* grad_out,
    blt_tensor* grad_input, blt_tensor* grad_weight_qkv,
    blt_tensor* grad_weight_proj,
    const blt_attention_config* config, blt_arena* arena);

#ifdef __cplusplus
}
#endif

#endif