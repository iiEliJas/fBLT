#ifndef BLT_CROSS_ATTENTION_H
#define BLT_CROSS_ATTENTION_H

#ifdef __cplusplus
extern "C" {
#endif
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "blt/core/backend.h"
#include "blt/core/allocator.h"
#include "blt/ops/mask_builder.h"
#include "blt/models/patcher.h"
#include "blt/ops/elementwise.h"
#include "blt/ops/matmul.h"
#include "blt/ops/softmax.h"

typedef enum {
    BLT_CROSS_ATTN_NO_SPLIT = 0,     // k = 1, current behavior, unchanged
    BLT_CROSS_ATTN_SPLIT_QUERY,      // query_in is [n_q, patch_dim]; kv_in is [n_kv, embed_dim]  (encoder usage)
    BLT_CROSS_ATTN_SPLIT_KV          // kv_in is [n_kv, patch_dim]; query_in is [n_q, embed_dim]   (decoder usage)
} blt_cross_attention_split_mode;

typedef struct {
    size_t embed_dim;                               // h_E or h_D local width; attention always projects to this width
    size_t patch_dim;                               // h_G global width; 0 means same as embed_dim so no splitting
    blt_cross_attention_split_mode split_mode;      // which side is patch_dim-wide
    size_t num_heads;                    // U_E, number of cross-attention heads 
    size_t head_dim;                     // 0 => infer embed_dim / num_heads
    const blt_mask_config* mask_config;  // REQUIRED: block-diagonal patch mask config
} blt_cross_attention_config;

typedef struct {
    const blt_tensor* weight_q;      // [embed_dim, embed_dim]
    const blt_tensor* weight_k;      // [embed_dim, embed_dim]
    const blt_tensor* weight_v;      // [embed_dim, embed_dim]
    const blt_tensor* weight_proj;   // [embed_dim, embed_dim]
} blt_cross_attention_weights;

typedef struct {
    blt_tensor grad_weight_q;
    blt_tensor grad_weight_k;
    blt_tensor grad_weight_v;
    blt_tensor grad_weight_proj;
} blt_cross_attention_grad;

// query_in [num_patches, embed_dim], kv_in [seq_len, embed_dim],
// projection weights, target output tensor, config, arena.
// output [num_patches, embed_dim] — pre-residual cross-attn result.
// Projects Q from query_in, K/V from kv_in, scaled dot-product attention
// with block-diagonal patch masking, output projection.
void blt_cross_attention_forward(const blt_tensor* query_in,
                                const blt_tensor* kv_in,
                                const blt_cross_attention_weights* weights,
                                blt_tensor* output,
                                const blt_cross_attention_config* config,
                                blt_arena* arena);

// Forward inputs plus grad_out [num_patches, embed_dim] (dL/dOutput).
// grad_query_in [num_patches, embed_dim], grad_kv_in [seq_len, embed_dim],
// and grad_weights — all overwritten.
// grad_kv_in is OVERWRITTEN, not accumulated: K and V input gradients are
// summed internally before writing. Callers needing accumulation across
// loss terms must add it themselves.
// Recomputes forward intermediates (Q, K, V, attention probs), propagates
// gradients backward into query, key/value, and weight matrices.
void blt_cross_attention_backward(const blt_tensor* query_in,
                                 const blt_tensor* kv_in,
                                 const blt_cross_attention_weights* weights,
                                 const blt_tensor* grad_out,
                                 blt_tensor* grad_query_in,
                                 blt_tensor* grad_kv_in,
                                 blt_cross_attention_grad* grad_weights,
                                 const blt_cross_attention_config* config,
                                  blt_arena* arena);

#ifdef __cplusplus
}
#endif
#endif //BLT_CROSS_ATTENTION_H
