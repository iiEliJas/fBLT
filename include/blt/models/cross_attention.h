#ifndef BLT_CROSS_ATTENTION_H
#define BLT_CROSS_ATTENTION_H

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


typedef struct {
    size_t embed_dim;                    // h_E, shared by Q-source and KV-source
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


// Gradient counterpart
// each tensor is [embed_dim, embed_dim], allocated and zero-initialized by the caller
typedef struct {
    blt_tensor grad_weight_q;
    blt_tensor grad_weight_k;
    blt_tensor grad_weight_v;
    blt_tensor grad_weight_proj;
} blt_cross_attention_grad;



// Input:     query_in [num_patches, embed_dim] (P_{l-1}), kv_in [seq_len, embed_dim] (h_l),
//            projection weights, target output tensor, config, arena.
// Output:    output [num_patches, embed_dim] containing pre-residual cross-attn result.
// Behavior: Projects Q from query_in and K,V from kv_in, computes scaled dot-product
//            attention with block-diagonal patch masking, and applies output projection.
void blt_cross_attention_forward(const blt_tensor* query_in,
                                const blt_tensor* kv_in,
                                const blt_cross_attention_weights* weights,
                                blt_tensor* output,
                                const blt_cross_attention_config* config,
                                blt_arena* arena);


// Input:     Forward pass inputs, grad_out [num_patches, embed_dim] (dL/dOutput), arena.
// Output:    grad_query_in [num_patches, embed_dim], grad_kv_in [seq_len, embed_dim],
//            and grad_weights struct (all overwritten).
// Behavior: Recomputes forward intermediates (Q, K, V, attention probabilities), then
//            propagates gradients backward into query, key/value, and weight matrices.
void blt_cross_attention_backward(const blt_tensor* query_in,
                                 const blt_tensor* kv_in,
                                 const blt_cross_attention_weights* weights,
                                 const blt_tensor* grad_out,
                                 blt_tensor* grad_query_in,
                                 blt_tensor* grad_kv_in,
                                 blt_cross_attention_grad* grad_weights,
                                 const blt_cross_attention_config* config,
                                 blt_arena* arena);


#endif //BLT_CROSS_ATTENTION_H