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



/*
 * blt_cross_attention_forward(query_in, kv_in, weights, output, config, arena)
 *   Input:  query_in [num_patches, embed_dim] FP32 — P_{l-1},
 *           kv_in    [seq_len, embed_dim]    FP32 — h_{l-1} (byte states),
 *           weights  separate W_q/W_k/W_v/W_o (Q and K/V come from different
 *                    input tensors, so no combined QKV matrix is possible),
 *           config   with mask_config != NULL.
 *   Output: output [num_patches, embed_dim] — PRE-RESIDUAL cross-attn result.
 *   Steps:  Q = query_in @ W_q; K = kv_in @ W_k; V = kv_in @ W_v; per head:
 *           scores = Q_h K_h^T / sqrt(head_dim) + block mask; softmax;
 *           attn_h = probs @ V_h; concat heads; output = attn @ W_o.
 *   arena is used for all intermediates and is not reset by the function.
 */
void blt_cross_attention_forward(const blt_tensor* query_in, const blt_tensor* kv_in,
                                const blt_cross_attention_weights* weights, blt_tensor* output,
                                const blt_cross_attention_config* config, blt_arena* arena);


/*
 * blt_cross_attention_backward(query_in, kv_in, weights, grad_out,
 *                              grad_query_in, grad_kv_in, grad_weights,
 *                              config, arena)
 *   Input:  same query_in/kv_in/weights/config as forward,
 *           grad_out [num_patches, embed_dim] FP32.
 *   Output: grad_query_in [num_patches, embed_dim]  (overwritten),
 *           grad_kv_in    [seq_len, embed_dim]      (overwritten),
 *           grad_weights  four [embed_dim, embed_dim] tensors (overwritten).
 *   Behavior: exact reverse of forward. Recomputes and caches the forward
 *   intermediates (Q, K, V, per-head attention probs) — the same recompute
 *   pattern as blt_multihead_attention_backward — then walks back through
 *   blt_matmul_backward / blt_softmax_backward. The only structural
 *   difference from self-attention backward is the two gradient sinks
 *   (grad_query_in and grad_kv_in) instead of one.
 */
void blt_cross_attention_backward(const blt_tensor* query_in, const blt_tensor* kv_in, 
                                const blt_cross_attention_weights* weights, const blt_tensor* grad_out, 
                                blt_tensor* grad_query_in, blt_tensor* grad_kv_in,
                                blt_cross_attention_grad* grad_weights, const blt_cross_attention_config* config,
                                blt_arena* arena);



#endif //BLT_CROSS_ATTENTION_H