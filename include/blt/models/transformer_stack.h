#ifndef BLT_MODELS_TRANSFORMER_STACK_H
#define BLT_MODELS_TRANSFORMER_STACK_H


#ifdef __cplusplus
extern "C" {
#endif
#include "blt/core/tensor.h"
#include "blt/core/allocator.h"
#include "blt/models/transformer.h"

// A stack of N identical RMSNorm/SwiGLU transformer blocks sharing one
// RoPE cache, with support for both plain forward and cached
// forward+backward.

typedef struct {
    // Shared per-layer/per-call template. Fully populated by
    // blt_transformer_stack_init() except the RoPE cos/sin views, which
    // are seq_len-sized and therefore vary per call — see
    // blt_transformer_stack_call_config(). Callers that also need a mask
    // (e.g. global_transformer's block-causal doc mask) set
    // cfg.attn_config.mask_config on the config returned from that helper.
    blt_transformer_config layer_config;

    blt_transformer_layer_storage* layer_storage;  // owned weights, [num_layers]
    blt_transformer_weights* layer_weights;        // const-views into layer_storage, [num_layers]

    blt_tensor rope_cos_cache;  // [max_seq_len, head_dim/2]
    blt_tensor rope_sin_cache;  // [max_seq_len, head_dim/2]

    size_t num_layers;
    size_t embed_dim;
    size_t hidden_dim;
    size_t num_heads;
    size_t head_dim;
    size_t max_seq_len;
} blt_transformer_stack;

typedef struct {
    blt_transformer_layer_grad* layer_grads;  // [num_layers]
} blt_transformer_stack_grad;

// Opaque handle bridging blt_transformer_stack_forward_cached() to
// blt_transformer_stack_backward(). Definition is private to
// transformer_stack.c.
typedef struct blt_transformer_stack_cache blt_transformer_stack_cache;

typedef struct {
    size_t num_layers;
    size_t embed_dim;
    size_t hidden_dim;
    size_t num_heads;    // embed_dim must be divisible by num_heads
    size_t max_seq_len;  // upper bound used to size the shared RoPE cache
    float rope_theta;
} blt_transformer_stack_config;



// Allocates layer weights and precomputes the shared RoPE cache. Caller
// fills weight data afterward (same contract as blt_tensor_create).
void blt_transformer_stack_init(
    blt_arena* arena, blt_transformer_stack* stack,
    const blt_transformer_stack_config* config
);

// Allocates a zero-initialized gradient struct matching stack's shapes.
blt_transformer_stack_grad* blt_transformer_stack_grad_create(
    blt_arena* arena, const blt_transformer_stack* stack
);

// Builds the per-call config: copies stack->layer_config and overlays a
// seq_len-sized view over the shared RoPE cache (blt_multihead_attention
// validates the cache shape as exactly [seq_len, head_dim/2]). cos_view
// and sin_view must outlive the returned config. Callers needing a custom
// mask (block-causal, sliding window, etc.) build it separately and
// assign it onto the returned config's attn_config.mask_config.
blt_transformer_config blt_transformer_stack_call_config(
    const blt_transformer_stack* stack, size_t seq_len,
    blt_tensor* cos_view, blt_tensor* sin_view
);

// Plain forward, no caching — for call sites that don't need backward().
void blt_transformer_stack_forward(
    const blt_transformer_stack* stack, const blt_tensor* x,
    const blt_transformer_config* call_cfg, size_t seq_len,
    blt_tensor* out, blt_arena* arena
);

// Forward with per-layer caching. Must be paired with
// blt_transformer_stack_backward() using the same call_cfg/seq_len; the
// returned cache (and everything it points to) lives in arena.
blt_transformer_stack_cache* blt_transformer_stack_forward_cached(
    const blt_transformer_stack* stack, const blt_tensor* x,
    const blt_transformer_config* call_cfg, size_t seq_len,
    blt_arena* arena, blt_tensor* out
);

// Backward through the whole stack (reverse layer order), writing weight
// gradients into grad and dL/d(stack input) into grad_x.
void blt_transformer_stack_backward(
    const blt_transformer_stack* stack, const blt_transformer_stack_cache* cache,
    const blt_transformer_config* call_cfg, size_t seq_len,
    const blt_tensor* grad_out, blt_transformer_stack_grad* grad,
    blt_tensor* grad_x, blt_arena* arena
);


// Single transformer block (e.g. cross-attention in the local encoder/decoder) 
// rather than running a whole blt_transformer_stack in one shot
// Same block math and caching blt_transformer_stack uses
// internally - blt_transformer_stack_forward_cached/backward are now
// just a loop over these two calls.
typedef struct blt_transformer_layer_cache blt_transformer_layer_cache;

// Recomputes one transformer blocks forward pass (pre-norm attention +
// residual, pre-norm SwiGLU FFN + residual), caching intermediates needed
// by blt_transformer_layer_backward. The returned cache — including a
// copy of x, so the caller doesn't need to keep its own copy around for
// backward — lives in arena.
blt_transformer_layer_cache* blt_transformer_layer_forward_cached(
    const blt_tensor* x, const blt_transformer_weights* w, const blt_transformer_config* cfg,
    blt_arena* arena, size_t seq_len, size_t embed_dim, size_t hidden_dim,
    blt_tensor* out
);

// Backward through one transformer block. grad_out is dL/d(the forward
// call's out); writes dL/d(x) into grad_x and weight gradients into lg.
void blt_transformer_layer_backward(
    const blt_transformer_layer_cache* cache, const blt_transformer_weights* w, const blt_transformer_config* cfg,
    blt_arena* arena, size_t seq_len, size_t embed_dim, size_t hidden_dim,
    const blt_tensor* grad_out, blt_transformer_layer_grad* lg, blt_tensor* grad_x
);

#ifdef __cplusplus
}
#endif
#endif  // BLT_MODELS_TRANSFORMER_STACK_H
