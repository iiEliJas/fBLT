#ifndef BLT_MODELS_LOCAL_COMMON_H
#define BLT_MODELS_LOCAL_COMMON_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stddef.h>

#include "blt/core/tensor.h"
#include "blt/core/allocator.h"
#include "blt/models/transformer.h"
#include "blt/models/cross_attention.h"

//----------------------------------------------------------------
// Shared per-layer storage for the local encoder and local decoder.
//
// Both models compose each layer from the same two blocks -- a
// cross-attention block (RMSNorm -> QKV cross-attn) and a byte-level
// transformer block (causal self-attn + FFN) -- so the weight layout is
// defined once here.
//
// blt_local_encoder_layer_storage / blt_local_decoder_layer_storage and
// their _grad counterparts are aliases of these types; field access via
// either name is valid.

typedef struct {
    // byte transformer block (causal self-attn + FFN)
    blt_tensor norm1_weight;       // [embed_dim]
    blt_tensor attn_qkv_w;         // [embed_dim, 3 * embed_dim]
    blt_tensor attn_proj_w;        // [embed_dim, embed_dim]
    blt_tensor norm2_weight;       // [embed_dim]
    blt_tensor ffn_up_w;           // [embed_dim, hidden_dim]
    blt_tensor ffn_gate_w;         // [embed_dim, hidden_dim]
    blt_tensor ffn_down_w;         // [hidden_dim, embed_dim]
    // cross-attention block (allocated on every layer, fires per config)
    blt_tensor cross_norm_weight;  // [embed_dim]
    blt_tensor cross_weight_q;     // [embed_dim, embed_dim]
    blt_tensor cross_weight_k;     // [embed_dim, embed_dim]
    blt_tensor cross_weight_v;     // [embed_dim, embed_dim]
    blt_tensor cross_weight_proj;  // [embed_dim, embed_dim]
} blt_local_layer_storage;


// Gradient counterpart of blt_local_layer_storage
typedef struct {
    blt_tensor norm1_weight;
    blt_tensor attn_qkv_w;
    blt_tensor attn_proj_w;
    blt_tensor norm2_weight;
    blt_tensor ffn_up_w;
    blt_tensor ffn_gate_w;
    blt_tensor ffn_down_w;
    blt_tensor cross_norm_weight;
    blt_tensor cross_weight_q;
    blt_tensor cross_weight_k;
    blt_tensor cross_weight_v;
    blt_tensor cross_weight_proj;
} blt_local_layer_grad;


// Allocates all tensors of a shared layer storage (zero-initialized).
void blt_local_layer_storage_alloc(blt_arena* arena, blt_local_layer_storage* s,
                                   size_t embed_dim, size_t hidden_dim);

// Allocates all tensors of a shared layer grad (zero-initialized).
void blt_local_layer_grad_alloc(blt_arena* arena, blt_local_layer_grad* g,
                                size_t embed_dim, size_t hidden_dim);


// Cross-attention placement modes for the ablation sweeps
// BLT_XATTN_DEFAULT keeps the legacy bool behavior of cross_attn_all_layers;
// explicit modes override it. FIRST is only meaningful for the local decoder.
typedef enum {
    BLT_XATTN_DEFAULT = 0,
    BLT_XATTN_NONE,
    BLT_XATTN_LAST,
    BLT_XATTN_ALL,
    BLT_XATTN_FIRST
} blt_xattn_placement;

// Cross-attention fires on every layer when cross_attn_all_layers is set,
// otherwise only after the final layer. An explicit placement (anything but
// BLT_XATTN_DEFAULT) overrides the bool entirely.
bool blt_local_cross_attn_fires(blt_xattn_placement placement,
                                bool cross_attn_all_layers, size_t num_layers, size_t layer);


// Views of a shared layer's tensors as the op-level weight/grad structs
// expected by blt_transformer_forward/backward and blt_cross_attention_*.
blt_transformer_weights blt_local_byte_weights_view(const blt_local_layer_storage* s);
blt_cross_attention_weights blt_local_cross_weights_view(const blt_local_layer_storage* s);
blt_transformer_layer_grad blt_local_byte_grad_view(blt_local_layer_grad* lg);


// Builds the byte-transformer-layer config used by both models:
// RMSNorm + SwiGLU + causal RoPE'd self-attention over local_mask.
blt_transformer_config blt_local_byte_layer_config(
    size_t embed_dim, size_t num_heads, float rope_theta, size_t hidden_dim,
    const blt_mask_config* local_mask,
    const blt_tensor* rope_cos_view, const blt_tensor* rope_sin_view);

#ifdef __cplusplus
}
#endif

#endif // BLT_MODELS_LOCAL_COMMON_H
