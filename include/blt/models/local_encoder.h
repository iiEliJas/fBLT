#ifndef BLT_LOCAL_ENCODER_H
#define BLT_LOCAL_ENCODER_H

#include <stddef.h>
#include <stdbool.h>

#include "blt/core/tensor.h"
#include "blt/core/allocator.h"
#include "blt/ops/patch_pool.h"
#include "blt/models/patcher.h"
#include "blt/models/hash_ngram.h"
#include "blt/models/local_common.h"

//----------------------------------------------------------------
// Local Encoder
//
// Bytes -> embeddings -> P_0 = pool(e_0), then l_E transformer layers.
// Patch reps update with block-diagonal cross-attention:
//   P_l = P_{l-1} + cross_attn_l(query = RMSNorm(P_{l-1}), kv = h_l)

typedef struct {
    size_t embed_dim;               // Hidden width (h_E)
    size_t num_layers;              // Byte transformer layers (l_E, default: 1)
    size_t patch_dim;               // Global width (h_G, default: 0 = same as embed_dim)
    size_t hidden_dim;              // FFN intermediate width
    size_t num_heads;               // Byte self-attention heads
    size_t cross_attn_heads;        // Cross-attention heads (U_E)
    size_t local_window;            // Window w_E for byte self-attn (0 = full causal)
    bool cross_attn_all_layers;     // false = cross-attn only after final layer
    blt_xattn_placement cross_attn_placement; // sweep knob; DEFAULT = use the bool above
    blt_patch_pool_type pool_type;  // Init strategy for P_0 (default: MEAN)
    blt_hash_ngram_config ngram_config; // embed_dim must match config.embed_dim
    float rope_theta;               // RoPE base frequency
    size_t max_seq_len;             // Maximum sequence length for RoPE cache
} blt_local_encoder_config;


// Per-layer weights: shared layout, see blt/models/local_common.h
typedef blt_local_layer_storage blt_local_encoder_layer_storage;


typedef struct {
    blt_local_encoder_config config;
    blt_tensor byte_embedding_weight;        // [256, embed_dim]
    blt_hash_ngram_weights ngram_weights;    // Hash n-gram lookup tables
    blt_tensor rope_cos_cache;               // [max_seq_len, head_dim / 2]
    blt_tensor rope_sin_cache;               // [max_seq_len, head_dim / 2]
    blt_local_encoder_layer_storage* layers; // [num_layers]
} blt_local_encoder;


// Gradient counterpart of blt_local_encoder_layer_storage (shared layout)
typedef blt_local_layer_grad blt_local_encoder_layer_grad;


// Gradient counterpart of blt_local_encoder.
// Note: embedding/ngram grads use scatter-add; layer grads are overwritten.
typedef struct {
    blt_tensor embedding_grad;               // [256, embed_dim]
    blt_hash_ngram_weights ngram_grads;        // Scatter-added gradient tables
    blt_local_encoder_layer_grad* layer_grads; // [num_layers]
} blt_local_encoder_grad;


// Input:     arena for allocation, model config parameters.
// Output:    Allocated local encoder struct pointer.
// Behavior: Allocates zero-initialized weights, initializes hash n-gram tables,
//            and precomputes shared RoPE cache. Caller fills weight data afterward.
blt_local_encoder* blt_local_encoder_create(blt_arena* arena, const blt_local_encoder_config* config);


// Input:     arena for allocation, target model pointer.
// Output:    Allocated zero-initialized local encoder gradient struct pointer.
// Behavior: Allocates a gradient structure mirroring the target model's dimensions.
blt_local_encoder_grad* blt_local_encoder_grad_create(blt_arena* arena, const blt_local_encoder* model);


// Input:     bytes_in [seq_len] UINT8, patches [num_patches] tiling [0, seq_len),
//            doc_boundaries/num_docs (optional), destination output tensors, arena.
// Output:    patch_out [num_patches, embed_dim] (P_final for global transformer),
//            byte_hidden_out [seq_len, embed_dim] (h_final, kept for decoder).
// Behavior: Embeds bytes + n-grams, pools initial patches (P_0), and passes through
//            byte transformer layers with block-diagonal cross-attention updates.
void blt_local_encoder_forward(const blt_local_encoder* model,
                               const blt_tensor* bytes_in,
                               const blt_patch_info* patches,
                               size_t num_patches,
                               const size_t* doc_boundaries,
                               size_t num_docs,
                               blt_tensor* patch_out,
                               blt_tensor* byte_hidden_out,
                               blt_arena* arena);


// Input:     Forward pass inputs, grad_patch_out [num_patches, embed_dim] (dL/dP_final),
//            grad_byte_hidden_out [seq_len, embed_dim] (dL/dh_final, optional), arena.
// Output:    grad struct populated with weight and embedding gradients.
// Behavior: Recomputes forward activations per layer, then walks in reverse executing
//            backwards for cross-attn, byte transformer, pooling, and embeddings.
void blt_local_encoder_backward(const blt_local_encoder* model,
                                const blt_tensor* bytes_in,
                                const blt_patch_info* patches,
                                size_t num_patches,
                                const size_t* doc_boundaries,
                                size_t num_docs,
                                const blt_tensor* grad_patch_out,
                                const blt_tensor* grad_byte_hidden_out,
                                blt_local_encoder_grad* grad,
                                blt_arena* arena);
                                
#endif // BLT_LOCAL_ENCODER_H