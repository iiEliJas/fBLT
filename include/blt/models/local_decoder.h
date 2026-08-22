#ifndef BLT_LOCAL_DECODER_H
#define BLT_LOCAL_DECODER_H

#include <stddef.h>
#include <stdbool.h>

#include "blt/core/tensor.h"
#include "blt/core/allocator.h"
#include "blt/ops/patch_pool.h"
#include "blt/models/patcher.h"
#include "blt/models/local_common.h"

typedef struct {
    size_t embed_dim;               // h_D
    size_t patch_dim;               // h_G
    size_t num_layers;              // l_D
    size_t hidden_dim;
    size_t num_heads;               // byte self-attention heads
    size_t cross_attn_heads;
    size_t local_window;            // 0 = full causal, else causal sliding window
    bool cross_attn_all_layers;     // paper finding: decoder wants "All Layers" (Table 7), unlike encoder's "Last Layer" default — still a config knob, not hardcoded
    float rope_theta;
    size_t max_seq_len;
    size_t vocab_size;              // 256, for the LM head
} blt_local_decoder_config;


// Per-layer weights: shared layout, see blt/models/local_common.h
// (cross-attention block runs first in decoder layers, byte transformer second)
typedef blt_local_layer_storage blt_local_decoder_layer_storage;


typedef struct {
    blt_local_decoder_config config;
    blt_tensor rope_cos_cache, rope_sin_cache;
    blt_local_decoder_layer_storage* layers;   // [num_layers]
    blt_tensor lm_head_weight;                  // [embed_dim, vocab_size]
} blt_local_decoder;


// Gradient counterpart of blt_local_decoder_layer_storage (shared layout)
typedef blt_local_layer_grad blt_local_decoder_layer_grad;

typedef struct {
    blt_local_decoder_layer_grad* layer_grads;   // [num_layers]
    blt_tensor lm_head_grad;
} blt_local_decoder_grad;




// Input:     arena for allocation, model config parameters.
// Output:    Allocated local decoder struct pointer.
// Behavior: Allocates zero-initialized weights, precomputes shared RoPE cache.
blt_local_decoder* blt_local_decoder_create(blt_arena* arena, const blt_local_decoder_config* config);


// Input:     arena for allocation, target model pointer.
// Output:    Allocated zero-initialized local decoder gradient struct pointer.
// Behavior: Allocates a gradient structure mirroring the target model's dimensions.
blt_local_decoder_grad* blt_local_decoder_grad_create(blt_arena* arena, const blt_local_decoder* model);


void blt_local_decoder_forward(
    const blt_local_decoder* model,
    const blt_tensor* byte_hidden_in,   // [seq_len, embed_dim] — h_final from local encoder
    const blt_tensor* patch_in,         // [num_patches, embed_dim] — O from global transformer
    const blt_patch_info* patches,
    size_t num_patches,
    const blt_tensor* bytes_in,         // [seq_len] UINT8 — needed only for the loss targets
    const size_t* doc_boundaries,       // byte-indexed
    size_t num_docs,
    blt_tensor* logits_out,             // [seq_len, vocab_size]
    blt_tensor* loss_out,               // scalar
    blt_arena* arena
);


void blt_local_decoder_backward(
    const blt_local_decoder* model,
    const blt_tensor* byte_hidden_in,
    const blt_tensor* patch_in,
    const blt_patch_info* patches,
    size_t num_patches,
    const blt_tensor* bytes_in,
    const size_t* doc_boundaries,
    size_t num_docs,
    blt_tensor* grad_byte_hidden_in,   // [seq_len, embed_dim] — feeds back into local encoder
    blt_tensor* grad_patch_in,         // [num_patches, embed_dim] — feeds back into global transformer
    blt_local_decoder_grad* grad,
    blt_arena* arena
);
#endif // BLT_LOCAL_DECODER_H