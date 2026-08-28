#ifndef BLT_MODELS_GLOBAL_TRANSFORMER_H
#define BLT_MODELS_GLOBAL_TRANSFORMER_H

#ifdef __cplusplus
extern "C" {
#endif

#include "blt/core/tensor.h"
#include "blt/core/allocator.h"
#include "blt/models/transformer.h"
#include "blt/models/transformer_stack.h"

//----------------------------------------------------------------------
// Global Patch Transformer (BLT §3.1)
//
// Standard decoder-only transformer over the patch sequence, block-causal
// at the patch level: patch j attends to all patches <= j in the same
// document, never across documents. No cross-attention here — that's the
// local encoder's/decoder's job. Structurally this is just
// blt_transformer_forward stacked num_layers times with a block-causal
// patch mask, the same primitive blt_entropy_lm uses for bytes, applied
// to patches instead.
//----------------------------------------------------------------------

typedef struct {
    size_t embed_dim;    // h_G
    size_t num_layers;   // l_G
    size_t hidden_dim;   // FFN width
    size_t num_heads;
    float  rope_theta;
    size_t max_seq_len;  // max on num_patches
    bool   use_bf16;     // mixed-precision matmuls via bf16 weight copies
} blt_global_transformer_config;

typedef struct {
    blt_global_transformer_config config;
    blt_transformer_stack stack;    // N transformer layers with RoPE
} blt_global_transformer;

typedef struct {
    blt_transformer_stack_grad* stack_grad;  // [num_layers]
} blt_global_transformer_grad;



//----------------------------------------------------------------------
// Create / grad-create
//----------------------------------------------------------------------

// Allocates every weight tensor zero-initialized, including the shared
// RoPE cache (precomputed here via blt_rope_precompute, sized to
// max_seq_len, shared by pointer across every layer's attn_config).
// Caller fills weight data afterward.
blt_global_transformer* blt_global_transformer_create(
    blt_arena* arena, const blt_global_transformer_config* config);

// Allocates a zero-initialized gradient struct matching model's shapes.
blt_global_transformer_grad* blt_global_transformer_grad_create(
    blt_arena* arena, const blt_global_transformer* model);



//----------------------------------------------------------------------
// Forward
//----------------------------------------------------------------------
//
// patch_in:        [num_patches, embed_dim] FP32 — P_final from the local encoder.
// doc_boundaries:   patch-index boundaries (NOT byte-index — caller must map
//                    the local encoder's byte-indexed doc_boundaries through
//                    patches[] into patch-index boundaries before calling).
// patch_out:        [num_patches, embed_dim] FP32 — O, the contextualized
//                    patch representations. This is the final layer's raw
//                    output; there is no separate output projection.
//
// Builds a block-causal patch mask once (full causal, no sliding window,
// document-scoped via doc_boundaries) and runs num_layers transformer
// blocks over it, output of layer l-1 feeding layer l.
void blt_global_transformer_forward(
    const blt_global_transformer* model,
    const blt_tensor* patch_in,
    const size_t* doc_boundaries,
    size_t num_docs,
    blt_tensor* patch_out,
    blt_arena* arena);



//----------------------------------------------------------------------
// Backward
//----------------------------------------------------------------------
//
// Recompute-then-reverse:
// reruns forward with per-layer intermediates cached, then walks layers
// in reverse
void blt_global_transformer_backward(
    const blt_global_transformer* model,
    const blt_tensor* patch_in,
    const size_t* doc_boundaries,
    size_t num_docs,
    const blt_tensor* grad_patch_out,
    blt_tensor* grad_patch_in,
    blt_global_transformer_grad* grad,
    blt_arena* arena);


    
#ifdef __cplusplus
}
#endif

#endif // BLT_MODELS_GLOBAL_TRANSFORMER_H