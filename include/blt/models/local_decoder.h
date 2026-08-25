#ifndef BLT_LOCAL_DECODER_H
#define BLT_LOCAL_DECODER_H

#ifdef __cplusplus
extern "C" {
#endif

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
    bool cross_attn_all_layers;     // paper finding: decoder wants "All Layers" (Table 7), unlike encoder's "Last Layer" default — still configurable, not hardcoded
    blt_xattn_placement cross_attn_placement; // sweep option; DEFAULT = use the bool above
    float rope_theta;
    size_t max_seq_len;
    size_t vocab_size;              // 256, for the LM head
} blt_local_decoder_config;


//----------------------------------------------------------------------
// D_0 policy for rows without an encoder h_final state.
//
// The decoder's canonical input D_0 is the encoder's per-byte hidden
// state h_final. Speculative draft bytes (BLT-S) and masked block rows
// (BLT-D/DV) have no h_final -- they sit beyond the encoded prefix --
// so their D_0 row is chosen by this policy. This only affects draft
// quality / acceptance rate: the verification step guarantees output
// equivalence regardless of the choice.

#define BLT_D0_VOCAB 257   // 256 byte values + MASK token at index 256

typedef enum {
    BLT_D0_HFINAL = 0,  // legacy: every row of byte_hidden_in is real h_final
    BLT_D0_ZEROS,       // rows >= num_hfinal_rows get zero vectors
    BLT_D0_LEARNED      // rows >= num_hfinal_rows read d0_embed_weight[token]
} blt_d0_mode;

typedef struct {
    blt_d0_mode d0_mode;             // policy for rows >= num_hfinal_rows
    size_t num_hfinal_rows;          // rows [0, num_hfinal_rows) are real h_final from byte_hidden_in;
                                     // ignored (treated as seq_len) when d0_mode == BLT_D0_HFINAL
    const uint32_t* d0_extra_tokens; // [seq_len - num_hfinal_rows] token ids in [0, BLT_D0_VOCAB);
                                     // required iff d0_mode == BLT_D0_LEARNED
} blt_local_decoder_d0_opts;


// Per-layer weights: shared layout, see blt/models/local_common.h
// (cross-attention block runs first in decoder layers, byte transformer second)
typedef blt_local_layer_storage blt_local_decoder_layer_storage;


typedef struct {
    blt_local_decoder_config config;
    blt_tensor rope_cos_cache, rope_sin_cache;
    blt_local_decoder_layer_storage* layers;   // [num_layers]
    blt_tensor lm_head_weight;                  // [embed_dim, vocab_size]
    blt_tensor d0_embed_weight;                 // [BLT_D0_VOCAB, embed_dim] — D_0 rows for
                                                // draft/MASK positions when d0_mode == BLT_D0_LEARNED;
                                                // row 256 doubles as the MASK embedding.
                                                // Gradient written by the diffusion backward;
                                                // the legacy decoder backward leaves it zero, so training
                                                // leaves it untouched.
} blt_local_decoder;


// Gradient counterpart of blt_local_decoder_layer_storage (shared layout)
typedef blt_local_layer_grad blt_local_decoder_layer_grad;

typedef struct {
    blt_local_decoder_layer_grad* layer_grads;   // [num_layers]
    blt_tensor lm_head_grad;
    blt_tensor d0_embed_grad;                    // [BLT_D0_VOCAB, embed_dim] (zero except under the diffusion backward)
} blt_local_decoder_grad;




// Input:     arena for allocation, model config parameters.
// Output:    Allocated local decoder struct pointer.
// Behavior: Allocates zero-initialized weights, precomputes shared RoPE cache.
blt_local_decoder* blt_local_decoder_create(blt_arena* arena, const blt_local_decoder_config* config);


// Input:     arena for allocation, target model pointer.
// Output:    Allocated zero-initialized local decoder gradient struct pointer.
// Behavior: Allocates a gradient structure mirroring the target model's dimensions.
blt_local_decoder_grad* blt_local_decoder_grad_create(blt_arena* arena, const blt_local_decoder* model);


// Legacy entry point: D_0 = byte_hidden_in for every row, loss always computed.
// Thin wrapper around blt_local_decoder_forward_ext with NULL opts.
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


// Extended decoder forward (Fast-BLT).
// - loss_out may be NULL: skips the shifted cross-entropy entirely; bytes_in
//   must then also be NULL (logits-only inference path).
// - d0_opts may be NULL: legacy behavior (D_0 = byte_hidden_in everywhere).
//   Otherwise byte_hidden_in supplies only its first num_hfinal_rows; rows
//   beyond that get their D_0 from the configured policy. Extra rows are
//   causally masked against nothing on their right (they sit at sequence
//   positions num_hfinal_rows..seq_len-1) and condition on the LAST patch's
//   latent sub-tokens via the cross-attention group mechanism.
// Numerics for rows [0, num_hfinal_rows) are bit-identical to running the
// legacy call on that prefix alone (causality guarantee).
void blt_local_decoder_forward_ext(
    const blt_local_decoder* model,
    const blt_tensor* byte_hidden_in,   // [seq_len, embed_dim]; only first num_hfinal_rows read in non-HFINAL modes
    const blt_tensor* patch_in,         // [num_patches, patch_dim]
    const blt_patch_info* patches,      // must tile [0, num_hfinal_rows) exactly
    size_t num_patches,
    const blt_tensor* bytes_in,         // [seq_len] UINT8 or NULL (iff loss_out == NULL)
    const size_t* doc_boundaries,       // byte-indexed
    size_t num_docs,
    const blt_local_decoder_d0_opts* d0_opts,
    blt_tensor* logits_out,             // [seq_len, vocab_size]
    blt_tensor* loss_out,               // scalar or NULL
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
#ifdef __cplusplus
}
#endif

#endif // BLT_LOCAL_DECODER_H