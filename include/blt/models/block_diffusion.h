#ifndef BLT_MODELS_BLOCK_DIFFUSION_H
#define BLT_MODELS_BLOCK_DIFFUSION_H

#include <stddef.h>
#include <stdint.h>

#include "blt/core/tensor.h"
#include "blt/core/allocator.h"
#include "blt/models/local_decoder.h"
#include "blt/models/patcher.h"


// BLT-D block-wise diffusion training support (Fast-BLT 3.2).
//
// Decoder sequence layout during a diffusion training step:
//   [ clean prefix x (N rows) ; corrupted blocks x_block^t (B*(M-1) rows) ]
//
// Deviations from the paper, both documented and intentional:
//   - Clean-row D_0 stays the encoder h_final states (this repo's BLT
//     convention) instead of fresh Embed(x); block rows follow the
//     per-call d0 policy like draft rows do.
//   - [PAD] cells reuse the MASK token id (256) for their embedding and
//     are excluded from L_mask via cell_valid instead of carrying a
//     separate PAD embedding. Keeps BLT_D0_VOCAB at 257.
//   - Self-attention follows the Figure 5 matrix exactly: block rows see
//     all clean bytes, all EARLIER blocks fully, own block bidirectionally
//     (the figure supersedes the looser conditioning text in Eq. 6; this
//     also matches semi-autoregressive inference where earlier unmasked
//     blocks are visible context).

#define BLT_MASK_TOKEN_ID 256u   // in [0, BLT_D0_VOCAB)


//----------------------------------------------------------------------
// Preprocessing (Fast-BLT §3.2.1)

typedef struct {
    size_t num_clean;      // N
    size_t block_size;     // B
    size_t num_blocks;     // M-1 (first patch excluded)
    size_t n_block_rows;   // B * (M-1)
    float t;               // sampled timestep in (0, 1]

    uint32_t* tokens;      // [n_block_rows] corrupted ids: byte value or BLT_MASK_TOKEN_ID (PAD too)
    size_t* positions;     // [n_block_rows] original byte positions (PAD cells clamp to N-1)
    uint8_t* targets;      // [n_block_rows] original bytes (PAD cells: 0, never used in loss)
    uint8_t* cell_valid;   // [n_block_rows] 0 for PAD cells (excluded from L_mask)
    uint8_t* cell_masked;  // [n_block_rows] 1 iff this cell is currently [MASK]
    size_t* groups;        // [n_block_rows] cross-attn group id == block index j
                           // (block j belongs to patch j+1 and attends latent o_j,
                           // i.e. the paper's o_{i-1} rule)
    float loss_scale;      // multiplier on L_mask (paper Eq. 7 uses 1.0); set to 0
                           // mid-training for warmup schedules. Weight only --
                           // masking probabilities are fixed at build time.
} blt_block_batch;


// Builds one corrupted training example.
//
// Blocks follow Fast-BLT §3.2.1: for every patch except the first, block
// j = B consecutive bytes starting at that patch's start index, padded with
// [PAD] beyond the sequence end, original positions recorded for RoPE.
// A timestep t ~ U(0,1) is drawn from the seeded RNG and each valid cell is
// replaced by [MASK] independently with probability t. Deterministic given
// (bytes, patches, B, rng_seed).
//
// Requires num_patches >= 2 (a single patch produces no blocks).
void blt_block_batch_build(blt_block_batch* out, blt_arena* arena,
                           const uint8_t* bytes, size_t N,
                           const blt_patch_info* patches, size_t num_patches,
                           size_t B, uint64_t rng_seed);


//----------------------------------------------------------------------
// Diffusion decoder forward / backward (Fast-BLT §3.2.2 / §3.2.3)

// Forward pass over [clean ; corrupted-blocks].
//
//   byte_hidden_in [N, embed_dim]      encoder h_final for the clean prefix
//   patch_in       [num_patches, pdim] frozen global latents O
//   clean_bytes    [N]                 raw clean bytes; shifted CE targets
//                                      for L_clean
//   batch          corrupted block section spec (built above)
//   d0_mode        D_0 policy for BLOCK rows (ZEROS default / LEARNED table)
//   logits_out     [N + n_block_rows, vocab_size]
//   loss_out       scalar: L_clean + L_mask/t   (Fast-BLT Eq. 7)
//
// Cross-attention reuses the standard group mechanism (clean rows: own
// patch; block rows: batch->groups). Self-attention uses the Figure 5
// matrix via blt_build_block_diffusion_mask(TRAIN). RoPE uses the recorded
// original positions for block rows (identity for clean rows).
void blt_local_decoder_forward_diffusion(
    const blt_local_decoder* model,
    const blt_tensor* byte_hidden_in,
    const blt_tensor* patch_in,
    const blt_patch_info* patches, size_t num_patches,
    const uint8_t* clean_bytes,
    const blt_block_batch* batch,
    blt_d0_mode d0_mode,
    blt_tensor* logits_out,
    blt_tensor* loss_out,
    blt_arena* arena
);

// Backward pass mirroring the forward above.
//
//   grad_byte_hidden_in [N, embed_dim] — dL/d(h_final), clean rows only
//                       (block-row D_0 gradients go to d0_embed_grad when
//                       d0_mode == LEARNED; they are discarded otherwise)
//   grad_patch_in       [num_patches, pdim] — accumulated over firing layers
//   grad                parameter gradients (lm_head, layers, d0 table)
void blt_local_decoder_backward_diffusion(
    const blt_local_decoder* model,
    const blt_tensor* byte_hidden_in,
    const blt_tensor* patch_in,
    const blt_patch_info* patches, size_t num_patches,
    const uint8_t* clean_bytes,
    const blt_block_batch* batch,
    blt_d0_mode d0_mode,
    blt_tensor* grad_byte_hidden_in,
    blt_tensor* grad_patch_in,
    blt_local_decoder_grad* grad,
    blt_arena* arena
);

// Inference-mode diffusion forward (Fast-BLT section 3.1.1): clean rows
// causal, block rows bidirectional over the whole sequence; no loss.
// Caller drives unmasking via batch->tokens/cell_masked across passes.
// See src/models/block_diffusion.c for the batch contract.
void blt_local_decoder_forward_diffusion_infer(
    const blt_local_decoder* model,
    const blt_tensor* byte_hidden_in,
    const blt_tensor* patch_in,
    const blt_patch_info* patches, size_t num_patches,
    const blt_block_batch* batch,
    blt_d0_mode d0_mode,
    blt_tensor* logits_out,
    blt_arena* arena
);

#endif // BLT_MODELS_BLOCK_DIFFUSION_H
