#ifndef BLT_INFER_BLOCK_GENERATION_H
#define BLT_INFER_BLOCK_GENERATION_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

#include "core/allocator.h"
#include "models/model.h"
#include "models/block_diffusion.h"
#include "models/entropy_lm.h"
#include "models/patcher.h"
#include "infer/stats.h"

//----------------------------------------------------------------------
// BLT-D / BLT-DV generation controllers (Fast-BLT section 3.1.2,
// Algorithm 1, and section 5.2).
//
// blt_draft_block runs the Algorithm 1 inner loop against frozen latents:
// start from a fully masked block, repeatedly run the diffusion decoder
// (INFER attention pattern), and unmask one or more cells per pass using
// the configured strategy until the block is complete.
//
// blt_generate_greedy_blockdiff is the outer Algorithm 1 loop without
// verification: per round it segments + encodes the committed prefix once,
// drafts a whole block, and accepts the draft verbatim (do_verify=false).
//
// blt_generate_greedy_blockdiff_verify is BLT-DV (section 5.2): the same
// drafting step, but each draft is verified by a full causal forward via
// blt_verify_draft (Algorithm 2), so every committed byte equals the plain
// greedy next-byte prediction at its position.
//----------------------------------------------------------------------

typedef enum {
    BLT_UNMASK_CONFIDENCE = 0, // unmask all cells with p_max >= alpha;
                               // fallback: single highest-confidence cell
    BLT_UNMASK_EB              // entropy-bounded: largest prefix of the
                               // ascending-entropy order with cumulative
                               // entropy <= gamma; fallback: lowest-entropy cell
} blt_unmask_strategy;

typedef struct {
    blt_unmask_strategy strategy;
    float threshold; // alpha for CONFIDENCE (probability), gamma for EB (nats)
    int use_top_p;   // 0 = greedy argmax predictions; 1 = sample within top-p
    float top_p;     // nucleus mass when use_top_p
    uint64_t seed;   // RNG seed for sampling (splitmix64 stream)
} blt_block_gen_opts;

typedef struct {
    size_t block_size;   // B (starting block size for adaptive mode)
    blt_d0_mode d0_mode; // D_0 policy for block rows during drafting
    blt_block_gen_opts opts;

    // Stage 6 add-ons (all default-off: zero/NULL/0 fields keep the
    // original Algorithm 1 / DV behavior).
    const blt_model *verifier_model; // NULL = self-verify (classic DV);
                                     // else verification runs through
                                     // this model (heterogeneous DV)
    int boundary_aligned;            // 1 = commit only at candidate patch
                                     // boundaries (see blt_verify_draft_aligned)
    size_t B_min;                    // adaptive-B lower bound (>= 1)
    size_t B_max;                    // adaptive-B upper bound; 0 disables
    float accept_target;             // rolling-acceptance target
    size_t adapt_window;             // rounds per rolling-average window
} blt_block_gen_config;

// Adaptive-B rule: given the current block size and the rolling acceptance
// over the last adapt_window rounds, return the next block size. Moves one
// byte at a time within [B_min, B_max]; grows when acceptance exceeds the
// target by more than ADAPT_MARGIN, shrinks when it falls short by the same.
// Pure helper over its arguments (unit-testable).
#define BLT_ADAPT_B_MARGIN 0.10f
size_t blt_block_adapt_b(size_t cur_b, double rolling_acceptance, size_t b_min, size_t b_max, float target);

// Selection kernels over the currently masked cells of one block row set.
// masked[b] != 0 marks a still-masked cell; scores[b] holds p_max
// (confidence strategy) or marginal entropy in nats (EB strategy).
// Returns a bitmask (bit b set = unmask this pass). Both guarantee at
// least one selected cell whenever any cell is masked (paper progress rule).
uint32_t blt_unmask_select_confidence(const float *scores, const uint8_t *masked, size_t B, float alpha);
uint32_t blt_unmask_select_eb(const float *scores, const uint8_t *masked, size_t B, float gamma);

// Algorithm 1 inner loop: draft the block that follows prefix[0..prefix_len)
// using the frozen encoder/global latents in enc (produced by
// blt_model_encode over exactly those prefix bytes and patches).
//
// out_block receives block_size drafted bytes (the caller clamps block_size
// to the remaining budget). Returns the number of decoder forward passes
// consumed. scratch is never reset inside -- enc and any caller state in
// this arena must survive; callers bound memory with round-level markers.
size_t blt_draft_block(const blt_model *model, const blt_model_enc_out *enc, const blt_patch_info *patches,
                       size_t num_patches, const uint8_t *prefix, size_t prefix_len, const blt_block_gen_config *config,
                       uint8_t *out_block, blt_arena *scratch);

// BLT-D generation (Algorithm 1 with do_verify=false): drafts are accepted
// as-is. Output length is always prompt_len + max_new_bytes.
void blt_generate_greedy_blockdiff(const blt_model *model, const blt_entropy_lm *entropy_model,
                                   const blt_patcher_config *patcher_config, const uint8_t *prompt_bytes,
                                   size_t prompt_len, size_t max_new_bytes, uint8_t *output_bytes,
                                   const blt_block_gen_config *config,
                                   blt_infer_stats *stats, // nullable
                                   blt_arena *scratch);

// BLT-DV generation (Algorithm 1 with do_verify=true): drafts are verified
// through blt_verify_draft before commitment, so output matches what plain
// greedy next-byte decoding would produce given the same committed prefix.
void blt_generate_greedy_blockdiff_verify(const blt_model *model, const blt_entropy_lm *entropy_model,
                                          const blt_patcher_config *patcher_config, const uint8_t *prompt_bytes,
                                          size_t prompt_len, size_t max_new_bytes, uint8_t *output_bytes,
                                          const blt_block_gen_config *config,
                                          blt_infer_stats *stats, // nullable
                                          blt_arena *scratch);

#ifdef __cplusplus
}
#endif

#endif // BLT_INFER_BLOCK_GENERATION_H
