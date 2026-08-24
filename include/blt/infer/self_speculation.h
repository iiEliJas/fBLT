#ifndef BLT_INFER_SELF_SPECULATION_H
#define BLT_INFER_SELF_SPECULATION_H

#include <stddef.h>

#include "blt/core/tensor.h"
#include "blt/core/allocator.h"
#include "blt/models/model.h"
#include "blt/models/entropy_lm.h"
#include "blt/models/patcher.h"
#include "blt/infer/stats.h"


// BLT-S configuration (Fast-BLT 5.1, Algorithm 2).
typedef struct {
    size_t window_k;      // speculative draft window; typical sweep {4, 8, 16}
    blt_d0_mode d0_mode;  // D_0 policy for draft rows; ZEROS default, LEARNED reads
                          // the decoder-owned d0_embed_weight[token=drafted byte]
} blt_self_spec_config;


// Algorithm 2: Verify(x, x', l, r).
//
// Verifies the draft bytes x[l .. l+r) against a full forward pass over the
// candidate sequence x[0 .. l+r): the entropy patcher re-segments the whole
// candidate, encoder+global run once, and the decoder produces next-byte
// predictions y_j = argmax(logits row j). Drafted bytes are accepted up to
// the first mismatch, which is replaced with the verified prediction; on a
// full match one extra free byte (the prediction after the last draft
// position) is appended.
//
// Progress guarantee: returns >= l+1 (at least one committed byte per call).
// Budget guarantee: the committed length never exceeds target_len; the free
// byte is skipped when l + r already reached it.
//
// Input/output: x holds the committed prefix x[0..l) plus the draft
// x[l..l+r) on entry; on exit x[0..returned_len) is the updated committed
// sequence. Everything beyond returned_len is scratch.
//
// Returns the new committed length (in [l+1, l+r+1]).
size_t blt_verify_draft(
    const blt_model* model,
    const blt_entropy_lm* entropy_model,
    const blt_patcher_config* patcher_config,
    uint8_t* x,
    size_t l,
    size_t r,
    size_t target_len,
    blt_infer_stats* stats,              // nullable
    blt_arena* arena
);


// Greedy generation with BLT self-speculation (Fast-BLT 5.1).
//
// Per round: segment + encode the committed prefix once (frozen latents),
// draft up to window_k bytes with decoder-only passes conditioning on the
// last latent token, then verify via blt_verify_draft. Output is intended
// to be byte-identical to blt_generate_greedy (greedy verification);
// tests/enforce this gate.
//
// Output: output_bytes receives exactly prompt_len + max_new_bytes bytes.
void blt_generate_greedy_selfspec(
    const blt_model* model,
    const blt_entropy_lm* entropy_model,
    const blt_patcher_config* patcher_config,
    const uint8_t* prompt_bytes,
    size_t prompt_len,
    size_t max_new_bytes,
    uint8_t* output_bytes,
    const blt_self_spec_config* config,
    blt_infer_stats* stats,              // nullable
    blt_arena* arena
);

#endif // BLT_INFER_SELF_SPECULATION_H
