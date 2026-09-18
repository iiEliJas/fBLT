#ifndef BLT_TRAIN_EVAL_H
#define BLT_TRAIN_EVAL_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>
#include "core/allocator.h"
#include "core/tensor.h"
#include "models/model.h"
#include "models/patcher.h"
#include "models/entropy_lm.h"
#include "models/block_diffusion.h"

// Mean next-byte CE (nats/byte) over the clean rows of one window, computed
// from the decoder logits. For BOTH training modes: with the Figure-5 plain-
// causal TRAIN mask, clean-row logits never depend on block rows, so this is
// the true causal BPB of the model in either case.
double window_causal_ce(blt_arena *arena, const blt_model *model, const uint8_t *text, size_t N,
                        const blt_block_batch *batch_or_null, int diffusion, blt_d0_mode d0m,
                        const blt_patch_info *patches, size_t M, size_t *masked_hits, size_t *masked_total);

// Fixed-stride patching helper.
size_t fixed_stride(size_t seq_len, size_t patch_len, blt_patch_info *out);

// Entropy-LM patching (same numerics as the inference controllers).
// The patcher is host-only, so under a device backend the byte ids are
// uploaded and the entropies staged back through host memory.
size_t entropy_segment(blt_arena *arena, blt_entropy_lm *lm, const uint8_t *bytes, size_t len, blt_patch_info *out,
                       size_t max_patches);

#ifdef __cplusplus
}
#endif

#endif // BLT_TRAIN_EVAL_H
