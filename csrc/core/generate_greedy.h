// tools/generate_greedy.h
#ifndef BLT_TOOLS_GENERATE_GREEDY_H
#define BLT_TOOLS_GENERATE_GREEDY_H

#include <stdint.h>
#include "core/allocator.h"
#include "models/model.h"
#include "models/entropy_lm.h"
#include "models/patcher.h"

// Repeatedly re-runs the entropy model + patcher + blt_model_forward
// over the growing byte sequence
// and greedily appends argmax(logits_out[-1]) each step.
void blt_generate_greedy(const blt_model *model, const blt_entropy_lm *entropy_model,
                         const blt_patcher_config *patcher_config, const uint8_t *prompt_bytes, size_t prompt_len,
                         size_t max_new_bytes,
                         uint8_t *output_bytes, // caller-allocated, size >= prompt_len + max_new_bytes
                         blt_arena *arena);

#endif // BLT_TOOLS_GENERATE_GREEDY_H