// tools/generate_greedy.h
#ifndef BLT_TOOLS_GENERATE_GREEDY_H
#define BLT_TOOLS_GENERATE_GREEDY_H

#include <stdint.h>
#include "core/allocator.h"
#include "core/decode_select.h"
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

// Same loop, but the next byte is drawn with the temperature / top-p /
// repeat-penalty / no-repeat-ngram controls in opts. Pass the same defaults
// blt_decode_options_defaults fills in to reproduce greedy output exactly.
void blt_generate_sample(const blt_model *model, const blt_entropy_lm *entropy_model,
                         const blt_patcher_config *patcher_config, const uint8_t *prompt_bytes, size_t prompt_len,
                         size_t max_new_bytes, uint8_t *output_bytes, const blt_decode_options *opts, blt_arena *arena);

#endif // BLT_TOOLS_GENERATE_GREEDY_H

// Batched greedy/sampling entry point. Runs `num_samples` independent
// generations with per-sample seeds (seeds[si]) into outputs[si] (each
// caller-allocated, size >= prompt_len + max_new_bytes). opts may be NULL
// for plain argmax. Single-sample (num_samples == 1) delegates to the
// existing functions, so output is byte-identical.
void blt_generate_greedy_batched(const blt_model *model, const blt_entropy_lm *entropy_model,
                                 const blt_patcher_config *patcher_config, const uint8_t *prompt_bytes,
                                 size_t prompt_len, size_t max_new_bytes, const blt_decode_options *opts,
                                 size_t num_samples, const uint64_t *seeds, uint8_t **outputs, blt_arena *arena);
