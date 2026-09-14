#ifndef BLT_MODEL_BUILDER_H
#define BLT_MODEL_BUILDER_H

#include "core/tensor.h"
#include "models/model.h"
#include "models/entropy_lm.h"
#include "models/patcher.h"

#ifdef __cplusplus
extern "C" {
#endif

// Fill blt_model_config with the standard fBLT defaults.
// All magic numbers (num_heads=4, rope_theta=500000, ngram config, etc.)
// are defined here — callers only pass the varying shape parameters.
//
// Parameters:
//   cfg          — output config (memset to 0 by this function)
//   embed        — model embed dim
//   hidden       — model hidden dim
//   enc_layers   — encoder layer count
//   glob_layers  — global transformer layer count
//   dec_layers   — decoder layer count
//   max_seq_len  — max sequence length (typically 512)
//   cross_last   — if true, cross-attention only after final local layer
void blt_model_config_defaults(blt_model_config *cfg,
                               size_t embed, size_t hidden,
                               size_t enc_layers, size_t glob_layers,
                               size_t dec_layers, size_t max_seq_len,
                               int cross_last);

// Create a standard entropy-LM with random weights (seeded by `seed`).
// If load_path is non-NULL, loads pretrained weights from that file.
//
// Parameters:
//   arena     — memory arena for allocation
//   ms        — max_seq_len for the entropy LM config (typically 512)
//   load_path — optional path to pretrained weights (NULL = skip)
//   seed      — RNG seed for random weight initialization
blt_entropy_lm *blt_make_entropy_lm(blt_arena *arena, size_t ms,
                                    const char *load_path, uint64_t seed);

// Fill blt_patcher_config with standard defaults.
//
// Parameters:
//   pcfg               — output config
//   fixed              — if true, fixed-stride-4 patching (thresholds=1e9, max_patch=4)
//   threshold_global   — global entropy threshold (use 2.5f for standard, 1e9f for fixed)
//   threshold_monotonic — monotonic threshold (use 1.0f for standard, 1e9f for fixed)
//   max_patch_length   — max patch size (use 16 for standard, 4 for fixed)
void blt_make_patcher_cfg(blt_patcher_config *pcfg,
                          int fixed,
                          float threshold_global,
                          float threshold_monotonic,
                          size_t max_patch_length);

// Fill a tensor with uniform random values in [-scale, +scale].
// Requires t->data to be allocated and t->numel to be set.
void blt_fill_small_uniform(blt_tensor *t, float scale);

#ifdef __cplusplus
}
#endif

#endif // BLT_MODEL_BUILDER_H
