#ifndef BLT_MODELS_PATCHER_H
#define BLT_MODELS_PATCHER_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

#include "core/tensor.h"
#include "models/entropy.h"

typedef enum { BLT_PATCH_RULE_GLOBAL, BLT_PATCH_RULE_MONOTONIC, BLT_PATCH_RULE_BOTH } blt_patch_rule;

typedef struct {
    size_t start_idx;
    size_t length;
    float peak_entropy;
} blt_patch_info;

typedef struct {
    float threshold_global;
    float threshold_monotonic;
    size_t max_patch_length;
    blt_patch_rule rule;
    bool reset_on_newline;
} blt_patcher_config;

// Segments a byte sequence into patches using entropy thresholds.
// `entropy`: 1D [seq_len], `patches_out`: caller-allocated array, returns patch count.
size_t blt_segment_patches(const blt_tensor *entropy, const uint8_t *bytes, blt_patch_info *patches_out,
                           size_t max_patches, const blt_patcher_config *config);

// Whether a new patch would start at position `len`, given the prefix's last
// patch [patch_start, patch_start+patch_len) and the per-byte entropy over the
// prefix. Uses only entropy_data[len-1] and bytes[len-1] (never x_len), so byte
// len-1's finality is decidable from x[0..len) alone.
int blt_next_starts_patch(const uint8_t *bytes, size_t len, const float *entropy_data, size_t patch_start,
                          size_t patch_len, const blt_patcher_config *config);

#ifdef __cplusplus
}
#endif

#endif // BLT_MODELS_PATCHER_H