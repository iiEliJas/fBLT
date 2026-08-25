#ifndef BLT_MODELS_PATCHER_H
#define BLT_MODELS_PATCHER_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

#include "blt/core/tensor.h"
#include "blt/models/entropy.h"


typedef enum {
    BLT_PATCH_RULE_GLOBAL,
    BLT_PATCH_RULE_MONOTONIC,
    BLT_PATCH_RULE_BOTH
} blt_patch_rule;

// dynamically sized patch of bytes
typedef struct {
    size_t start_idx;
    size_t length;        // number of bytes in this patch
    float peak_entropy;   // entropy value that triggered this patch
} blt_patch_info;

typedef struct {
    float threshold_global;
    float threshold_monotonic;
    size_t max_patch_length;
    blt_patch_rule rule;
    bool reset_on_newline;
} blt_patcher_config;


// Segments a sequence based on entropy spikes
// Input: 1D tensor of entropy values (shape: [seq_len])
// Output: Array of blt_patch_info structs
// Returns: The total number of patches created
size_t blt_segment_patches( const blt_tensor* entropy, const uint8_t* bytes, blt_patch_info* patches_out,
                            size_t max_patches, const blt_patcher_config* config
);

#ifdef __cplusplus
}
#endif

#endif // BLT_MODELS_PATCHER_H