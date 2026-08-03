#ifndef BLT_MODELS_PATCHER_H
#define BLT_MODELS_PATCHER_H

#include "blt/core/tensor.h"
#include "blt/models/entropy.h"

// dynamically sized patch of bytes
typedef struct {
    size_t start_idx;
    size_t length;        // number of bytes in this patch
    float peak_entropy;   // entropy value that triggered this patch
} blt_patch_info;

// Segments a sequence based on entropy spikes
// Input: 1D tensor of entropy values (shape: [seq_len])
// Output: Array of blt_patch_info structs
// Returns: The total number of patches created
size_t blt_segment_patches(const blt_tensor* entropy, blt_patch_info* patches_out, size_t max_patches, const blt_entropy_config* config);

#endif // BLT_MODELS_PATCHER_H