#include "blt/models/patcher.h"
#include "blt/core/backend.h"

#include <string.h>



//-----------------------------------------------------
// Checks all patch rules
//
// Returns non-zero if a patch boundary should be placed at index i

static int blt_patch_boundary(size_t i, size_t patch_start, size_t patch_len,
                               const float* entropy_data, const uint8_t* bytes,
                               const blt_patcher_config* config)
{
    // Rule 1: Structural context reset
    if (bytes && config->reset_on_newline && bytes[i] == 0x0A) {
        return 1;
    }

    // Rule 2: Max patch length
    if (patch_len >= config->max_patch_length) {
        return 1;
    }

    // Rules 3 & 4: Entropy-based threshold
    float current_entropy = entropy_data[i];

    switch (config->rule) {
        case BLT_PATCH_RULE_GLOBAL:
            return current_entropy > config->threshold_global;

        case BLT_PATCH_RULE_MONOTONIC:
            return (i > patch_start) &&
                   ((current_entropy - entropy_data[i - 1]) > config->threshold_monotonic);

        case BLT_PATCH_RULE_BOTH:
            return (current_entropy > config->threshold_global) ||
                   ((i > patch_start) &&
                    ((current_entropy - entropy_data[i - 1]) > config->threshold_monotonic));

        default:
            return 0;
    }
}



//-----------------------------------------------------
// Appends the in-progress patch to the output array
//
// Returns 1 on success, 0 if max_patches was already reached

static int blt_patch_emit(blt_patch_info* patches_out, size_t max_patches, size_t* patch_count,
                           size_t start_idx, size_t length, float peak_entropy)
{
    if (*patch_count >= max_patches) {
        return 0;
    }
    patches_out[*patch_count].start_idx = start_idx;
    patches_out[*patch_count].length = length;
    patches_out[*patch_count].peak_entropy = peak_entropy;
    (*patch_count)++;
    return 1;
}



//-----------------------------------------------------
// Dynamic Entropy Patcher 
//
// Divides byte sequence X into patches based on per-byte entropy H_i:
// H_i = - sum [ P(x_i = v) * log2 P(x_i = v) ]
//
// Boundary triggers at index i if any active condition is true:
//      1. Newline Reset: byte == '\n' (0x0A) and reset_on_newline enabled
//      2. Max Length:    patch_length >= max_patch_length
//      3. Global Rule:   H_i > threshold_global
//      4. Relative Delta: (H_i - H_{i-1}) > threshold_monotonic

size_t blt_segment_patches(const blt_tensor* entropy, const uint8_t* bytes, blt_patch_info* patches_out,
                            size_t max_patches, const blt_patcher_config* config){
    // Validation
    blt_check_nd_fp32(entropy, 1, (const size_t[]){0}, "Entropy tensor must be 1D");
    BLT_REQUIRE(entropy->numel != 0, "Entropy tensor must not be empty");
    BLT_REQUIRE(config != NULL, "Config must not be null");

    size_t seq_len = entropy->numel;
    size_t patch_count = 0;
    const float* entropy_data = (const float*)entropy->data;

    size_t current_patch_start = 0;
    size_t current_patch_len = 1;
    float current_peak_entropy = entropy_data[0];

    for (size_t i = 1; i < seq_len; ++i) {
        if (blt_patch_boundary(i, current_patch_start, current_patch_len, entropy_data, bytes, config)) {
            if (!blt_patch_emit(patches_out, max_patches, &patch_count,
                                 current_patch_start, current_patch_len, current_peak_entropy)) {
                BLT_WARN("Reached maximum number of patches (%zu)", max_patches);
                break;
            }

            current_patch_start = i;
            current_patch_len = 1;
            current_peak_entropy = entropy_data[i];
        } else {
            current_patch_len++;
            if (entropy_data[i] > current_peak_entropy) {
                current_peak_entropy = entropy_data[i];
            }
        }
    }

    // Save remaining patch
    if (!blt_patch_emit(patches_out, max_patches, &patch_count,
                         current_patch_start, current_patch_len, current_peak_entropy)) {
        BLT_WARN("Reached maximum number of patches (%zu) while closing final patch", max_patches);
    }

    return patch_count;
}