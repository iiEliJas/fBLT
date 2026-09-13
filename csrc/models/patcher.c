#include "models/patcher.h"
#include "core/backend.h"

#include <string.h>

static int blt_patch_boundary(size_t i, size_t patch_start, size_t patch_len, const float *entropy_data,
                              const uint8_t *bytes, const blt_patcher_config *config) {
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
        return (i > patch_start) && ((current_entropy - entropy_data[i - 1]) > config->threshold_monotonic);

    case BLT_PATCH_RULE_BOTH:
        return (current_entropy > config->threshold_global) ||
               ((i > patch_start) && ((current_entropy - entropy_data[i - 1]) > config->threshold_monotonic));

    default:
        return 0;
    }
}

static int blt_patch_emit(blt_patch_info *patches_out, size_t max_patches, size_t *patch_count, size_t start_idx,
                          size_t length, float peak_entropy) {
    if (*patch_count >= max_patches) {
        return 0;
    }
    patches_out[*patch_count].start_idx = start_idx;
    patches_out[*patch_count].length = length;
    patches_out[*patch_count].peak_entropy = peak_entropy;
    (*patch_count)++;
    return 1;
}

// Entropy-based patcher: divides bytes into patches using per-byte entropy.
// Boundary at index i if: newline reset, max length, H_i > threshold, or
// (H_i - H_{i-1}) > monotonic threshold.

size_t blt_segment_patches(const blt_tensor *entropy, const uint8_t *bytes, blt_patch_info *patches_out,
                           size_t max_patches, const blt_patcher_config *config) {
    blt_check_nd_fp32(entropy, 1, (const size_t[]){0}, "Entropy tensor must be 1D");
    BLT_REQUIRE(entropy->numel != 0, "Entropy tensor must not be empty");
    BLT_REQUIRE(config != NULL, "Config must not be null");
    BLT_REQUIRE(max_patches >= 1, "max_patches must be at least 1");

    size_t seq_len = entropy->numel;
    size_t patch_count = 0;
    const float *entropy_data = (const float *)entropy->data;

    size_t current_patch_start = 0;
    size_t current_patch_len = 1;
    float current_peak_entropy = entropy_data[0];

    for (size_t i = 1; i < seq_len; ++i) {
        if (blt_patch_boundary(i, current_patch_start, current_patch_len, entropy_data, bytes, config)) {
            if (!blt_patch_emit(patches_out, max_patches, &patch_count, current_patch_start, current_patch_len,
                                current_peak_entropy)) {
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

    if (!blt_patch_emit(patches_out, max_patches, &patch_count, current_patch_start, current_patch_len,
                        current_peak_entropy)) {
        BLT_WARN("Reached maximum number of patches (%zu) while closing final patch", max_patches);
    }

    return patch_count;
}