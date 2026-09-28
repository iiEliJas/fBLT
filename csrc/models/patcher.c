#include "models/patcher.h"
#include "core/backend.h"

#include <string.h>

// Paper alignment: a patch starts at i iff H(x_i | x_<i) > threshold. Since
// entropy_data[j] = H(x_{j+1} | x_{0..j}), that is entropy_data[i-1]. All rules
// read only x[0..i-1], so "byte p is final" (a patch starts at p+1) is
// decidable from x[0..p]. Position 0 always opens a patch.
static int blt_patch_boundary(size_t i, size_t patch_start, size_t patch_len, const float *entropy_data,
                              const uint8_t *bytes, const blt_patcher_config *config) {
    // Newline: a patch ends after a newline byte, so the next starts at
    // newline_index+1; reads bytes[i-1], not bytes[i].
    if (i >= 1 && bytes && config->reset_on_newline && bytes[i - 1] == 0x0A) {
        return 1;
    }

    // Max patch length
    if (patch_len >= config->max_patch_length) {
        return 1;
    }

    const float cur = entropy_data[i - 1];
    const float prev = (i >= 2) ? entropy_data[i - 2] : cur;

    switch (config->rule) {
    case BLT_PATCH_RULE_GLOBAL:
        return cur > config->threshold_global;

    case BLT_PATCH_RULE_MONOTONIC:
        return (i >= patch_start + 2) && ((cur - prev) > config->threshold_monotonic);

    case BLT_PATCH_RULE_BOTH:
        return (cur > config->threshold_global) ||
               ((i >= patch_start + 2) && ((cur - prev) > config->threshold_monotonic));

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
// Boundary at index i if: byte i-1 is a newline, max length,
// H(x_i|x_<i)=entropy_data[i-1] > threshold, or
// (entropy_data[i-1] - entropy_data[i-2]) > monotonic threshold.

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

int blt_next_starts_patch(const uint8_t *bytes, size_t len, const float *entropy_data, size_t patch_start,
                          size_t patch_len, const blt_patcher_config *config) {
    BLT_REQUIRE(config != NULL, "blt_next_starts_patch: Config must not be null");
    BLT_REQUIRE(len >= 1, "blt_next_starts_patch: len must be >= 1");
    BLT_REQUIRE(entropy_data != NULL, "blt_next_starts_patch: entropy_data must not be null");
    // Position 0 always opens a patch, so there is no "previous" patch there.
    if (len == 0) return 0;
    return blt_patch_boundary(len, patch_start, patch_len, entropy_data, bytes, config);
}