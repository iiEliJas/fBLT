#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "test_helpers.h"
#include "test_suite.h"

#include "core/allocator.h"
#include "core/backend.h"
#include "infer/stats.h"
#include "infer/rope_gather.h"
#include "ops/rope.h"

//----------------------------------------------------------------------
// Test: rope position gather.
//
// Gathered rows must be exact copies of the cache rows at the requested
// positions, in request order, with correct output shape.

int run_rope_position_gather_test(void) {
    blt_arena *arena = blt_arena_create(1024 * 1024, BLT_BACKEND_CPU);
    TEST_ASSERT(arena != NULL);

    const size_t max_seq_len = 16;
    const size_t head_dim = 8;
    const size_t half = head_dim / 2;

    size_t cache_shape[2] = {max_seq_len, half};
    blt_tensor cos_cache = blt_tensor_create(arena, cache_shape, 2, BLT_DTYPE_FP32);
    blt_tensor sin_cache = blt_tensor_create(arena, cache_shape, 2, BLT_DTYPE_FP32);

    blt_rope_config rope_cfg = {.theta = 10000.0f, .head_dim = head_dim};
    blt_rope_precompute(max_seq_len, &rope_cfg, &cos_cache, &sin_cache);

    const size_t n = 4;
    size_t positions[4] = {5, 0, 15, 5}; // includes repeat + boundaries

    blt_tensor cos_out, sin_out;
    blt_rope_position_gather(&cos_cache, &sin_cache, positions, n, &cos_out, &sin_out, arena);

    TEST_ASSERT(cos_out.ndim == 2 && cos_out.shape[0] == n && cos_out.shape[1] == half);
    TEST_ASSERT(sin_out.ndim == 2 && sin_out.shape[0] == n && sin_out.shape[1] == half);

    const float *cos_src = (const float *)cos_cache.data;
    const float *sin_src = (const float *)sin_cache.data;
    const float *cos_dst = (const float *)cos_out.data;
    const float *sin_dst = (const float *)sin_out.data;

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < half; j++) {
            if (cos_dst[i * half + j] != cos_src[positions[i] * half + j]) {
                TEST_ASSERT(!"cos row mismatch");
            }
            if (sin_dst[i * half + j] != sin_src[positions[i] * half + j]) {
                TEST_ASSERT(!"sin row mismatch");
            }
        }
    }

    blt_arena_destroy(arena);
    return 1;
}

//----------------------------------------------------------------------
// Test: infer stats accounting helpers.

int run_infer_stats_test(void) {
    blt_infer_stats stats;

    blt_infer_stats_reset(&stats);
    TEST_ASSERT(stats.nfe_encoder_global == 0 && stats.nfe_decoder == 0);
    TEST_ASSERT(stats.bytes_drafted == 0 && stats.bytes_accepted == 0);
    TEST_ASSERT(blt_infer_stats_acceptance_rate(&stats) == 0.0f);

    // simulate one BLT-S round: draft 8, accept 3
    stats.bytes_drafted = 8;
    stats.bytes_accepted = 3;
    float rate = blt_infer_stats_acceptance_rate(&stats);
    TEST_ASSERT(fabsf(rate - 0.375f) < 1e-6f);

    // simulate full acceptance
    blt_infer_stats_reset(&stats);
    stats.bytes_drafted = 16;
    stats.bytes_accepted = 16;
    TEST_ASSERT(fabsf(blt_infer_stats_acceptance_rate(&stats) - 1.0f) < 1e-6f);

    // print smoke (stdout)
    blt_infer_stats_print(&stats, "stats-smoke");

    return 1;
}
