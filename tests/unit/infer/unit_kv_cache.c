#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "test_helpers.h"
#include "test_suite.h"

#include "core/allocator.h"
#include "core/backend.h"
#include "models/local_decoder.h"
#include "models/local_common.h"
#include "infer/kv_cache.h"

//----------------------------------------------------------------------
// Harness: tiny decoder with deterministic random weights; fabricated
// h_final / patch latents so the KV-cache path can be compared against
// the dense decoder row-for-row.

static void fill_small_uniform(blt_tensor *t, float scale) {
    float *data = (float *)t->data;
    for (size_t i = 0; i < t->numel; i++) {
        float r = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
        data[i] = r * scale;
    }
}

static void fill_constant(blt_tensor *t, float value) {
    float *data = (float *)t->data;
    for (size_t i = 0; i < t->numel; i++) {
        data[i] = value;
    }
}

typedef struct {
    size_t embed_dim;
    size_t patch_dim;
    size_t local_window;
} dec_dims;

static blt_local_decoder *make_random_decoder(blt_arena *arena, const dec_dims *dims, float scale) {
    blt_local_decoder_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.embed_dim = dims->embed_dim;
    cfg.patch_dim = dims->patch_dim;
    cfg.num_layers = 1;
    cfg.hidden_dim = 32;
    cfg.num_heads = 2;
    cfg.cross_attn_heads = 2;
    cfg.local_window = dims->local_window;
    cfg.cross_attn_all_layers = true;
    cfg.rope_theta = 10000.0f;
    cfg.max_seq_len = 64;
    cfg.vocab_size = 256;

    blt_local_decoder *dec = blt_local_decoder_create(arena, &cfg);
    TEST_ASSERT(dec != NULL);

    blt_local_layer_storage *l = &dec->layers[0];
    fill_constant(&l->norm1_weight, 1.0f);
    fill_small_uniform(&l->attn_qkv_w, scale);
    fill_small_uniform(&l->attn_proj_w, scale);
    fill_constant(&l->norm2_weight, 1.0f);
    fill_small_uniform(&l->ffn_up_w, scale);
    fill_small_uniform(&l->ffn_gate_w, scale);
    fill_small_uniform(&l->ffn_down_w, scale);
    fill_constant(&l->cross_norm_weight, 1.0f);
    fill_small_uniform(&l->cross_weight_q, scale);
    fill_small_uniform(&l->cross_weight_k, scale);
    fill_small_uniform(&l->cross_weight_v, scale);
    fill_small_uniform(&l->cross_weight_proj, scale);
    fill_small_uniform(&dec->lm_head_weight, scale);
    return dec;
}

static void build_scenario(blt_arena *arena, const dec_dims *dims, size_t S, size_t patch_len,
                           blt_local_decoder **dec_out, blt_tensor *h_out, blt_tensor *patch_in_out,
                           blt_patch_info *patches_out, size_t *num_patches_out) {
    const size_t E = dims->embed_dim;
    const size_t pdim = (dims->patch_dim == 0) ? E : dims->patch_dim;

    *dec_out = make_random_decoder(arena, dims, 0.1f);

    size_t h_shape[2] = {S, E};
    *h_out = blt_tensor_create(arena, h_shape, 2, BLT_DTYPE_FP32);
    fill_small_uniform(h_out, 0.5f);

    // tile [0, S) first so patch_in can be sized exactly
    size_t np = 0;
    size_t start = 0;
    while (start < S) {
        size_t len = (start + patch_len <= S) ? patch_len : (S - start);
        patches_out[np].start_idx = start;
        patches_out[np].length = len;
        patches_out[np].peak_entropy = 0.0f;
        np++;
        start += len;
    }
    *num_patches_out = np;

    size_t pi_shape[2] = {np, pdim};
    *patch_in_out = blt_tensor_create(arena, pi_shape, 2, BLT_DTYPE_FP32);
    fill_small_uniform(patch_in_out, 0.5f);
}

// Runs the incremental decoder over rows [start..S) according to a chunk
// pattern (lengths summing to S-start) and writes concatenated logits into
// *out ([S, V]; rows before start are left untouched).
static int run_incremental(blt_arena *arena, blt_kv_cache *cache, const blt_tensor *patch_in,
                           const blt_patch_info *patches, size_t num_patches, const blt_tensor *h, size_t start,
                           const size_t *chunks, size_t num_chunks, size_t V, blt_tensor *out) {
    const size_t E = h->shape[1];
    const size_t S = h->shape[0];
    TEST_ASSERT(cache->self_len == start);

    size_t pos = start;
    for (size_t c = 0; c < num_chunks; c++) {
        TEST_ASSERT(chunks[c] >= 1);
        TEST_ASSERT(pos + chunks[c] <= S);

        size_t d0_shape[2] = {chunks[c], E};
        blt_tensor d0 = blt_tensor_create(arena, d0_shape, 2, BLT_DTYPE_FP32);
        memcpy(d0.data, (float *)h->data + pos * E, chunks[c] * E * sizeof(float));

        size_t lg_shape[2] = {chunks[c], V};
        blt_tensor lg = blt_tensor_create(arena, lg_shape, 2, BLT_DTYPE_FP32);
        blt_kv_decode_step(cache, patch_in, patches, num_patches, &d0, &lg, arena);

        memcpy((float *)out->data + pos * V, lg.data, chunks[c] * V * sizeof(float));
        pos += chunks[c];
    }
    TEST_ASSERT(cache->self_len == pos);
    return 1;
}

static int require_rows_equal(const blt_tensor *a, const blt_tensor *b, size_t from_row, size_t num_rows) {
    TEST_ASSERT(a->shape[1] == b->shape[1]);
    const float *x = (const float *)a->data;
    const float *y = (const float *)b->data;
    const size_t V = a->shape[1];
    TEST_ASSERT((from_row + num_rows) * V <= a->numel);
    TEST_ASSERT((from_row + num_rows) * V <= b->numel);
    for (size_t i = from_row * V; i < (from_row + num_rows) * V; i++) {
        if (x[i] != y[i]) {
            fprintf(stderr, "  mismatch at flat idx %zu (row %zu): %f vs %f\n", i, i / V, (double)x[i], (double)y[i]);
            (void)num_rows;
            TEST_ASSERT(!"kv-cache logits diverged from dense");
        }
    }
    return 1;
}

//----------------------------------------------------------------------
// Test 1: cached logits == dense logits under several processing orders
// (single chunk, per-row steps, chunk+singles with mid-run truncation),
// plus sliding-window and k-split decoder variants.

static int run_pattern_case(const dec_dims *dims) {
    srand(31);

    blt_arena *arena = blt_arena_create(8 * 1024 * 1024, BLT_BACKEND_CPU);
    TEST_ASSERT(arena != NULL);

    const size_t S = 12;
    const size_t PATCH = 4;
    const size_t V = 256;

    blt_local_decoder *dec;
    blt_tensor h, patch_in;
    blt_patch_info patches[8];
    size_t num_patches;
    build_scenario(arena, dims, S, PATCH, &dec, &h, &patch_in, patches, &num_patches);

    // Dense reference (legacy semantics: all rows are h_final)
    size_t ref_shape[2] = {S, V};
    blt_tensor ref = blt_tensor_create(arena, ref_shape, 2, BLT_DTYPE_FP32);
    blt_local_decoder_forward_ext(dec, &h, &patch_in, patches, num_patches, NULL, NULL, 0, NULL, &ref, NULL, arena);

    size_t got_shape[2] = {S, V};

    // Pattern A: one big chunk
    {
        blt_kv_cache *cache = blt_kv_cache_create(arena, dec, 32);
        blt_kv_cache_truncate(cache, 0, 0);
        blt_kv_cache_refresh_cross(cache, &patch_in, patches, num_patches, 0, arena);

        blt_tensor got = blt_tensor_create(arena, got_shape, 2, BLT_DTYPE_FP32);
        size_t chunks[1] = {S};
        TEST_ASSERT(run_incremental(arena, cache, &patch_in, patches, num_patches, &h, 0, chunks, 1, V, &got));
        TEST_ASSERT(require_rows_equal(&got, &ref, 0, S));
    }

    // Pattern B: first patch as a chunk, then single-row steps
    {
        blt_kv_cache *cache = blt_kv_cache_create(arena, dec, 32);
        blt_kv_cache_truncate(cache, 0, 0);
        blt_kv_cache_refresh_cross(cache, &patch_in, patches, num_patches, 0, arena);

        blt_tensor got = blt_tensor_create(arena, got_shape, 2, BLT_DTYPE_FP32);
        size_t chunks[4] = {PATCH, 1, 1, S - PATCH - 2};
        TEST_ASSERT(run_incremental(arena, cache, &patch_in, patches, num_patches, &h, 0, chunks, 4, V, &got));
        TEST_ASSERT(require_rows_equal(&got, &ref, 0, S));
    }

    // Pattern C: mid-run rollback -- process the first patch, truncate back
    // to it, refresh remaining tier-2 entries from scratch, continue.
    // Exercises truncate + partial refresh without changing results.
    {
        blt_kv_cache *cache = blt_kv_cache_create(arena, dec, 32);
        blt_kv_cache_truncate(cache, 0, 0);
        blt_kv_cache_refresh_cross(cache, &patch_in, patches, num_patches, 0, arena);

        blt_tensor part = blt_tensor_create(arena, got_shape, 2, BLT_DTYPE_FP32);
        size_t chunks1[1] = {PATCH};
        TEST_ASSERT(run_incremental(arena, cache, &patch_in, patches, num_patches, &h, 0, chunks1, 1, V, &part));
        TEST_ASSERT(require_rows_equal(&part, &ref, 0, PATCH));

        blt_kv_cache_truncate(cache, PATCH, 1);
        size_t common = blt_kv_cache_common_patches(cache, patches, num_patches);
        TEST_ASSERT(common == 1);
        blt_kv_cache_refresh_cross(cache, &patch_in, patches, num_patches, common, arena);

        blt_tensor rest = blt_tensor_create(arena, got_shape, 2, BLT_DTYPE_FP32);
        size_t chunks2[2] = {4, S - PATCH - 4};
        TEST_ASSERT(run_incremental(arena, cache, &patch_in, patches, num_patches, &h, PATCH, chunks2, 2, V, &rest));
        TEST_ASSERT(require_rows_equal(&rest, &ref, PATCH, S - PATCH));
    }

    blt_arena_destroy(arena);
    return 1;
}

int run_kv_cache_logits_equal_dense(void) {
    dec_dims plain = {.embed_dim = 16, .patch_dim = 0, .local_window = 0};
    dec_dims windowed = {.embed_dim = 16, .patch_dim = 0, .local_window = 4};
    dec_dims ksplit = {.embed_dim = 16, .patch_dim = 32, .local_window = 0};

    TEST_ASSERT(run_pattern_case(&plain));
    TEST_ASSERT(run_pattern_case(&windowed));
    TEST_ASSERT(run_pattern_case(&ksplit));
    return 1;
}

//----------------------------------------------------------------------
// Test 2: verification-commit simulation. A round runs against one
// segmentation; afterwards the trailing patch changes length (as happens
// when the patcher re-segments committed bytes). The cache must roll back
// to the LCP, refresh the tail patch, and reproduce dense logits for the
// new sequence while untouched prefix rows remain valid.

int run_kv_cache_rollback_commit(void) {
    srand(32);

    blt_arena *arena = blt_arena_create(8 * 1024 * 1024, BLT_BACKEND_CPU);
    TEST_ASSERT(arena != NULL);

    dec_dims dims = {.embed_dim = 16, .patch_dim = 0, .local_window = 0};
    const size_t E = dims.embed_dim;
    const size_t V = 256;

    // Round 1: prefix [0..10), patches [0,4) [4,8) [8,10)
    const size_t S1 = 10;
    blt_local_decoder *dec;
    blt_tensor h1, pin1;
    blt_patch_info p1[8];
    size_t np1;
    build_scenario(arena, &dims, S1, 4, &dec, &h1, &pin1, p1, &np1);
    TEST_ASSERT(np1 == 3 && p1[2].length == 2);

    blt_kv_cache *cache = blt_kv_cache_create(arena, dec, 32);
    blt_kv_cache_truncate(cache, 0, 0);
    blt_kv_cache_refresh_cross(cache, &pin1, p1, np1, 0, arena);

    size_t r1_shape[2] = {S1, V};
    blt_tensor r1 = blt_tensor_create(arena, r1_shape, 2, BLT_DTYPE_FP32);
    size_t one[1] = {S1};
    TEST_ASSERT(run_incremental(arena, cache, &pin1, p1, np1, &h1, 0, one, 1, V, &r1));

    // Round 2: prefix grows to [0..11); the last patch becomes [8,11)
    // while patches 0 and 1 stay identical (prefix-stable patcher).
    const size_t S2 = 11;
    size_t h2_shape[2] = {S2, E};
    blt_tensor h2 = blt_tensor_create(arena, h2_shape, 2, BLT_DTYPE_FP32);
    fill_small_uniform(&h2, 0.5f);
    memcpy(h2.data, h1.data, S1 * E * sizeof(float)); // shared history bytes

    size_t pin2_shape[2] = {np1, E};
    blt_tensor pin2 = blt_tensor_create(arena, pin2_shape, 2, BLT_DTYPE_FP32);
    fill_small_uniform(&pin2, 0.5f);
    memcpy(pin2.data, pin1.data, 2 * E * sizeof(float)); // completed patches keep latents

    blt_patch_info p2[8] = {
        {.start_idx = 0, .length = 4, .peak_entropy = 0.0f},
        {.start_idx = 4, .length = 4, .peak_entropy = 0.0f},
        {.start_idx = 8, .length = 3, .peak_entropy = 0.0f},
    };

    // Dense reference over the new sequence
    size_t ref2_shape[2] = {S2, V};
    blt_tensor ref2 = blt_tensor_create(arena, ref2_shape, 2, BLT_DTYPE_FP32);
    blt_local_decoder_forward_ext(dec, &h2, &pin2, p2, np1, NULL, NULL, 0, NULL, &ref2, NULL, arena);

    // Cache path: LCP against the previous round's boundaries must be 2
    size_t common = blt_kv_cache_common_patches(cache, p2, np1);
    TEST_ASSERT(common == 2);
    size_t self_valid = p2[common - 1].start_idx + p2[common - 1].length;
    TEST_ASSERT(self_valid == 8);
    blt_kv_cache_truncate(cache, self_valid, common);
    blt_kv_cache_refresh_cross(cache, &pin2, p2, np1, common, arena);

    // Prefill only the open patch's rows [8..11)
    size_t d0_shape[2] = {S2 - self_valid, E};
    blt_tensor d0 = blt_tensor_create(arena, d0_shape, 2, BLT_DTYPE_FP32);
    memcpy(d0.data, (float *)h2.data + self_valid * E, (S2 - self_valid) * E * sizeof(float));

    size_t lg_shape[2] = {S2 - self_valid, V};
    blt_tensor lg = blt_tensor_create(arena, lg_shape, 2, BLT_DTYPE_FP32);
    blt_kv_decode_step(cache, &pin2, p2, np1, &d0, &lg, arena);

    // Untouched rows [0..8) must equal BOTH the dense new reference and
    // what round 1 produced for them; new rows [8..11) must equal dense.
    TEST_ASSERT(require_rows_equal(&ref2, &r1, 0, self_valid));
    {
        const float *x = (const float *)ref2.data;
        const float *y = (const float *)lg.data;
        for (size_t i = 0; i < (S2 - self_valid) * V; i++) {
            if (x[(self_valid * V) + i] != y[i]) {
                TEST_ASSERT(!"post-commit rows diverged from dense");
            }
        }
    }

    blt_arena_destroy(arena);
    return 1;
}
