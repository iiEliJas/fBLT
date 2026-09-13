#include "test_helpers.h"
#include "test_suite.h"
#include "ops/mask_builder.h"
#include "ops/patch_pool.h"
#include "core/allocator.h"

static int is_allowed(const blt_tensor *mask, size_t seq_len_kv, size_t i, size_t j) {
    float *data = (float *)mask->data;
    return data[i * seq_len_kv + j] == 0.0f;
}

//----------------------------------------------------------------------------
// Plain causal mask test
static int run_mask_builder_causal_test(void) {
    blt_arena *arena = blt_arena_create(1024 * 1024, BLT_BACKEND_CPU);
    if (!arena) {
        return 0;
    }

    blt_mask_config config = {0};
    config.seq_len_q = 4;
    config.seq_len_kv = 4;
    config.is_causal = true;

    blt_tensor mask = {0};
    blt_build_attention_mask(&config, &mask, arena);

    TEST_ASSERT(mask.shape[0] == 4 && mask.shape[1] == 4);

    for (size_t i = 0; i < 4; i++) {
        for (size_t j = 0; j < 4; j++) {
            int expected_allowed = (j <= i);
            TEST_ASSERT(is_allowed(&mask, 4, i, j) == expected_allowed);
        }
    }

    blt_arena_destroy(arena);
    return 1;
}

//----------------------------------------------------------------------------
// Sliding window with causal masking test
// only the current position and the (window-1) positions before it should be allowed
static int run_mask_builder_sliding_window_test(void) {
    blt_arena *arena = blt_arena_create(1024 * 1024, BLT_BACKEND_CPU);
    if (!arena) {
        return 0;
    }

    blt_mask_config config = {0};
    config.seq_len_q = 6;
    config.seq_len_kv = 6;
    config.is_causal = true;
    config.sliding_window = 3; // query i can see keys i, i-1, i-2

    blt_tensor mask = {0};
    blt_build_attention_mask(&config, &mask, arena);

    size_t i = 5;
    TEST_ASSERT(is_allowed(&mask, 6, i, 5) == 1); // self
    TEST_ASSERT(is_allowed(&mask, 6, i, 4) == 1); // 1 back
    TEST_ASSERT(is_allowed(&mask, 6, i, 3) == 1); // 2 back
    TEST_ASSERT(is_allowed(&mask, 6, i, 2) == 0); // 3 back, outside window
    TEST_ASSERT(is_allowed(&mask, 6, i, 0) == 0); // far outside window

    blt_arena_destroy(arena);
    return 1;
}

//----------------------------------------------------------------------------
// Document boundariestest
// positions in different documents should never attend to each other, even without causal masking
static int run_mask_builder_doc_boundary_test(void) {
    blt_arena *arena = blt_arena_create(1024 * 1024, BLT_BACKEND_CPU);
    if (!arena) {
        return 0;
    }

    // Two documents: doc 0 = positions [0, 3), doc 1 = positions [3, 6).
    size_t doc_boundaries[1] = {3};

    blt_mask_config config = {0};
    config.seq_len_q = 6;
    config.seq_len_kv = 6;
    config.is_causal = false;
    config.doc_boundaries = doc_boundaries;
    config.num_docs = 2;

    blt_tensor mask = {0};
    blt_build_attention_mask(&config, &mask, arena);

    // Within doc 0.
    TEST_ASSERT(is_allowed(&mask, 6, 1, 2) == 1);
    // Within doc 1.
    TEST_ASSERT(is_allowed(&mask, 6, 4, 5) == 1);
    // Across the boundary, either direction.
    TEST_ASSERT(is_allowed(&mask, 6, 2, 3) == 0);
    TEST_ASSERT(is_allowed(&mask, 6, 5, 0) == 0);

    blt_arena_destroy(arena);
    return 1;
}

//------------------------------------------------------------------------
// K-split test
//
// Verifies blt_patch_expand_group_ids produces the correct repeated-id
// pattern, and that reinterpreting a [num_patches, patch_dim] tensor as
// [num_patches*k, E] and back is a lossless round-trip

static int run_patch_k_split(void) {
    blt_arena *arena = blt_arena_create(1024 * 1024, BLT_BACKEND_CPU);
    if (!arena) return 0;

    // --- group id expansion ---
    size_t group_ids_in[2] = {0, 1};
    size_t k = 3;
    size_t num_patches = 2;
    size_t *expanded = (size_t *)blt_arena_alloc(arena, num_patches * k * sizeof(size_t), 64);
    TEST_ASSERT(expanded != NULL);

    blt_patch_expand_group_ids(group_ids_in, num_patches, k, expanded);

    size_t expected[6] = {0, 0, 0, 1, 1, 1};
    for (size_t i = 0; i < num_patches * k; i++) {
        TEST_ASSERT(expanded[i] == expected[i]);
    }

    // --- reshape round-trip ---
    size_t E = 4;
    size_t patch_dim = E * k; // = 12
    size_t patch_shape[2] = {num_patches, patch_dim};
    blt_tensor p = blt_tensor_create(arena, patch_shape, 2, BLT_DTYPE_FP32);

    float *d = (float *)p.data;
    for (size_t i = 0; i < num_patches * patch_dim; i++) d[i] = (float)i;

    // view as [num_patches*k, E]
    blt_tensor split_view;
    blt_tensor_view_2d(&split_view, p.data, num_patches * k, E, p.backend);
    TEST_ASSERT(split_view.shape[0] == num_patches * k && split_view.shape[1] == E);

    // view back as [num_patches, patch_dim]
    blt_tensor concat_view;
    blt_tensor_view_2d(&concat_view, split_view.data, num_patches, patch_dim, split_view.backend);
    TEST_ASSERT(concat_view.shape[0] == num_patches && concat_view.shape[1] == patch_dim);

    // must be bit-identical to the original data, since its the same buffer
    const float *orig = (const float *)p.data;
    const float *roundtrip = (const float *)concat_view.data;
    for (size_t i = 0; i < num_patches * patch_dim; i++) {
        TEST_ASSERT(orig[i] == roundtrip[i]);
    }

    blt_arena_destroy(arena);
    return 1;
}

//----------------------------------------------------------------------------
// Block-diagonal group mask test
// only matching query/kv groups may attend to each other
static int run_mask_builder_group_test(void) {
    blt_arena *arena = blt_arena_create(1024 * 1024, BLT_BACKEND_CPU);
    if (!arena) {
        return 0;
    }

    size_t query_groups[4] = {0, 0, 1, 1};
    size_t kv_groups[4] = {0, 0, 1, 1};

    blt_mask_config config = {0};
    config.seq_len_q = 4;
    config.seq_len_kv = 4;
    config.is_causal = false;
    config.query_group_ids = query_groups;
    config.kv_group_ids = kv_groups;
    config.bidirectional_within_group = false;

    blt_tensor mask = {0};
    blt_build_attention_mask(&config, &mask, arena);

    // Different groups: always masked.
    TEST_ASSERT(is_allowed(&mask, 4, 0, 2) == 0);
    TEST_ASSERT(is_allowed(&mask, 4, 3, 1) == 0);
    // Same group, causal order respected (j <= i allowed).
    TEST_ASSERT(is_allowed(&mask, 4, 1, 0) == 1);
    // Same group, j > i should be masked since bidirectional_within_group is false.
    TEST_ASSERT(is_allowed(&mask, 4, 0, 1) == 0);

    blt_arena_destroy(arena);
    return 1;
}

int run_mask_builder_backend_tests(void) {
    int ok = 1;
    ok &= run_mask_builder_causal_test();
    ok &= run_mask_builder_sliding_window_test();
    ok &= run_mask_builder_doc_boundary_test();
    ok &= run_mask_builder_group_test();
    ok &= run_patch_k_split();
    return ok;
}