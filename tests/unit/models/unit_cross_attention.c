#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>

#include "blt/core/allocator.h"
#include "blt/models/cross_attention.h"

#include "test_helpers.h"
#include "test_suite.h"


static void fill_identity(float* data, size_t dim) {
    for (size_t r = 0; r < dim; ++r) {
        for (size_t c = 0; c < dim; ++c) {
            data[r * dim + c] = (r == c) ? 1.0f : 0.0f;
        }
    }
}



int run_cross_attention_model_tests(void) {
    blt_arena* arena = blt_arena_create(65536, BLT_BACKEND_CPU);
    blt_arena* ca_arena = blt_arena_create(16384, BLT_BACKEND_CPU);

    if (!arena || !ca_arena) {
        fprintf(stderr, "[FAIL] arena creation for cross attention tests\n");
        return 0;
    }

    size_t query_shape[2] = {2, 4};   // num_patches=2, embed_dim=4
    size_t kv_shape[2] = {4, 4};      // seq_len=4, embed_dim=4
    size_t weight_shape[2] = {4, 4};
    size_t output_shape[2] = {2, 4};

    blt_tensor query_in = blt_tensor_create(arena, query_shape, 2, BLT_DTYPE_FP32);
    blt_tensor kv_in = blt_tensor_create(arena, kv_shape, 2, BLT_DTYPE_FP32);
    blt_tensor weight_q = blt_tensor_create(arena, weight_shape, 2, BLT_DTYPE_FP32);
    blt_tensor weight_k = blt_tensor_create(arena, weight_shape, 2, BLT_DTYPE_FP32);
    blt_tensor weight_v = blt_tensor_create(arena, weight_shape, 2, BLT_DTYPE_FP32);
    blt_tensor weight_proj = blt_tensor_create(arena, weight_shape, 2, BLT_DTYPE_FP32);
    blt_tensor output = blt_tensor_create(arena, output_shape, 2, BLT_DTYPE_FP32);
    blt_tensor expected = blt_tensor_create(arena, output_shape, 2, BLT_DTYPE_FP32);

    float* query_data = (float*)query_in.data;
    float* kv_data = (float*)kv_in.data;

    // Patch 0 query, patch 1 query
    query_data[0] = 1.0f; query_data[1] = 0.0f; query_data[2] = 1.0f; query_data[3] = 0.0f;
    query_data[4] = 0.0f; query_data[5] = 1.0f; query_data[6] = 0.0f; query_data[7] = 1.0f;

    // kv[0:2] = group 0 (belongs to patch 0), kv[2:4] = group 1 (belongs to patch 1)
    kv_data[0]  = 1.0f; kv_data[1]  = 0.0f; kv_data[2]  = 1.0f; kv_data[3]  = 0.0f;
    kv_data[4]  = 0.0f; kv_data[5]  = 1.0f; kv_data[6]  = 0.0f; kv_data[7]  = 1.0f;
    kv_data[8]  = 1.0f; kv_data[9]  = 0.0f; kv_data[10] = 1.0f; kv_data[11] = 0.0f;
    kv_data[12] = 0.0f; kv_data[13] = 1.0f; kv_data[14] = 0.0f; kv_data[15] = 1.0f;

    fill_identity((float*)weight_q.data, 4);
    fill_identity((float*)weight_k.data, 4);
    fill_identity((float*)weight_v.data, 4);
    fill_identity((float*)weight_proj.data, 4);

    // Block-diagonal patch mask
    // query i only attends to kv positions in its group
    size_t query_group_ids[2] = {0, 1};
    size_t kv_group_ids[4] = {0, 0, 1, 1};

    blt_mask_config mask_cfg = {0};
    mask_cfg.seq_len_q = 2;
    mask_cfg.seq_len_kv = 4;
    mask_cfg.sliding_window = 0;
    mask_cfg.doc_boundaries = NULL;
    mask_cfg.num_docs = 0;
    mask_cfg.query_group_ids = query_group_ids;
    mask_cfg.kv_group_ids = kv_group_ids;
    mask_cfg.bidirectional_within_group = true;
    mask_cfg.is_causal = false;

    blt_cross_attention_weights weights = {0};
    weights.weight_q = &weight_q;
    weights.weight_k = &weight_k;
    weights.weight_v = &weight_v;
    weights.weight_proj = &weight_proj;

    blt_cross_attention_config config = {0};
    config.embed_dim = 4;
    config.num_heads = 2;
    config.head_dim = 2;
    config.mask_config = &mask_cfg;

    blt_cross_attention_forward(&query_in, &kv_in, &weights, &output, &config, ca_arena);

    // each patch softmax reduces to a 2-way choice within own group
    float* expected_data = (float*)expected.data;
    const float expected_values[] = {
        0.6697615f, 0.3302385f, 0.6697615f, 0.3302385f,
        0.3302385f, 0.6697615f, 0.3302385f, 0.6697615f
    };
    for (size_t i = 0; i < 8; ++i) {
        expected_data[i] = expected_values[i];
    }

    TEST_ASSERT_CLOSE(&output, &expected, 1e-4f);

    blt_arena_destroy(arena);
    blt_arena_destroy(ca_arena);
    return 1;
}



// ----------------------------------------------------------------------------------
// Verifies the block-diagonal mask actually isolates patches: perturbing kv rows
// that belong to patch 1 group must not change patch 0 output row at all
// ----------------------------------------------------------------------------------

int run_cross_attention_masked_model_tests(void) {
    blt_arena* arena = blt_arena_create(65536, BLT_BACKEND_CPU);
    blt_arena* ca_arena = blt_arena_create(16384, BLT_BACKEND_CPU);

    if (!arena || !ca_arena) {
        fprintf(stderr, "[FAIL] arena creation for cross attention mask isolation tests\n");
        return 0;
    }

    size_t query_shape[2] = {2, 4};
    size_t kv_shape[2] = {4, 4};
    size_t weight_shape[2] = {4, 4};
    size_t output_shape[2] = {2, 4};

    blt_tensor query_in = blt_tensor_create(arena, query_shape, 2, BLT_DTYPE_FP32);
    blt_tensor kv_in_a = blt_tensor_create(arena, kv_shape, 2, BLT_DTYPE_FP32);
    blt_tensor kv_in_b = blt_tensor_create(arena, kv_shape, 2, BLT_DTYPE_FP32);
    blt_tensor weight_q = blt_tensor_create(arena, weight_shape, 2, BLT_DTYPE_FP32);
    blt_tensor weight_k = blt_tensor_create(arena, weight_shape, 2, BLT_DTYPE_FP32);
    blt_tensor weight_v = blt_tensor_create(arena, weight_shape, 2, BLT_DTYPE_FP32);
    blt_tensor weight_proj = blt_tensor_create(arena, weight_shape, 2, BLT_DTYPE_FP32);
    blt_tensor output_a = blt_tensor_create(arena, output_shape, 2, BLT_DTYPE_FP32);
    blt_tensor output_b = blt_tensor_create(arena, output_shape, 2, BLT_DTYPE_FP32);

    float* query_data = (float*)query_in.data;
    query_data[0] = 1.0f; query_data[1] = 0.0f; query_data[2] = 1.0f; query_data[3] = 0.0f;
    query_data[4] = 0.0f; query_data[5] = 1.0f; query_data[6] = 0.0f; query_data[7] = 1.0f;

    float* kv_a = (float*)kv_in_a.data;
    kv_a[0]  = 1.0f; kv_a[1]  = 0.0f; kv_a[2]  = 1.0f; kv_a[3]  = 0.0f;
    kv_a[4]  = 0.0f; kv_a[5]  = 1.0f; kv_a[6]  = 0.0f; kv_a[7]  = 1.0f;
    kv_a[8]  = 1.0f; kv_a[9]  = 0.0f; kv_a[10] = 1.0f; kv_a[11] = 0.0f;
    kv_a[12] = 0.0f; kv_a[13] = 1.0f; kv_a[14] = 0.0f; kv_a[15] = 1.0f;

    // kv_in_b is identical to kv_in_a in group 0 (rows 0-1), but wildly different
    // in group 1 (rows 2-3), which belongs only to patch 1
    float* kv_b = (float*)kv_in_b.data;
    memcpy(kv_b, kv_a, 16 * sizeof(float));
    kv_b[8]  = 500.0f; kv_b[9]  = -500.0f; kv_b[10] = 250.0f; kv_b[11] = -250.0f;
    kv_b[12] = -500.0f; kv_b[13] = 500.0f; kv_b[14] = -250.0f; kv_b[15] = 250.0f;

    fill_identity((float*)weight_q.data, 4);
    fill_identity((float*)weight_k.data, 4);
    fill_identity((float*)weight_v.data, 4);
    fill_identity((float*)weight_proj.data, 4);

    size_t query_group_ids[2] = {0, 1};
    size_t kv_group_ids[4] = {0, 0, 1, 1};

    blt_mask_config mask_cfg = {0};
    mask_cfg.seq_len_q = 2;
    mask_cfg.seq_len_kv = 4;
    mask_cfg.sliding_window = 0;
    mask_cfg.doc_boundaries = NULL;
    mask_cfg.num_docs = 0;
    mask_cfg.query_group_ids = query_group_ids;
    mask_cfg.kv_group_ids = kv_group_ids;
    mask_cfg.bidirectional_within_group = true;
    mask_cfg.is_causal = false;

    blt_cross_attention_weights weights = {0};
    weights.weight_q = &weight_q;
    weights.weight_k = &weight_k;
    weights.weight_v = &weight_v;
    weights.weight_proj = &weight_proj;

    blt_cross_attention_config config = {0};
    config.embed_dim = 4;
    config.num_heads = 2;
    config.head_dim = 2;
    config.mask_config = &mask_cfg;

    blt_cross_attention_forward(&query_in, &kv_in_a, &weights, &output_a, &config, ca_arena);
    blt_arena_reset(ca_arena);
    blt_cross_attention_forward(&query_in, &kv_in_b, &weights, &output_b, &config, ca_arena);

    // Patch 0 row (row 0) has to be identical between the runs
    blt_tensor patch0_a, patch0_b;
    blt_tensor_view_2d(&patch0_a, (float*)output_a.data, 1, 4, output_a.backend);
    blt_tensor_view_2d(&patch0_b, (float*)output_b.data, 1, 4, output_b.backend);

    TEST_ASSERT_CLOSE(&patch0_a, &patch0_b, 1e-4f);

    // Patch 1 row must be different
    float* out_a = (float*)output_a.data;
    float* out_b = (float*)output_b.data;
    bool patch1_differs = false;
    for (size_t d = 0; d < 4; ++d) {
        if (fabsf(out_a[4 + d] - out_b[4 + d]) > 1e-3f) {
            patch1_differs = true;
            break;
        }
    }
    TEST_ASSERT(patch1_differs);

    blt_arena_destroy(arena);
    blt_arena_destroy(ca_arena);
    return 1;
}



// num_patches = 1, seq_len = 1, embed_dim = 2, num_heads = 1, head_dim = 2,
// no causal masking (single-position mask trivially allows attention).
// With only one kv position per patch, softmax over a length-1 row is always 1,
// so the softmax backward term is 0 -- same simplification trick as the
// self-attention backward test.
//
//      query_in    = [1, 2]                          (1x2)
//      kv_in       = [3, 4]                           (1x2)
//      weight_q = weight_k = weight_v = weight_proj = I (2x2)  -> Q=query_in, K=V=kv_in
//      grad_out    = [1, 1]                           (1x2)
//
//      combined = W @ V = V = [3, 4]     (W = softmax(QK^T) = 1 for length-1 row)
//      grad_combined = grad_out @ proj^T = grad_out = [1, 1]
//      grad_weight_proj = combined^T @ grad_out = [[3,3],[4,4]]
//
//      grad_v = W^T @ grad_combined = grad_combined = [1, 1]
//      grad_scores = 0  (softmax backward is 0 for a length-1 row)
//      => grad_q = grad_k = [0, 0]
//
//      grad_query_in = grad_q @ weight_q^T = [0, 0]
//      grad_weight_q = query_in^T @ grad_q = [[0,0],[0,0]]
//      grad_weight_k = kv_in^T @ grad_k    = [[0,0],[0,0]]
//      grad_weight_v = kv_in^T @ grad_v    = [[3,3],[4,4]]
//      grad_kv_in = (grad_k @ weight_k^T) + (grad_v @ weight_v^T) = [0,0] + [1,1] = [1, 1]
 
int run_cross_attention_backward_model_tests(void) {
    blt_arena* arena = blt_arena_create(65536, BLT_BACKEND_CPU);
    blt_arena* ca_arena = blt_arena_create(16384, BLT_BACKEND_CPU);
 
    if (!arena || !ca_arena) {
        fprintf(stderr, "[FAIL] arena creation for cross attention backward tests\n");
        return 0;
    }
 
    size_t query_shape[2] = {1, 2};
    size_t kv_shape[2] = {1, 2};
    size_t weight_shape[2] = {2, 2};
    size_t grad_out_shape[2] = {1, 2};
 
    blt_tensor query_in = blt_tensor_create(arena, query_shape, 2, BLT_DTYPE_FP32);
    blt_tensor kv_in = blt_tensor_create(arena, kv_shape, 2, BLT_DTYPE_FP32);
    blt_tensor weight_q = blt_tensor_create(arena, weight_shape, 2, BLT_DTYPE_FP32);
    blt_tensor weight_k = blt_tensor_create(arena, weight_shape, 2, BLT_DTYPE_FP32);
    blt_tensor weight_v = blt_tensor_create(arena, weight_shape, 2, BLT_DTYPE_FP32);
    blt_tensor weight_proj = blt_tensor_create(arena, weight_shape, 2, BLT_DTYPE_FP32);
    blt_tensor grad_out = blt_tensor_create(arena, grad_out_shape, 2, BLT_DTYPE_FP32);
 
    blt_tensor grad_query_in = blt_tensor_create(arena, query_shape, 2, BLT_DTYPE_FP32);
    blt_tensor grad_kv_in = blt_tensor_create(arena, kv_shape, 2, BLT_DTYPE_FP32);
 
    float* query_data = (float*)query_in.data;
    query_data[0] = 1.0f; query_data[1] = 2.0f;
 
    float* kv_data = (float*)kv_in.data;
    kv_data[0] = 3.0f; kv_data[1] = 4.0f;
 
    float* wq = (float*)weight_q.data;
    float* wk = (float*)weight_k.data;
    float* wv = (float*)weight_v.data;
    float* wp = (float*)weight_proj.data;
    wq[0] = 1.0f; wq[1] = 0.0f; wq[2] = 0.0f; wq[3] = 1.0f;
    wk[0] = 1.0f; wk[1] = 0.0f; wk[2] = 0.0f; wk[3] = 1.0f;
    wv[0] = 1.0f; wv[1] = 0.0f; wv[2] = 0.0f; wv[3] = 1.0f;
    wp[0] = 1.0f; wp[1] = 0.0f; wp[2] = 0.0f; wp[3] = 1.0f;
 
    float* grad_out_data = (float*)grad_out.data;
    grad_out_data[0] = 1.0f; grad_out_data[1] = 1.0f;
 
    // Single query position, single kv position
    // mask allows attention
    size_t query_group_ids[1] = {0};
    size_t kv_group_ids[1] = {0};
 
    blt_mask_config mask_cfg = {0};
    mask_cfg.seq_len_q = 1;
    mask_cfg.seq_len_kv = 1;
    mask_cfg.sliding_window = 0;
    mask_cfg.doc_boundaries = NULL;
    mask_cfg.num_docs = 0;
    mask_cfg.query_group_ids = query_group_ids;
    mask_cfg.kv_group_ids = kv_group_ids;
    mask_cfg.bidirectional_within_group = true;
    mask_cfg.is_causal = false;
 
    blt_cross_attention_weights weights = {0};
    weights.weight_q = &weight_q;
    weights.weight_k = &weight_k;
    weights.weight_v = &weight_v;
    weights.weight_proj = &weight_proj;
 
    blt_cross_attention_config config = {0};
    config.embed_dim = 2;
    config.num_heads = 1;
    config.head_dim = 2;
    config.mask_config = &mask_cfg;
 
    blt_cross_attention_grad grad_weights = {0};
    grad_weights.grad_weight_q = blt_tensor_create(arena, weight_shape, 2, BLT_DTYPE_FP32);
    grad_weights.grad_weight_k = blt_tensor_create(arena, weight_shape, 2, BLT_DTYPE_FP32);
    grad_weights.grad_weight_v = blt_tensor_create(arena, weight_shape, 2, BLT_DTYPE_FP32);
    grad_weights.grad_weight_proj = blt_tensor_create(arena, weight_shape, 2, BLT_DTYPE_FP32);
 
    blt_cross_attention_backward(&query_in, &kv_in, &weights, &grad_out,
                                  &grad_query_in, &grad_kv_in, &grad_weights,
                                  &config, ca_arena);
 
    float* gqi = (float*)grad_query_in.data;
    TEST_ASSERT(fabsf(gqi[0] - 0.0f) < 1e-4f);
    TEST_ASSERT(fabsf(gqi[1] - 0.0f) < 1e-4f);
 
    float* gkv = (float*)grad_kv_in.data;
    TEST_ASSERT(fabsf(gkv[0] - 1.0f) < 1e-4f);
    TEST_ASSERT(fabsf(gkv[1] - 1.0f) < 1e-4f);
 
    float* gwq = (float*)grad_weights.grad_weight_q.data;
    for (size_t i = 0; i < 4; ++i) {
        TEST_ASSERT(fabsf(gwq[i] - 0.0f) < 1e-4f);
    }
 
    float* gwk = (float*)grad_weights.grad_weight_k.data;
    for (size_t i = 0; i < 4; ++i) {
        TEST_ASSERT(fabsf(gwk[i] - 0.0f) < 1e-4f);
    }
 
    float* gwv = (float*)grad_weights.grad_weight_v.data;
    const float expected_grad_wv[] = {3.0f, 3.0f, 4.0f, 4.0f};
    for (size_t i = 0; i < 4; ++i) {
        TEST_ASSERT(fabsf(gwv[i] - expected_grad_wv[i]) < 1e-4f);
    }
 
    float* gwp = (float*)grad_weights.grad_weight_proj.data;
    const float expected_grad_wp[] = {3.0f, 3.0f, 4.0f, 4.0f};
    for (size_t i = 0; i < 4; ++i) {
        TEST_ASSERT(fabsf(gwp[i] - expected_grad_wp[i]) < 1e-4f);
    }
 
    blt_arena_destroy(arena);
    blt_arena_destroy(ca_arena);
    return 1;
}