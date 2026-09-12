#include <math.h>
#include <stdio.h>

#include "blt/core/allocator.h"
#include "blt/models/attention.h"

#include "test_helpers.h"
#include "test_suite.h"

int run_attention_model_tests(void) {
    blt_arena *arena = blt_arena_create(65536, BLT_BACKEND_CPU);
    blt_arena *attention_arena = blt_arena_create(4096, BLT_BACKEND_CPU);

    if (!arena || !attention_arena) {
        fprintf(stderr, "[FAIL] arena creation for attention tests\n");
        return 0;
    }

    size_t input_shape[2] = {2, 4};
    size_t qkv_shape[2] = {4, 12};
    size_t proj_shape[2] = {4, 4};
    size_t output_shape[2] = {2, 4};

    blt_tensor input = blt_tensor_create(arena, input_shape, 2, BLT_DTYPE_FP32);
    blt_tensor weight_qkv = blt_tensor_create(arena, qkv_shape, 2, BLT_DTYPE_FP32);
    blt_tensor weight_proj = blt_tensor_create(arena, proj_shape, 2, BLT_DTYPE_FP32);
    blt_tensor output = blt_tensor_create(arena, output_shape, 2, BLT_DTYPE_FP32);
    blt_tensor expected = blt_tensor_create(arena, output_shape, 2, BLT_DTYPE_FP32);

    float *input_data = (float *)input.data;
    float *qkv_data = (float *)weight_qkv.data;
    float *proj_data = (float *)weight_proj.data;

    input_data[0] = 1.0f;
    input_data[1] = 0.0f;
    input_data[2] = 1.0f;
    input_data[3] = 0.0f;
    input_data[4] = 0.0f;
    input_data[5] = 1.0f;
    input_data[6] = 0.0f;
    input_data[7] = 1.0f;

    for (size_t row = 0; row < 4; ++row) {
        for (size_t col = 0; col < 12; ++col) {
            qkv_data[row * 12 + col] = 0.0f;
        }
    }

    for (size_t i = 0; i < 4; ++i) {
        qkv_data[i * 12 + i] = 1.0f;
        qkv_data[i * 12 + 4 + i] = 1.0f;
        qkv_data[i * 12 + 8 + i] = 1.0f;
    }

    for (size_t i = 0; i < 4; ++i) {
        proj_data[i * 4 + i] = 1.0f;
    }

    blt_attention_config config = {0};
    config.embed_dim = 4;
    config.num_heads = 2;
    config.head_dim = 2;
    config.is_causal = false;
    config.use_rope = false;

    blt_multihead_attention(&input, &weight_qkv, &weight_proj, &output, &config, attention_arena);

    remove("build/rope_cache.bin");
    blt_attention_config rope_config = config;
    rope_config.use_rope = true;
    rope_config.rope_theta = 10000.0f;

    blt_tensor rope_output = blt_tensor_create(arena, output_shape, 2, BLT_DTYPE_FP32);
    blt_tensor rope_output_repeat = blt_tensor_create(arena, output_shape, 2, BLT_DTYPE_FP32);

    blt_multihead_attention(&input, &weight_qkv, &weight_proj, &rope_output, &rope_config, attention_arena);
    blt_multihead_attention(&input, &weight_qkv, &weight_proj, &rope_output_repeat, &rope_config, attention_arena);

    TEST_ASSERT_CLOSE(&rope_output, &rope_output_repeat, 1e-4f);

    float *expected_data = (float *)expected.data;
    const float expected_values[] = {0.6697615f, 0.3302385f, 0.6697615f, 0.3302385f,
                                     0.3302385f, 0.6697615f, 0.3302385f, 0.6697615f};
    for (size_t i = 0; i < 8; ++i) {
        expected_data[i] = expected_values[i];
    }

    TEST_ASSERT_CLOSE(&output, &expected, 1e-4f);

    blt_arena_destroy(arena);
    blt_arena_destroy(attention_arena);
    return 1;
}

int run_attention_masked_model_tests(void) {
    blt_arena *arena = blt_arena_create(65536, BLT_BACKEND_CPU);
    blt_arena *attention_arena = blt_arena_create(4096, BLT_BACKEND_CPU);

    if (!arena || !attention_arena) {
        fprintf(stderr, "[FAIL] arena creation for masked attention tests\n");
        return 0;
    }

    size_t input_shape[2] = {2, 4};
    size_t qkv_shape[2] = {4, 12};
    size_t proj_shape[2] = {4, 4};
    size_t output_shape[2] = {2, 4};

    blt_tensor input = blt_tensor_create(arena, input_shape, 2, BLT_DTYPE_FP32);
    blt_tensor weight_qkv = blt_tensor_create(arena, qkv_shape, 2, BLT_DTYPE_FP32);
    blt_tensor weight_proj = blt_tensor_create(arena, proj_shape, 2, BLT_DTYPE_FP32);
    blt_tensor masked_output = blt_tensor_create(arena, output_shape, 2, BLT_DTYPE_FP32);
    blt_tensor causal_output = blt_tensor_create(arena, output_shape, 2, BLT_DTYPE_FP32);
    blt_tensor expected = blt_tensor_create(arena, output_shape, 2, BLT_DTYPE_FP32);

    float *input_data = (float *)input.data;
    float *qkv_data = (float *)weight_qkv.data;
    float *proj_data = (float *)weight_proj.data;

    // Same setup as run_attention_model_tests: W_qkv and W_proj are identity matrices.
    input_data[0] = 1.0f;
    input_data[1] = 0.0f;
    input_data[2] = 1.0f;
    input_data[3] = 0.0f;
    input_data[4] = 0.0f;
    input_data[5] = 1.0f;
    input_data[6] = 0.0f;
    input_data[7] = 1.0f;

    for (size_t row = 0; row < 4; ++row) {
        for (size_t col = 0; col < 12; ++col) {
            qkv_data[row * 12 + col] = 0.0f;
        }
    }
    for (size_t i = 0; i < 4; ++i) {
        qkv_data[i * 12 + i] = 1.0f;
        qkv_data[i * 12 + 4 + i] = 1.0f;
        qkv_data[i * 12 + 8 + i] = 1.0f;
    }
    for (size_t i = 0; i < 4; ++i) {
        proj_data[i * 4 + i] = 1.0f;
    }

    // masked config for causal masking
    blt_mask_config mask_cfg = {0};
    mask_cfg.seq_len_q = 2;
    mask_cfg.seq_len_kv = 2;
    mask_cfg.sliding_window = 0;
    mask_cfg.doc_boundaries = NULL;
    mask_cfg.num_docs = 0;
    mask_cfg.query_group_ids = NULL;
    mask_cfg.kv_group_ids = NULL;
    mask_cfg.bidirectional_within_group = false;
    mask_cfg.is_causal = true;

    blt_attention_config masked_config = {0};
    masked_config.embed_dim = 4;
    masked_config.num_heads = 2;
    masked_config.head_dim = 2;
    masked_config.is_causal = false; // false to use the masked config
    masked_config.use_rope = false;
    masked_config.mask_config = &mask_cfg;

    blt_multihead_attention(&input, &weight_qkv, &weight_proj, &masked_output, &masked_config, attention_arena);

    float *expected_data = (float *)expected.data;
    const float expected_values[] = {1.0f, 0.0f, 1.0f, 0.0f, 0.3302385f, 0.6697615f, 0.3302385f, 0.6697615f};
    for (size_t i = 0; i < 8; ++i) {
        expected_data[i] = expected_values[i];
    }
    TEST_ASSERT_CLOSE(&masked_output, &expected, 1e-4f);

    // Parity check
    blt_attention_config legacy_causal_config = {0};
    legacy_causal_config.embed_dim = 4;
    legacy_causal_config.num_heads = 2;
    legacy_causal_config.head_dim = 2;
    legacy_causal_config.is_causal = true;
    legacy_causal_config.use_rope = false;
    legacy_causal_config.mask_config = NULL;

    blt_multihead_attention(&input, &weight_qkv, &weight_proj, &causal_output, &legacy_causal_config, attention_arena);

    TEST_ASSERT_CLOSE(&masked_output, &causal_output, 1e-6f);

    blt_arena_destroy(arena);
    blt_arena_destroy(attention_arena);
    return 1;
}
