#include <math.h>
#include <stdio.h>

#include "blt/core/allocator.h"
#include "blt/models/attention.h"

#include "test_helpers.h"
#include "test_suite.h"

// seq_len = 1, embed_dim = 2, num_heads = 1, head_dim = 2,
// no RoPE, non-causal
// sofmax returns always 1 so the backward pass is simplified:
//      input       = [1, 2]
//      weight_qkv  = [[1,0, 1,0, 1,0],
//                      [0,1, 0,1, 0,1]]              (2x6) -> Q=K=V=input
//      weight_proj = I (2x2)
//      grad_out    = [1, 1]                         (1x2)
//
//      combined = W @ V = V = [1, 2]  (because: W = softmax(QK^T) = 1 for length-1 row)
//      grad_combined = grad_out @ proj^T = grad_out = [1, 1]
//      grad_weight_proj = combined^T @ grad_out = [[1,1],[2,2]]
//
//      grad_v = W^T @ grad_combined = grad_combined = [1, 1]
//      grad_scores = 0  (softmax backward is 0 for a length 1 row)
//      => grad_q = grad_k = [0, 0]
//      grad_qkv = [grad_q | grad_k | grad_v] = [0,0, 0,0, 1,1]
//
//      grad_input = grad_qkv @ weight_qkv^T = [1, 1]
//      grad_weight_qkv = input^T @ grad_qkv = [[0,0,0,0,1,1],
//                                           [0,0,0,0,2,2]]

int run_attention_backward_model_tests(void) {
    blt_arena* arena = blt_arena_create(65536, BLT_BACKEND_CPU);
    blt_arena* attention_arena = blt_arena_create(8192, BLT_BACKEND_CPU);

    if (!arena || !attention_arena) {
        fprintf(stderr, "[FAIL] arena creation for attention backward tests\n");
        return 0;
    }

    size_t input_shape[2] = {1, 2};
    size_t qkv_shape[2] = {2, 6};
    size_t proj_shape[2] = {2, 2};
    size_t out_shape[2] = {1, 2};

    blt_tensor input = blt_tensor_create(arena, input_shape, 2, BLT_DTYPE_FP32);
    blt_tensor weight_qkv = blt_tensor_create(arena, qkv_shape, 2, BLT_DTYPE_FP32);
    blt_tensor weight_proj = blt_tensor_create(arena, proj_shape, 2, BLT_DTYPE_FP32);
    blt_tensor grad_out = blt_tensor_create(arena, out_shape, 2, BLT_DTYPE_FP32);

    blt_tensor grad_input = blt_tensor_create(arena, input_shape, 2, BLT_DTYPE_FP32);
    blt_tensor grad_weight_qkv = blt_tensor_create(arena, qkv_shape, 2, BLT_DTYPE_FP32);
    blt_tensor grad_weight_proj = blt_tensor_create(arena, proj_shape, 2, BLT_DTYPE_FP32);

    float* input_data = (float*)input.data;
    input_data[0] = 1.0f; input_data[1] = 2.0f;

    float* qkv_data = (float*)weight_qkv.data;
    for (size_t row = 0; row < 2; ++row) {
        for (size_t col = 0; col < 6; ++col) {
            qkv_data[row * 6 + col] = 0.0f;
        }
    }
    for (size_t i = 0; i < 2; ++i) {
        qkv_data[i * 6 + i] = 1.0f;       // Q
        qkv_data[i * 6 + 2 + i] = 1.0f;   // K
        qkv_data[i * 6 + 4 + i] = 1.0f;   // V
    }

    float* proj_data = (float*)weight_proj.data;
    proj_data[0] = 1.0f; proj_data[1] = 0.0f;
    proj_data[2] = 0.0f; proj_data[3] = 1.0f;

    float* grad_out_data = (float*)grad_out.data;
    grad_out_data[0] = 1.0f; grad_out_data[1] = 1.0f;

    blt_attention_config config = {0};
    config.embed_dim = 2;
    config.num_heads = 1;
    config.head_dim = 2;
    config.is_causal = false;
    config.use_rope = false;

    blt_multihead_attention_backward(&input, &weight_qkv, &weight_proj, &grad_out,
                                      &grad_input, &grad_weight_qkv, &grad_weight_proj,
                                      &config, attention_arena);

    float* gi = (float*)grad_input.data;
    TEST_ASSERT(fabsf(gi[0] - 1.0f) < 1e-4f);
    TEST_ASSERT(fabsf(gi[1] - 1.0f) < 1e-4f);

    float* gp = (float*)grad_weight_proj.data;
    const float expected_grad_proj[] = {1.0f, 1.0f, 2.0f, 2.0f};
    for (size_t i = 0; i < 4; ++i) {
        TEST_ASSERT(fabsf(gp[i] - expected_grad_proj[i]) < 1e-4f);
    }

    float* gq = (float*)grad_weight_qkv.data;
    const float expected_grad_qkv[] = {
        0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 2.0f, 2.0f
    };
    for (size_t i = 0; i < 12; ++i) {
        TEST_ASSERT(fabsf(gq[i] - expected_grad_qkv[i]) < 1e-4f);
    }

    blt_arena_destroy(arena);
    blt_arena_destroy(attention_arena);
    return 1;
}