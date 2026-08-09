#include <stdio.h>
#include <math.h>

#include "blt/core/allocator.h"
#include "blt/ops/softmax.h"
#include "blt/ops/rope.h"
#include "blt/ops/cross_entropy.h"

#include "test_helpers.h"
#include "test_suite.h"


int run_softmax_backend_tests(void) {
    blt_arena* arena = blt_arena_create(4096, BLT_BACKEND_CPU);
    if (!arena) {
        fprintf(stderr, "[FAIL] arena creation for softmax tests\n");
        return 0;
    }

    // ---------------------------------------------------------------
    // Softmax
    // input = [[1, 2, 3], [1, 1, 1]]
    // check results sum to 1 and that the first row is increasing
    {
        size_t shape[2] = {2, 3};
        blt_tensor input = blt_tensor_create(arena, shape, 2, BLT_DTYPE_FP32);
        blt_tensor output = blt_tensor_create(arena, shape, 2, BLT_DTYPE_FP32);

        float* in_data = (float*)input.data;
        float* out_data = (float*)output.data;
        in_data[0] = 1.0f; in_data[1] = 2.0f; in_data[2] = 3.0f;
        in_data[3] = 1.0f; in_data[4] = 1.0f; in_data[5] = 1.0f;

        blt_softmax(&input, &output);

        TEST_ASSERT(out_data[0] + out_data[1] + out_data[2] >= 0.999f && out_data[0] + out_data[1] + out_data[2] <= 1.001f);
        TEST_ASSERT(out_data[2] > out_data[1] && out_data[1] > out_data[0]);
    }


    // ---------------------------------------------------------------
    // Softmax backward
    // y = [0.5, 0.5], dy = [1, 0]
    // dot = dy.y = 0.5
    // dx0 = y0*(dy0-dot) = 0.5*(1-0.5) = 0.25
    // dx1 = y1*(dy1-dot) = 0.5*(0-0.5) = -0.25
    {
        size_t shape[1] = {2};
        blt_tensor softmax_out = blt_tensor_create(arena, shape, 1, BLT_DTYPE_FP32);
        blt_tensor grad_out = blt_tensor_create(arena, shape, 1, BLT_DTYPE_FP32);
        blt_tensor grad_in = blt_tensor_create(arena, shape, 1, BLT_DTYPE_FP32);

        float* yd = (float*)softmax_out.data;
        yd[0] = 0.5f; yd[1] = 0.5f;

        float* god = (float*)grad_out.data;
        god[0] = 1.0f; god[1] = 0.0f;

        blt_softmax_backward(&grad_out, &softmax_out, &grad_in);

        float* gid = (float*)grad_in.data;
        TEST_ASSERT(fabsf(gid[0] - 0.25f) < 1e-5f);
        TEST_ASSERT(fabsf(gid[1] - (-0.25f)) < 1e-5f);
    }


    blt_arena_destroy(arena);
    return 1;
}



int run_softmax_test(void) {
    blt_arena* arena = blt_arena_create(1024 * 1024, BLT_BACKEND_CPU);
    blt_tensor input = {0};
    blt_tensor expected = {0};
    blt_tensor output = {0};

    int ok = 0;
    if (!load_binary_tensor("data/golden_softmax_in.bin", arena, &input) ||
        !load_binary_tensor("data/golden_softmax_out.bin", arena, &expected)) {
        blt_arena_destroy(arena);
        return ok;
    }

    output = blt_tensor_create(arena, input.shape, input.ndim, BLT_DTYPE_FP32);
    blt_softmax(&input, &output);
    ok = check_close(&output, &expected, 1e-4f);

    blt_arena_destroy(arena);
    return ok;
}



int run_cross_entropy_tests(void) {
    blt_arena* arena = blt_arena_create(4096, BLT_BACKEND_CPU);
    if (!arena) {
        fprintf(stderr, "[FAIL] arena creation for cross entropy tests\n");
        return 0;
    }
 
    size_t logits_shape[2] = {2, 2};
    size_t targets_shape[1] = {2};
    size_t loss_shape[1] = {1};
 
    blt_tensor logits = blt_tensor_create(arena, logits_shape, 2, BLT_DTYPE_FP32);
    blt_tensor targets = blt_tensor_create(arena, targets_shape, 1, BLT_DTYPE_UINT8);
    blt_tensor loss = blt_tensor_create(arena, loss_shape, 1, BLT_DTYPE_FP32);
    blt_tensor grad_logits = blt_tensor_create(arena, logits_shape, 2, BLT_DTYPE_FP32);
    
    // seq_len=2, vocab_size=2. logits=[[0,0],[0,0]], targets=[0,1]
    //      softmax([0,0]) = [0.5,0.5]
    //      loss = mean(-log(0.5), -log(0.5)) = -log(0.5) so approx. 0.693147
    //      grad = (softmax - one_hot) / seq_len
    //      row0: ([0.5,0.5] - [1,0]) / 2 = [-0.25, 0.25]
    //      row1: ([0.5,0.5] - [0,1]) / 2 = [0.25, -0.25]

    float* logits_data = (float*)logits.data;
    logits_data[0] = 0.0f; logits_data[1] = 0.0f;
    logits_data[2] = 0.0f; logits_data[3] = 0.0f;
 
    uint8_t* targets_data = (uint8_t*)targets.data;
    targets_data[0] = 0;
    targets_data[1] = 1;
 
    blt_cross_entropy_forward(&logits, &targets, &loss);
 
    float* loss_data = (float*)loss.data;
    TEST_ASSERT(fabsf(loss_data[0] - 0.693147f) < 1e-4f);
 
    blt_cross_entropy_backward(&logits, &targets, &grad_logits);
 
    float* grad_data = (float*)grad_logits.data;
    TEST_ASSERT(fabsf(grad_data[0] - (-0.25f)) < 1e-4f);
    TEST_ASSERT(fabsf(grad_data[1] - 0.25f) < 1e-4f);
    TEST_ASSERT(fabsf(grad_data[2] - 0.25f) < 1e-4f);
    TEST_ASSERT(fabsf(grad_data[3] - (-0.25f)) < 1e-4f);
 
    blt_arena_destroy(arena);
    return 1;
}



int run_rope_backend_test(void) {
    blt_arena* arena = blt_arena_create(4096, BLT_BACKEND_CPU);
    if (!arena) {
        fprintf(stderr, "[FAIL] arena creation for rope tests\n");
        return 0;
    }

    // ---------------------------------------------------------------
    // RoPE backward
    // seq_len=1, num_heads=1, head_dim=2, half=1, cos=0, sin=1
    // grad_out = [1, 0]
    // giv0 = g0*c + g1*s = 1*0 + 0*1 = 0
    // giv1 = g1*c - g0*s = 0*0 - 1*1 = -1
    {
        size_t seq_shape[3] = {1, 1, 2};
        size_t cs_shape[2] = {1, 1};
        blt_tensor grad_out = blt_tensor_create(arena, seq_shape, 3, BLT_DTYPE_FP32);
        blt_tensor cos_t = blt_tensor_create(arena, cs_shape, 2, BLT_DTYPE_FP32);
        blt_tensor sin_t = blt_tensor_create(arena, cs_shape, 2, BLT_DTYPE_FP32);
        blt_tensor grad_in = blt_tensor_create(arena, seq_shape, 3, BLT_DTYPE_FP32);
 
        float* god = (float*)grad_out.data;
        god[0] = 1.0f; god[1] = 0.0f;
 
        ((float*)cos_t.data)[0] = 0.0f;
        ((float*)sin_t.data)[0] = 1.0f;
 
        blt_rope_apply_backward(&grad_out, &cos_t, &sin_t, &grad_in);
 
        float* gid = (float*)grad_in.data;
        TEST_ASSERT(fabsf(gid[0] - 0.0f) < 1e-5f);
        TEST_ASSERT(fabsf(gid[1] - (-1.0f)) < 1e-5f);
    }

    blt_arena_destroy(arena);
    return 1;
}