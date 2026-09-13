#include <stdio.h>
#include <math.h>

#include "core/allocator.h"
#include "ops/elementwise.h"
#include "ops/gelu.h"

#include "test_helpers.h"
#include "test_suite.h"

int run_elementwise_backend_tests(void) {
    blt_arena *arena = blt_arena_create(4096, BLT_BACKEND_CPU);
    if (!arena) {
        fprintf(stderr, "[FAIL] arena creation for elementwise tests\n");
        return 0;
    }

    size_t shape[1] = {4};
    blt_tensor a = blt_tensor_create(arena, shape, 1, BLT_DTYPE_FP32);
    blt_tensor b = blt_tensor_create(arena, shape, 1, BLT_DTYPE_FP32);
    blt_tensor out = blt_tensor_create(arena, shape, 1, BLT_DTYPE_FP32);
    blt_tensor expected = blt_tensor_create(arena, shape, 1, BLT_DTYPE_FP32);

    float *a_data = (float *)a.data;
    float *b_data = (float *)b.data;
    float *expected_data = (float *)expected.data;
    for (size_t i = 0; i < 4; ++i) {
        a_data[i] = (float)(i + 1);
        b_data[i] = 0.5f * (float)(i + 1);
    }

    blt_add(&a, &b, &out);
    expected_data[0] = 1.5f;
    expected_data[1] = 3.0f;
    expected_data[2] = 4.5f;
    expected_data[3] = 6.0f;
    TEST_ASSERT_CLOSE(&out, &expected, 1e-6f);

    blt_mul(&a, &b, &out);
    expected_data[0] = 0.5f;
    expected_data[1] = 2.0f;
    expected_data[2] = 4.5f;
    expected_data[3] = 8.0f;
    TEST_ASSERT_CLOSE(&out, &expected, 1e-6f);

    blt_arena_destroy(arena);
    return 1;
}

int run_gelu_backend_tests(void) {
    blt_arena *arena = blt_arena_create(4096, BLT_BACKEND_CPU);
    if (!arena) {
        fprintf(stderr, "[FAIL] arena creation for gelu tests\n");
        return 0;
    }

    size_t shape[1] = {4};
    blt_tensor a = blt_tensor_create(arena, shape, 1, BLT_DTYPE_FP32);
    blt_tensor out = blt_tensor_create(arena, shape, 1, BLT_DTYPE_FP32);
    blt_tensor expected = blt_tensor_create(arena, shape, 1, BLT_DTYPE_FP32);

    float *a_data = (float *)a.data;
    float *expected_data = (float *)expected.data;
    for (size_t i = 0; i < 4; ++i) {
        a_data[i] = (float)(i + 1);
        float v = a_data[i];
        float v3 = v * v * v;
        float inner = 0.7978845608f * (v + 0.044715f * v3);
        expected_data[i] = 0.5f * v * (1.0f + tanhf(inner));
    }

    blt_gelu_forward(&a, &out);
    TEST_ASSERT_CLOSE(&out, &expected, 1e-6f);

    //-------------------------------------
    // Test backward pass of GELU

    size_t shape2[1] = {2};
    blt_tensor x = blt_tensor_create(arena, shape2, 1, BLT_DTYPE_FP32);
    blt_tensor grad_out = blt_tensor_create(arena, shape2, 1, BLT_DTYPE_FP32);
    blt_tensor grad_x = blt_tensor_create(arena, shape2, 1, BLT_DTYPE_FP32);

    float *xd = (float *)x.data;
    float *god = (float *)grad_out.data;
    xd[0] = 0.0f;
    xd[1] = 0.0f;
    god[0] = 1.0f;
    god[1] = 1.0f;

    blt_gelu_backward(&grad_out, &x, &grad_x);

    float *gxd = (float *)grad_x.data;
    TEST_ASSERT(fabsf(gxd[0] - 0.5f) < 1e-4f);
    TEST_ASSERT(fabsf(gxd[1] - 0.5f) < 1e-4f);

    blt_arena_destroy(arena);
    return 1;
}
