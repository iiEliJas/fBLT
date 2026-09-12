#include <stdio.h>
#include <math.h>

#include "blt/core/allocator.h"
#include "blt/ops/matmul.h"

#include "test_helpers.h"
#include "test_suite.h"

// ---------------------------------------------------------------
// Matmul
//      a = [[1,2],[3,4]], b = [[5,6],[7,8]], out = [[19,22],[43,50]]

int run_matmul_backend_tests(void) {
    blt_arena *arena = blt_arena_create(4096, BLT_BACKEND_CPU);
    if (!arena) {
        fprintf(stderr, "[FAIL] arena creation for matmul tests\n");
        return 0;
    }

    size_t a_shape[2] = {2, 2};
    size_t b_shape[2] = {2, 2};
    size_t out_shape[2] = {2, 2};
    blt_tensor a = blt_tensor_create(arena, a_shape, 2, BLT_DTYPE_FP32);
    blt_tensor b = blt_tensor_create(arena, b_shape, 2, BLT_DTYPE_FP32);
    blt_tensor out = blt_tensor_create(arena, out_shape, 2, BLT_DTYPE_FP32);
    blt_tensor expected = blt_tensor_create(arena, out_shape, 2, BLT_DTYPE_FP32);

    float *a_data = (float *)a.data;
    float *b_data = (float *)b.data;
    float *expected_data = (float *)expected.data;

    a_data[0] = 1.0f;
    a_data[1] = 2.0f;
    a_data[2] = 3.0f;
    a_data[3] = 4.0f;
    b_data[0] = 5.0f;
    b_data[1] = 6.0f;
    b_data[2] = 7.0f;
    b_data[3] = 8.0f;

    blt_matmul(&a, &b, &out);

    expected_data[0] = 19.0f;
    expected_data[1] = 22.0f;
    expected_data[2] = 43.0f;
    expected_data[3] = 50.0f;
    TEST_ASSERT_CLOSE(&out, &expected, 1e-6f);

    blt_arena_destroy(arena);
    return 1;
}

// ---------------------------------------------------------------
// Matmul backward
//      a = [[1,2],[3,4]], b = I (2x2), grad_out = [[1,1],[1,1]]
//      grad_a = grad_out @ b^T = grad_out @ I = grad_out = [[1,1],[1,1]]
//      grad_b = a^T @ grad_out = [[1,3],[2,4]] @ [[1,1],[1,1]] = [[4,4],[6,6]]

int run_matmul_backward_backend_tests(void) {
    blt_arena *arena = blt_arena_create(4096, BLT_BACKEND_CPU);
    if (!arena) {
        fprintf(stderr, "[FAIL] arena creation for matmul tests\n");
        return 0;
    }

    size_t shape[2] = {2, 2};
    blt_tensor a = blt_tensor_create(arena, shape, 2, BLT_DTYPE_FP32);
    blt_tensor b = blt_tensor_create(arena, shape, 2, BLT_DTYPE_FP32);
    blt_tensor grad_out = blt_tensor_create(arena, shape, 2, BLT_DTYPE_FP32);
    blt_tensor grad_a = blt_tensor_create(arena, shape, 2, BLT_DTYPE_FP32);
    blt_tensor grad_b = blt_tensor_create(arena, shape, 2, BLT_DTYPE_FP32);

    float *ad = (float *)a.data;
    ad[0] = 1.0f;
    ad[1] = 2.0f;
    ad[2] = 3.0f;
    ad[3] = 4.0f;

    float *bd = (float *)b.data;
    bd[0] = 1.0f;
    bd[1] = 0.0f;
    bd[2] = 0.0f;
    bd[3] = 1.0f;

    float *god = (float *)grad_out.data;
    god[0] = 1.0f;
    god[1] = 1.0f;
    god[2] = 1.0f;
    god[3] = 1.0f;

    blt_matmul_backward(&a, &b, &grad_out, &grad_a, &grad_b);

    float *gad = (float *)grad_a.data;
    TEST_ASSERT(fabsf(gad[0] - 1.0f) < 1e-5f);
    TEST_ASSERT(fabsf(gad[1] - 1.0f) < 1e-5f);
    TEST_ASSERT(fabsf(gad[2] - 1.0f) < 1e-5f);
    TEST_ASSERT(fabsf(gad[3] - 1.0f) < 1e-5f);

    float *gbd = (float *)grad_b.data;
    TEST_ASSERT(fabsf(gbd[0] - 4.0f) < 1e-5f);
    TEST_ASSERT(fabsf(gbd[1] - 4.0f) < 1e-5f);
    TEST_ASSERT(fabsf(gbd[2] - 6.0f) < 1e-5f);
    TEST_ASSERT(fabsf(gbd[3] - 6.0f) < 1e-5f);

    blt_arena_destroy(arena);
    return 1;
}
