#include <stdio.h>
#include "blt/core/allocator.h"
#include "blt/ops/matmul.h"

#include "test_helpers.h"
#include "test_suite.h"

int run_matmul_backend_tests(void) {
    blt_arena* arena = blt_arena_create(4096, BLT_BACKEND_CPU);
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

    float* a_data = (float*)a.data;
    float* b_data = (float*)b.data;
    float* expected_data = (float*)expected.data;

    a_data[0] = 1.0f; a_data[1] = 2.0f;
    a_data[2] = 3.0f; a_data[3] = 4.0f;
    b_data[0] = 5.0f; b_data[1] = 6.0f;
    b_data[2] = 7.0f; b_data[3] = 8.0f;

    blt_matmul(&a, &b, &out);

    expected_data[0] = 19.0f; expected_data[1] = 22.0f;
    expected_data[2] = 43.0f; expected_data[3] = 50.0f;
    TEST_ASSERT_CLOSE(&out, &expected, 1e-6f);

    blt_arena_destroy(arena);
    return 1;
}
