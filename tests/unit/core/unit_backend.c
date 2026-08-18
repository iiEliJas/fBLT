#include <stdio.h>
#include "blt/core/allocator.h"
#include "blt/core/backend.h"
#include "blt/ops/elementwise.h"

#include "test_helpers.h"
#include "test_suite.h"

int run_backend_core_tests(void) {
    blt_arena* arena = blt_arena_create(4096, BLT_BACKEND_CPU);
    if (!arena) {
        fprintf(stderr, "[FAIL] arena creation for backend dispatch\n");
        return 0;
    }

    size_t shape[1] = {4};
    blt_tensor a = blt_tensor_create(arena, shape, 1, BLT_DTYPE_FP32);
    blt_tensor b = blt_tensor_create(arena, shape, 1, BLT_DTYPE_FP32);
    blt_tensor out = blt_tensor_create(arena, shape, 1, BLT_DTYPE_FP32);
    blt_tensor expected_add = blt_tensor_create(arena, shape, 1, BLT_DTYPE_FP32);
    blt_tensor expected_mul = blt_tensor_create(arena, shape, 1, BLT_DTYPE_FP32);

    float* a_data = (float*)a.data;
    float* b_data = (float*)b.data;
    float* expected_add_data = (float*)expected_add.data;
    float* expected_mul_data = (float*)expected_mul.data;
    for (size_t i = 0; i < 4; ++i) {
        a_data[i] = (float)(i + 1);
        b_data[i] = 2.0f;
    }

    blt_add(&a, &b, &out);
    expected_add_data[0] = 3.0f; expected_add_data[1] = 4.0f; expected_add_data[2] = 5.0f; expected_add_data[3] = 6.0f;
    TEST_ASSERT_CLOSE(&out, &expected_add, 1e-6f);

    blt_mul(&a, &b, &out);
    expected_mul_data[0] = 2.0f; expected_mul_data[1] = 4.0f; expected_mul_data[2] = 6.0f; expected_mul_data[3] = 8.0f;
    TEST_ASSERT_CLOSE(&out, &expected_mul, 1e-6f);

    blt_arena_destroy(arena);
    return 1;
}
