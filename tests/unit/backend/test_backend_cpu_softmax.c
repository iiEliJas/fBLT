#include <stdio.h>
#include "blt/core/allocator.h"
#include "blt/ops/softmax.h"

#include "test_helpers.h"
#include "test_suite.h"


int run_softmax_backend_tests(void) {
    blt_arena* arena = blt_arena_create(4096, BLT_BACKEND_CPU);
    if (!arena) {
        fprintf(stderr, "[FAIL] arena creation for softmax tests\n");
        return 0;
    }

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
