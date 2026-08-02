#include <stdio.h>
#include "blt/core/allocator.h"
#include "blt/core/backend.h"
#include "blt/ops/elementwise.h"

static int check_close(float actual, float expected, float tol) {
    return actual >= expected - tol && actual <= expected + tol;
}

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

    float* a_data = (float*)a.data;
    float* b_data = (float*)b.data;
    float* out_data = (float*)out.data;
    for (size_t i = 0; i < 4; ++i) {
        a_data[i] = (float)(i + 1);
        b_data[i] = 2.0f;
    }

    blt_add(&a, &b, &out);
    if (!check_close(out_data[0], 3.0f, 1e-6f) || !check_close(out_data[3], 6.0f, 1e-6f)) {
        fprintf(stderr, "[FAIL] backend add dispatch\n");
        blt_arena_destroy(arena);
        return 0;
    }

    blt_mul(&a, &b, &out);
    if (!check_close(out_data[0], 2.0f, 1e-6f) || !check_close(out_data[3], 8.0f, 1e-6f)) {
        fprintf(stderr, "[FAIL] backend mul dispatch\n");
        blt_arena_destroy(arena);
        return 0;
    }

    blt_arena_destroy(arena);
    return 1;
}
