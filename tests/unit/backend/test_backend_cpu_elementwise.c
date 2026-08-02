#include <stdio.h>
#include "blt/core/allocator.h"
#include "blt/ops/elementwise.h"

int run_elementwise_backend_tests(void) {
    blt_arena* arena = blt_arena_create(4096, BLT_BACKEND_CPU);
    if (!arena) {
        fprintf(stderr, "[FAIL] arena creation for elementwise tests\n");
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
        b_data[i] = 0.5f * (float)(i + 1);
    }

    blt_add(&a, &b, &out);
    if (out_data[0] != 1.5f || out_data[3] != 6.0f) {
        fprintf(stderr, "[FAIL] elementwise add\n");
        blt_arena_destroy(arena);
        return 0;
    }

    blt_mul(&a, &b, &out);
    if (out_data[0] != 0.5f || out_data[3] != 8.0f) {
        fprintf(stderr, "[FAIL] elementwise mul\n");
        blt_arena_destroy(arena);
        return 0;
    }

    blt_arena_destroy(arena);
    return 1;
}
