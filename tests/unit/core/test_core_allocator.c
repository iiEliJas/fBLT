#include <stdio.h>
#include "blt/core/allocator.h"

int run_allocator_core_tests(void) {
    blt_arena* arena = blt_arena_create(4096, BLT_BACKEND_CPU);
    if (!arena) {
        fprintf(stderr, "[FAIL] arena creation\n");
        return 0;
    }

    size_t shape[2] = {2, 3};
    blt_tensor tensor = blt_tensor_create(arena, shape, 2, BLT_DTYPE_FP32);
    if (tensor.numel != 6 || tensor.ndim != 2 || tensor.shape[0] != 2 || tensor.shape[1] != 3) {
        fprintf(stderr, "[FAIL] tensor creation metadata\n");
        blt_arena_destroy(arena);
        return 0;
    }

    if (!tensor.data) {
        fprintf(stderr, "[FAIL] tensor allocation data\n");
        blt_arena_destroy(arena);
        return 0;
    }

    ((float*)tensor.data)[0] = 1.0f;
    blt_arena_reset(arena);
    if (arena->offset != 0) {
        fprintf(stderr, "[FAIL] arena reset\n");
        blt_arena_destroy(arena);
        return 0;
    }

    blt_arena_destroy(arena);
    return 1;
}
