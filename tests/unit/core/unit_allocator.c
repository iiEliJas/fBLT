#include <stdio.h>
#include "core/allocator.h"

#include "test_helpers.h"
#include "test_suite.h"

int run_allocator_core_tests(void) {
    blt_arena *arena = blt_arena_create(4096, BLT_BACKEND_CPU);
    if (!arena) {
        fprintf(stderr, "[FAIL] arena creation\n");
        return 0;
    }

    size_t shape[2] = {2, 3};
    blt_tensor tensor = blt_tensor_create(arena, shape, 2, BLT_DTYPE_FP32);
    TEST_ASSERT(tensor.numel == 6);
    TEST_ASSERT(tensor.ndim == 2);
    TEST_ASSERT(tensor.shape[0] == 2);
    TEST_ASSERT(tensor.shape[1] == 3);
    TEST_ASSERT(tensor.data != NULL);

    ((float *)tensor.data)[0] = 1.0f;
    blt_arena_reset(arena);
    TEST_ASSERT(arena->offset == 0);

    blt_arena_destroy(arena);
    return 1;
}
