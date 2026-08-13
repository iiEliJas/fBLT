#include <stdio.h>

#include "blt/core/allocator.h"
#include "blt/ops/matmul.h"
#include "blt/ops/softmax.h"

#include "test_helpers.h"
#include "test_suite.h"



int run_softmax_parity_test(void) {
    blt_arena* arena = blt_arena_create(1024 * 1024, BLT_BACKEND_CPU);
    blt_tensor input = {0};
    blt_tensor expected = {0};
    blt_tensor output = {0};

    int ok = 0;
    if (!load_binary_tensor("data/math_softmax_in.bin", arena, &input) ||
        !load_binary_tensor("data/math_softmax_out.bin", arena, &expected)) {
        blt_arena_destroy(arena);
        return ok;
    }

    output = blt_tensor_create(arena, input.shape, input.ndim, BLT_DTYPE_FP32);
    blt_softmax(&input, &output);
    ok = check_close(&output, &expected, 1e-4f);

    blt_arena_destroy(arena);
    return ok;
}



// ---------------------------------------------------------------
// Matmul parity test
      
int run_matmul_parity_test(void) {
    blt_arena* arena = blt_arena_create(1024 * 1024, BLT_BACKEND_CPU);
    blt_tensor a = {0};
    blt_tensor b = {0};
    blt_tensor out = {0};
    blt_tensor expected = {0};

    int ok = 0;
    if (!load_binary_tensor("data/math_matmul_a.bin", arena, &a) ||
        !load_binary_tensor("data/math_matmul_b.bin", arena, &b) ||
        !load_binary_tensor("data/math_matmul_out.bin", arena, &expected)) {
        blt_arena_destroy(arena);
        return ok;
    }

    size_t out_shape[2] = {a.shape[0], b.shape[1]};
    out = blt_tensor_create(arena, out_shape, 2, BLT_DTYPE_FP32);
    blt_matmul(&a, &b, &out);
    ok = check_close(&out, &expected, 1e-4f);

    blt_arena_destroy(arena);
    return ok;
}