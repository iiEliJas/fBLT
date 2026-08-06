#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "blt/core/allocator.h"
#include "blt/core/backend.h"
#include "blt/ops/matmul.h"
#include "blt/ops/softmax.h"

#include "test_helpers.h"
#include "test_suite.h"

typedef struct {
    const char* name;
    int (*fn)(void);
} test_case;


static int run_matmul_test(void) {
    blt_arena* arena = blt_arena_create(1024 * 1024, BLT_BACKEND_CPU);
    blt_tensor a = {0};
    blt_tensor b = {0};
    blt_tensor out = {0};
    blt_tensor expected = {0};

    int ok = 0;
    if (!load_binary_tensor("data/golden_matmul_a.bin", arena, &a) ||
        !load_binary_tensor("data/golden_matmul_b.bin", arena, &b) ||
        !load_binary_tensor("data/golden_matmul_out.bin", arena, &expected)) {
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

static int run_softmax_test(void) {
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


int main(void) {
    const test_case tests[] = {
        {"matmul parity", run_matmul_test},
        {"softmax parity", run_softmax_test},
        {"tensor core helpers", run_tensor_core_tests},
        {"allocator core helpers", run_allocator_core_tests},
        {"backend dispatch", run_backend_core_tests},
        {"elementwise backend", run_elementwise_backend_tests},
        {"matmul backend", run_matmul_backend_tests},
        {"softmax backend", run_softmax_backend_tests},
        {"entropy model", run_entropy_model_tests},
        {"patcher model", run_patcher_model_tests},
        {"phase1 parity", run_phase1_tests},
        {"attention model", run_attention_model_tests},
    };

    int passed = 0;
    const size_t test_count = sizeof(tests) / sizeof(tests[0]);

    printf("\n------------------------------------------\n");
    printf("            Running FBLT tests...\n");
    printf("------------------------------------------\n");

    for (size_t i = 0; i < test_count; ++i) {
        if (tests[i].fn()) {
            printf("[PASS] %s\n", tests[i].name);
            passed++;
        } else {
            printf("[FAIL] %s\n", tests[i].name);
        }
    }

    printf("------------------------------------------\n");
    printf("Summary: %d/%zu tests passed\n", passed, test_count);
    printf("------------------------------------------\n");
    
    return passed == (int)test_count ? 0 : 1;
}
