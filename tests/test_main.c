#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "blt/core/allocator.h"
#include "blt/core/backend.h"
#include "blt/ops/matmul.h"
#include "blt/ops/softmax.h"

int run_tensor_core_tests(void);
int run_allocator_core_tests(void);
int run_backend_core_tests(void);
int run_elementwise_backend_tests(void);
int run_matmul_backend_tests(void);
int run_softmax_backend_tests(void);
int run_entropy_model_tests(void);
int run_patcher_model_tests(void);
int run_phase1_tests(void);

typedef struct {
    const char* name;
    int (*fn)(void);
} test_case;

static int load_binary_tensor(const char* path, blt_arena* arena, blt_tensor* out_tensor) {
    FILE* fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "failed to open %s\n", path);
        return 0;
    }

    uint32_t ndim = 0;
    if (fread(&ndim, sizeof(ndim), 1, fp) != 1) {
        fclose(fp);
        return 0;
    }

    size_t shape[BLT_MAX_NDIM] = {0};
    for (uint32_t i = 0; i < ndim; ++i) {
        uint32_t dim = 0;
        if (fread(&dim, sizeof(dim), 1, fp) != 1) {
            fclose(fp);
            return 0;
        }
        shape[i] = dim;
    }

    size_t numel = 1;
    for (uint32_t i = 0; i < ndim; ++i) {
        numel *= shape[i];
    }

    *out_tensor = blt_tensor_create(arena, shape, ndim, BLT_DTYPE_FP32);
    float* data = (float*)out_tensor->data;
    for (size_t i = 0; i < numel; ++i) {
        float value = 0.0f;
        if (fread(&value, sizeof(value), 1, fp) != 1) {
            fclose(fp);
            return 0;
        }
        data[i] = value;
    }

    fclose(fp);
    return 1;
}

static int check_close(const blt_tensor* actual, const blt_tensor* expected, float tol) {
    if (actual->numel != expected->numel) {
        return 0;
    }
    const float* a = (const float*)actual->data;
    const float* b = (const float*)expected->data;
    for (size_t i = 0; i < actual->numel; ++i) {
        if (fabsf(a[i] - b[i]) > tol) {
            fprintf(stderr, "mismatch at %zu: got %.6f expected %.6f\n", i, a[i], b[i]);
            return 0;
        }
    }
    return 1;
}

static int run_matmul_test(void) {
    blt_arena* arena = blt_arena_create(1024 * 1024, BLT_BACKEND_CPU);
    blt_tensor a = {0};
    blt_tensor b = {0};
    blt_tensor out = {0};
    blt_tensor expected = {0};

    int ok = 0;
    if (!load_binary_tensor("tests/golden_matmul_a.bin", arena, &a) ||
        !load_binary_tensor("tests/golden_matmul_b.bin", arena, &b) ||
        !load_binary_tensor("tests/golden_matmul_out.bin", arena, &expected)) {
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
    if (!load_binary_tensor("tests/golden_softmax_in.bin", arena, &input) ||
        !load_binary_tensor("tests/golden_softmax_out.bin", arena, &expected)) {
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

    printf("Summary: %d/%zu tests passed\n", passed, test_count);
    return passed == (int)test_count ? 0 : 1;
}
