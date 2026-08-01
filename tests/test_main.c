#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "blt/core/allocator.h"
#include "blt/core/backend.h"
#include "blt/ops/matmul.h"
#include "blt/ops/softmax.h"

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
        goto cleanup;
    }

    size_t out_shape[2] = {a.shape[0], b.shape[1]};
    out = blt_tensor_create(arena, out_shape, 2, BLT_DTYPE_FP32);
    blt_matmul(&a, &b, &out);
    ok = check_close(&out, &expected, 1e-4f);

cleanup:
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
        goto cleanup;
    }

    output = blt_tensor_create(arena, input.shape, input.ndim, BLT_DTYPE_FP32);
    blt_softmax(&input, &output);
    ok = check_close(&output, &expected, 1e-4f);

cleanup:
    blt_arena_destroy(arena);
    return ok;
}

int main(void) {
    int passed = 0;
    printf("Running BLT Phase 0 tests...\n");

    if (run_matmul_test()) {
        printf("[PASS] matmul parity\n");
        passed++;
    } else {
        printf("[FAIL] matmul parity\n");
    }

    if (run_softmax_test()) {
        printf("[PASS] softmax parity\n");
        passed++;
    } else {
        printf("[FAIL] softmax parity\n");
    }

    printf("Summary: %d/2 tests passed\n", passed);
    return passed == 2 ? 0 : 1;
}
