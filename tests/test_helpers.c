#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include "test_helpers.h"


int blt_test_load_binary_tensor(const char* path, blt_arena* arena, blt_tensor* out_tensor) {
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


int blt_test_check_close(const blt_tensor* actual, const blt_tensor* expected, float tol) {
    if (actual->numel != expected->numel) {
        fprintf(stderr, "Shape mismatch: actual numel %zu != expected numel %zu\n", 
                actual->numel, expected->numel);
        return 0;
    }
    const float* a = (const float*)actual->data;
    const float* b = (const float*)expected->data;
    for (size_t i = 0; i < actual->numel; ++i) {
        if (fabsf(a[i] - b[i]) > tol) {
            fprintf(stderr, "Mismatch at index %zu: got %.6f, expected %.6f\n", i, a[i], b[i]);
            return 0;
        }
    }
    return 1;
}