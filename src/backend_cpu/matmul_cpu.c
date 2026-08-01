#include "blt/core/backend.h"
#include "blt/ops/matmul.h"

void blt_matmul_cpu(const blt_tensor* a, const blt_tensor* b, blt_tensor* out) {
    if (a->ndim != 2 || b->ndim != 2 || out->ndim != 2) {
        BLT_FATAL("matmul expects 2D tensors");
    }
    if (a->shape[1] != b->shape[0]) {
        BLT_FATAL("matmul inner dimensions do not match");
    }
    if (out->shape[0] != a->shape[0] || out->shape[1] != b->shape[1]) {
        BLT_FATAL("matmul output shape is incorrect");
    }
    if (a->dtype != BLT_DTYPE_FP32 || b->dtype != BLT_DTYPE_FP32 || out->dtype != BLT_DTYPE_FP32) {
        BLT_FATAL("matmul only supports FP32 tensors");
    }

    const float* a_data = (const float*)a->data;
    const float* b_data = (const float*)b->data;
    float* out_data = (float*)out->data;

    size_t m = a->shape[0];
    size_t k = a->shape[1];
    size_t n = b->shape[1];

    for (size_t row = 0; row < m; ++row) {
        for (size_t col = 0; col < n; ++col) {
            float sum = 0.0f;
            for (size_t inner = 0; inner < k; ++inner) {
                sum += a_data[row * k + inner] * b_data[inner * n + col];
            }
            out_data[row * n + col] = sum;
        }
    }
}
