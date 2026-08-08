#include "blt/core/backend.h"
#include "blt/ops/matmul.h"

void blt_matmul_cpu(const blt_tensor* a, const blt_tensor* b, blt_tensor* out) {
    blt_check_nd_fp32(a, 2, (const size_t[]){0, 0}, "matmul: input tensor a must be 2D FP32");
    blt_check_nd_fp32(b, 2, (const size_t[]){0, 0}, "matmul: input tensor b must be 2D FP32");
    blt_check_nd_fp32(out, 2, (const size_t[]){0, 0}, "matmul: output tensor must be 2D FP32");
    BLT_REQUIRE(a->shape[1] == b->shape[0], "matmul inner dimensions do not match");
    BLT_REQUIRE(out->shape[0] == a->shape[0] && out->shape[1] == b->shape[1], "matmul output shape is incorrect");

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
