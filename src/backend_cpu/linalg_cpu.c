#include "blt/core/backend.h"
#include "blt/ops/matmul.h"

#include <string.h>


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



// a: [M,K], b: [K,N], grad_out: [M,N]
// grad_a: [M,K] = grad_out @ b^T
// grad_b: [K,N] = a^T @ grad_out
void blt_matmul_backward_cpu(const blt_tensor* a, const blt_tensor* b, const blt_tensor* grad_out,
                          blt_tensor* grad_a, blt_tensor* grad_b) {
    BLT_REQUIRE(a != NULL && b != NULL && grad_out != NULL,
                "blt_matmul_backward: a, b, grad_out must not be NULL");
    BLT_REQUIRE(a->dtype == BLT_DTYPE_FP32 && b->dtype == BLT_DTYPE_FP32 && grad_out->dtype == BLT_DTYPE_FP32,
                "blt_matmul_backward: a, b, grad_out must be FP32");
    BLT_REQUIRE(a->ndim == 2 && b->ndim == 2 && grad_out->ndim == 2,
                "blt_matmul_backward: a, b, grad_out must be 2D");
 
    size_t M = a->shape[0];
    size_t K = a->shape[1];
    BLT_REQUIRE(b->shape[0] == K, "blt_matmul_backward: a.shape[1] must equal b.shape[0]");
    size_t N = b->shape[1];
    BLT_REQUIRE(grad_out->shape[0] == M && grad_out->shape[1] == N,
                "blt_matmul_backward: grad_out shape must be [M, N]");
 
    const float* a_data = (const float*)a->data;
    const float* b_data = (const float*)b->data;
    const float* go_data = (const float*)grad_out->data;    // go_data ;D really cool name right?
 
    if (grad_a != NULL) {
        BLT_REQUIRE(grad_a->dtype == BLT_DTYPE_FP32 && grad_a->ndim == 2 &&
                    grad_a->shape[0] == M && grad_a->shape[1] == K,
                    "blt_matmul_backward: grad_a must be [M, K] FP32");
        float* ga_data = (float*)grad_a->data;
 
        // grad_a[m,k] = sum_n grad_out[m,n] * b[k,n]
        for (size_t m = 0; m < M; m++) {
            const float* go_row = go_data + m * N;
            float* ga_row = ga_data + m * K;
            for (size_t k = 0; k < K; k++) {
                float acc = 0.0f;
                for (size_t n = 0; n < N; n++) {
                    acc += go_row[n] * b_data[k * N + n];
                }
                ga_row[k] = acc;
            }
        }
    }
 
    if (grad_b != NULL) {
        BLT_REQUIRE(grad_b->dtype == BLT_DTYPE_FP32 && grad_b->ndim == 2 &&
                    grad_b->shape[0] == K && grad_b->shape[1] == N,
                    "blt_matmul_backward: grad_b must be [K, N] FP32");
        float* gb_data = (float*)grad_b->data;
 
        // grad_b[k,n] = sum_m a[m,k] * grad_out[m,n]
        memset(gb_data, 0, K * N * sizeof(float));
        for (size_t m = 0; m < M; m++) {
            const float* a_row = a_data + m * K;
            const float* go_row = go_data + m * N;
            for (size_t k = 0; k < K; k++) {
                float av = a_row[k];
                if (av == 0.0f) {
                    continue;
                }
                float* gb_row = gb_data + k * N;
                for (size_t n = 0; n < N; n++) {
                    gb_row[n] += av * go_row[n];
                }
            }
        }
    }
}
