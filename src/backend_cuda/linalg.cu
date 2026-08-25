#include "blt/core/backend.h"
#include "blt/ops/matmul.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

// fp32 GEMMs on cuBLAS. cuBLAS is column-major; row-major C = A @ B maps to
// the standard swapped-operand formulation C' = B' @ A' (primes denote the
// col-major views of the same row-major buffers), so no transposes or
// packing copies are needed anywhere.

static cublasHandle_t blt_cublas_handle(void) {
    // Lazily-created per-process handle; never destroyed (process teardown reclaims it).
    static cublasHandle_t handle = NULL;
    if (!handle) {
        if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
            BLT_FATAL("blt_matmul: cublasCreate failed");
        }
    }
    return handle;
}

static void blt_cublas_check(cublasStatus_t st, const char* what) {
    if (st != CUBLAS_STATUS_SUCCESS) {
        BLT_FATAL("%s failed with cuBLAS status %d", what, (int)st);
    }
}

extern "C" void blt_matmul_cuda(const blt_tensor* a, const blt_tensor* b, blt_tensor* out) {
    const size_t m = a->shape[0];
    const size_t k = a->shape[1];
    const size_t n = b->shape[1];
    if (m == 0 || k == 0 || n == 0) {
        return;
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasHandle_t h = blt_cublas_handle();
    blt_cublas_check(
        cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N,
                    (int)n, (int)m, (int)k,
                    &alpha,
                    (const float*)b->data, (int)n,
                    (const float*)a->data, (int)k,
                    &beta,
                    (float*)out->data, (int)n),
        "blt_matmul: cublasSgemm");
}

// a: [M,K], b: [K,N], grad_out: [M,N]
// grad_a: [M,K] = grad_out @ b^T   (overwritten when non-NULL)
// grad_b: [K,N] = a^T @ grad_out   (overwritten when non-NULL)
extern "C" void blt_matmul_backward_cuda(const blt_tensor* a, const blt_tensor* b, const blt_tensor* grad_out,
                                         blt_tensor* grad_a, blt_tensor* grad_b) {
    const size_t m = a->shape[0];
    const size_t k = a->shape[1];
    const size_t n = b->shape[1];
    if (m == 0 || k == 0 || n == 0) {
        return;
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasHandle_t h = blt_cublas_handle();

    if (grad_a != NULL) {
        // Rc[K,M] = Bc^T @ Gc : op(T) on b, op(N) on grad_out
        blt_cublas_check(
            cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N,
                        (int)k, (int)m, (int)n,
                        &alpha,
                        (const float*)b->data, (int)n,
                        (const float*)grad_out->data, (int)n,
                        &beta,
                        (float*)grad_a->data, (int)k),
            "blt_matmul_backward: cublasSgemm (grad_a)");
    }

    if (grad_b != NULL) {
        // Sc[N,K] = Gc @ Ac^T : op(N) on grad_out, op(T) on a
        blt_cublas_check(
            cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_T,
                        (int)n, (int)k, (int)m,
                        &alpha,
                        (const float*)grad_out->data, (int)n,
                        (const float*)a->data, (int)k,
                        &beta,
                        (float*)grad_b->data, (int)n),
            "blt_matmul_backward: cublasSgemm (grad_b)");
    }
}
