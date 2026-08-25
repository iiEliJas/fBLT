#include "blt/ops/vecmath.h"
#include "blt/core/backend.h"

#include <cuda_runtime.h>
#include <math.h>

// One block reduces one dot product: strided partial sums in shared memory,
// then a deterministic tree reduction. Block size is fixed so the summation
// order is reproducible run to run.
#define BLT_VEC_DOT_BLOCK 256

__global__ void blt_vec_dot_kernel(const float* a, const float* b, size_t n, float* out) {
    __shared__ float partial[BLT_VEC_DOT_BLOCK];
    const size_t tid = threadIdx.x;

    float sum = 0.0f;
    for (size_t i = tid; i < n; i += BLT_VEC_DOT_BLOCK) {
        sum += a[i] * b[i];
    }
    partial[tid] = sum;
    __syncthreads();

    for (size_t stride = BLT_VEC_DOT_BLOCK / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        out[0] = partial[0];
    }
}

extern "C" float blt_vec_dot_cuda(blt_backend backend, const float* a, const float* b, size_t n) {
    (void)backend;
    BLT_REQUIRE(backend == BLT_BACKEND_CUDA, "blt_vec_dot: CUDA implementation called with non-CUDA backend");

    float* d_out = NULL;
    cudaError_t err = cudaMalloc(&d_out, sizeof(float));
    if (err != cudaSuccess) {
        BLT_FATAL("blt_vec_dot: cudaMalloc failed: %s", cudaGetErrorString(err));
    }
    blt_vec_dot_kernel<<<1, BLT_VEC_DOT_BLOCK>>>(a, b, n, d_out);
    err = cudaGetLastError();
    if (err == cudaSuccess) {
        err = cudaDeviceSynchronize();
    }
    if (err != cudaSuccess) {
        cudaFree(d_out);
        BLT_FATAL("blt_vec_dot: kernel failed: %s", cudaGetErrorString(err));
    }
    float result = 0.0f;
    err = cudaMemcpy(&result, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_out);
    if (err != cudaSuccess) {
        BLT_FATAL("blt_vec_dot: result copy failed: %s", cudaGetErrorString(err));
    }
    return result;
}

// Softmax over one contiguous row. Runs single-threaded and mirrors the CPU
// loop order exactly, which keeps masked rows bit-identical between backends.
__global__ void blt_softmax_masked_row_kernel(
    float* row, size_t row_len, size_t row_idx,
    int is_causal, const float* mask_row, float scale) {
    float max_val = -INFINITY;

    for (size_t col = 0; col < row_len; ++col) {
        if (mask_row != NULL) {
            row[col] = row[col] * scale + mask_row[col];
        } else if (is_causal && col > row_idx) {
            row[col] = -INFINITY;
        } else {
            row[col] *= scale;
        }

        if (isfinite(row[col]) && row[col] > max_val) {
            max_val = row[col];
        }
    }

    float sum = 0.0f;
    for (size_t col = 0; col < row_len; ++col) {
        if (isfinite(row[col])) {
            row[col] = expf(row[col] - max_val);
            sum += row[col];
        } else {
            row[col] = 0.0f;
        }
    }

    if (sum > 0.0f) {
        for (size_t col = 0; col < row_len; ++col) {
            row[col] /= sum;
        }
    }
}

extern "C" void blt_softmax_masked_row_inplace_cuda(
    blt_backend backend, float* row, size_t row_len, size_t row_idx,
    int is_causal, const float* mask_row, float scale) {
    (void)backend;
    BLT_REQUIRE(backend == BLT_BACKEND_CUDA, "blt_softmax_masked_row_inplace: CUDA implementation called with non-CUDA backend");

    blt_softmax_masked_row_kernel<<<1, 1>>>(row, row_len, row_idx, is_causal ? 1 : 0, mask_row, scale);
    cudaError_t err = cudaGetLastError();
    if (err == cudaSuccess) {
        err = cudaDeviceSynchronize();
    }
    if (err != cudaSuccess) {
        BLT_FATAL("blt_softmax_masked_row_inplace: kernel failed: %s", cudaGetErrorString(err));
    }
}
