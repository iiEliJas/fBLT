#include "blt/ops/row_stats.h"
#include "blt/core/backend.h"
#include "blt/core/cuda_shim.h"
#include "blt/core/allocator.h"

#include <cuda_runtime.h>
#include <math.h>
#include <stdlib.h>

static int blt_cuda_launch_check(const char* what) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        BLT_FATAL("%s failed: %s", what, cudaGetErrorString(err));
    }
    return 1;
}

__global__ void blt_entropy_rows_kernel(const float* probs, float* out, size_t rows,
                                        size_t vocab, int use_log2) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < rows;
         i += (size_t)gridDim.x * blockDim.x) {
        const float* p = probs + i * vocab;
        float entropy = 0.0f;
        for (size_t j = 0; j < vocab; ++j) {
            float pv = p[j];
            if (pv > 1e-9f) {
                if (use_log2) {
                    entropy -= pv * log2f(pv);
                } else {
                    entropy -= pv * logf(pv);
                }
            }
        }
        out[i] = entropy;
    }
}

extern "C" void blt_entropy_rows_cuda(const blt_tensor* probs, blt_tensor* entropy_out, int use_log2) {
    const size_t rows = probs->shape[0];
    const size_t vocab = probs->shape[1];
    blt_entropy_rows_kernel<<<(unsigned)(rows > 4096 ? 4096 : (rows > 0 ? rows : 1)), 256>>>(
        (const float*)probs->data, (float*)entropy_out->data, rows, vocab, use_log2);
    blt_cuda_launch_check("blt_entropy_rows");
}

// One block per row: threads find local maxima, block reduce to the argmax
// with strict > so ties resolve to the lowest index like the CPU path.
__global__ void blt_argmax_rows_kernel(const float* logits, unsigned int* out, size_t vocab) {
    const size_t i = blockIdx.x;
    const size_t tid = threadIdx.x;
    const float* row = logits + i * vocab;

    __shared__ float best_val[256];
    __shared__ unsigned int best_idx[256];

    float lv = -INFINITY;
    unsigned int li = 0;
    for (size_t j = tid; j < vocab; j += blockDim.x) {
        const float v = row[j];
        if (v > lv) {   // strictly greater keeps the lowest index on ties
            lv = v;
            li = (unsigned int)j;
        }
    }
    best_val[tid] = lv;
    best_idx[tid] = li;
    __syncthreads();

    for (size_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if ((size_t)tid < stride) {
            // Prefer the larger value; on ties prefer the lower vocab index
            // so results match the CPU reference exactly.
            if (best_val[tid + stride] > best_val[tid] ||
                (best_val[tid + stride] == best_val[tid] && best_idx[tid + stride] < best_idx[tid])) {
                best_val[tid] = best_val[tid + stride];
                best_idx[tid] = best_idx[tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        out[i] = best_idx[0];
    }
}

extern "C" void blt_argmax_rows_cuda(const blt_tensor* logits, uint32_t* out_ids_host) {
    BLT_REQUIRE(logits->shape[1] >= 1, "blt_argmax_rows: empty vocab dimension");
    const size_t rows = logits->shape[0];
    const size_t vocab = logits->shape[1];

    unsigned int* d_out = (unsigned int*)blt_arena_alloc(blt_cuda_get_scratch_arena(),
                                                         rows * sizeof(unsigned int), 16);
    cudaError_t err = cudaSuccess;
    blt_argmax_rows_kernel<<<(unsigned)rows, 256>>>((const float*)logits->data, d_out, vocab);
    err = cudaGetLastError();
    if (err == cudaSuccess) {
        blt_cuda_memcpy_d2h(out_ids_host, d_out, rows * sizeof(unsigned int));
    }
    if (err != cudaSuccess) {
        BLT_FATAL("blt_argmax_rows failed: %s", cudaGetErrorString(err));
    }
}
