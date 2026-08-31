#include "blt/core/cuda_shim.h"
#include "blt/core/backend.h"
#include "blt/core/allocator.h"

#include <cuda_runtime.h>

static void blt_cuda_check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        BLT_FATAL("CUDA %s failed: %s", what, cudaGetErrorString(err));
    }
}

// Thread-local scratch arena for temporary CUDA allocations.
// Avoids per-call cudaMalloc/free churn. Initialized on first use with 256MB capacity.
static __thread blt_arena* blt_cuda_scratch_arena = NULL;

extern "C" blt_arena* blt_cuda_get_scratch_arena(void) {
    if (blt_cuda_scratch_arena == NULL) {
        blt_cuda_scratch_arena = blt_arena_create(256 * 1024 * 1024, BLT_BACKEND_CUDA);
    }
    return blt_cuda_scratch_arena;
}

extern "C" void blt_cuda_scratch_reset(void) {
    if (blt_cuda_scratch_arena != NULL) {
        blt_arena_reset(blt_cuda_scratch_arena);
    }
}

extern "C" void* blt_cuda_malloc(size_t bytes) {
    void* ptr = NULL;
    blt_cuda_check(cudaMalloc(&ptr, bytes), "cudaMalloc");
    return ptr;
}

extern "C" void blt_cuda_free(void* ptr) {
    if (!ptr) {
        return;
    }
    cudaError_t err = cudaFree(ptr);
    if (err != cudaSuccess) {
        BLT_WARN("CUDA cudaFree failed: %s", cudaGetErrorString(err));
    }
}

extern "C" void blt_cuda_memset(void* dst, int value, size_t bytes) {
    blt_cuda_check(cudaMemset(dst, value, bytes), "cudaMemset");
}

extern "C" void blt_cuda_memcpy_h2d(void* dst, const void* src, size_t bytes) {
    blt_cuda_check(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice), "cudaMemcpy H2D");
}

extern "C" void blt_cuda_memcpy_d2h(void* dst, const void* src, size_t bytes) {
    blt_cuda_check(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy D2H");
}

extern "C" void blt_backend_pass_sync_cuda(void) {
    cudaError_t err = cudaGetLastError();
    if (err == cudaSuccess) {
        err = cudaDeviceSynchronize();
    }
    if (err != cudaSuccess) {
        BLT_FATAL("blt_backend_pass_sync: %s", cudaGetErrorString(err));
    }
}
