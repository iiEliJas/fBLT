#ifndef BLT_CORE_CUDA_SHIM_H
#define BLT_CORE_CUDA_SHIM_H

#include <stddef.h>
#include "blt/core/allocator.h"

// Internal host-side bridge to CUDA memory management. Declarations are
// visible in every build mode, but the symbols only exist when the build
// defines BLT_WITH_CUDA (implemented in src/backend_cuda/memory.cu); C99
// translation units therefore never need CUDA headers.
//
// All functions abort via BLT_FATAL on CUDA API failure.

#ifdef BLT_WITH_CUDA

// Internal host-side bridge to CUDA memory management. Declarations are
// visible in every build mode, but the symbols only exist when the build
// defines BLT_WITH_CUDA (implemented in src/backend_cuda/memory.cu); C99
// translation units therefore never need CUDA headers.
//
// All functions abort via BLT_FATAL on CUDA API failure.

#ifdef __cplusplus
extern "C" {
#endif

// Base CUDA memory management
void* blt_cuda_malloc(size_t bytes);
void blt_cuda_free(void* ptr);
void blt_cuda_memset(void* dst, int value, size_t bytes);
void blt_cuda_memcpy_h2d(void* dst, const void* src, size_t bytes);
void blt_cuda_memcpy_d2h(void* dst, const void* src, size_t bytes);

// Scratch arena for temporary CUDA allocations (avoids per-call cudaMalloc/free).
// Initialized on first use with 256MB capacity. Thread-local.
blt_arena* blt_cuda_get_scratch_arena(void);
void blt_cuda_scratch_reset(void);

// Pass synchronization: flush stream and check for kernel execution errors.
void blt_backend_pass_sync_cuda(void);

// Fused RoPE on packed QKV layout
void blt_rope_apply_packed_cuda(float* qkv_data, size_t qkv_stride,
                                size_t head_offset, size_t seq_len,
                                size_t head_dim, const float* cos, const float* sin);
void blt_rope_apply_packed_backward_cuda(float* qkv_data, size_t qkv_stride,
                                         size_t head_offset, size_t seq_len,
                                         size_t head_dim, const float* cos, const float* sin);

#ifdef __cplusplus
}
#endif

#endif // BLT_WITH_CUDA

#endif // BLT_CORE_CUDA_SHIM_H
