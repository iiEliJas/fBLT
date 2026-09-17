#ifndef BLT_CORE_CUDA_SHIM_H
#define BLT_CORE_CUDA_SHIM_H

#include <stddef.h>
#include "core/allocator.h"

// Host-side bridge to CUDA memory. Symbols exist in every build mode but
// only resolve when BLT_WITH_CUDA is defined (src/backend_cuda/memory.cu),
// so C99 translation units never need CUDA headers.
// All functions abort via BLT_FATAL on CUDA API failure.

#ifdef BLT_WITH_CUDA

#ifdef __cplusplus
extern "C" {
#endif

void *blt_cuda_malloc(size_t bytes);
void blt_cuda_free(void *ptr);
void blt_cuda_memset(void *dst, int value, size_t bytes);
void blt_cuda_memcpy_h2d(void *dst, const void *src, size_t bytes);
void blt_cuda_memcpy_d2h(void *dst, const void *src, size_t bytes);

// Scratch arena: avoids per-call cudaMalloc/free. Thread-local, init on first use.
// Call blt_cuda_set_scratch_size before first use to override the default (512MB).
blt_arena *blt_cuda_get_scratch_arena(void);
void blt_cuda_set_scratch_size(size_t bytes);
void blt_cuda_scratch_reset(void);

// Flush stream + check for kernel errors.
void blt_backend_pass_sync_cuda(void);

void blt_rope_apply_packed_cuda(float *qkv_data, size_t qkv_stride, size_t head_offset, size_t seq_len, size_t head_dim,
                                const float *cos, const float *sin);
void blt_rope_apply_packed_backward_cuda(float *qkv_data, size_t qkv_stride, size_t head_offset, size_t seq_len,
                                         size_t head_dim, const float *cos, const float *sin);

#ifdef __cplusplus
}
#endif

#endif

#endif
