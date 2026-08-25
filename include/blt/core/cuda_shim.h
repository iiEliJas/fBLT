#ifndef BLT_CORE_CUDA_SHIM_H
#define BLT_CORE_CUDA_SHIM_H

#include <stddef.h>

// Internal host-side bridge to CUDA memory management. Declarations are
// visible in every build mode, but the symbols only exist when the build
// defines BLT_WITH_CUDA (implemented in src/backend_cuda/memory.cu); C99
// translation units therefore never need CUDA headers.
//
// All functions abort via BLT_FATAL on CUDA API failure.

#ifdef BLT_WITH_CUDA

#ifdef __cplusplus
extern "C" {
#endif

void* blt_cuda_malloc(size_t bytes);
void blt_cuda_free(void* ptr);
void blt_cuda_memset(void* dst, int value, size_t bytes);
void blt_cuda_memcpy_h2d(void* dst, const void* src, size_t bytes);
void blt_cuda_memcpy_d2h(void* dst, const void* src, size_t bytes);

#ifdef __cplusplus
}
#endif

#endif // BLT_WITH_CUDA

#endif // BLT_CORE_CUDA_SHIM_H
