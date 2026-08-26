#ifndef BLT_CORE_ALLOCATOR_H
#define BLT_CORE_ALLOCATOR_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include "blt/core/tensor.h"

typedef struct {
    void* buffer;
    size_t capacity;
    size_t offset;
    blt_backend backend;
} blt_arena;

blt_arena* blt_arena_create(size_t capacity_bytes, blt_backend backend);
void blt_arena_destroy(blt_arena* arena);
void blt_arena_reset(blt_arena* arena);
void* blt_arena_alloc(blt_arena* arena, size_t bytes, size_t alignment);
blt_tensor blt_tensor_create(blt_arena* arena, const size_t* shape, size_t ndim, blt_dtype dtype);

// Copies a host tensor's contents into a fresh tensor allocated from a CUDA
// device arena. Source must use the CPU backend. Requires BLT_WITH_CUDA.
blt_tensor blt_tensor_to_device(const blt_tensor* src, blt_arena* device_arena);

// Copies any tensor's contents into a fresh tensor allocated from a host
// (CPU-backend) arena. Device sources are copied D2H; host sources memcpy.
blt_tensor blt_tensor_to_host(const blt_tensor* src, blt_arena* host_arena);

// Allocates a host-side bookkeeping structure (model containers, handle
// arrays). On CPU arenas this is ordinary arena memory; on CUDA arenas the
// block is malloc'd on the host so containers never land in device memory.
void* blt_container_alloc(blt_arena* arena, size_t bytes);

// Copies `bytes` from host memory into an EXISTING tensor's storage
// (H2D when the tensor lives on the CUDA backend; plain memcpy on CPU).
void blt_tensor_upload(blt_tensor* dst, const void* host_src, size_t bytes);

// Copies a tensor's storage into host memory (D2H on CUDA; memcpy on CPU).
void blt_tensor_download(const blt_tensor* src, void* host_dst, size_t bytes);

#ifdef __cplusplus
}
#endif

#endif // BLT_CORE_ALLOCATOR_H
