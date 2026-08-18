#ifndef BLT_CORE_ALLOCATOR_H
#define BLT_CORE_ALLOCATOR_H

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

#endif // BLT_CORE_ALLOCATOR_H
