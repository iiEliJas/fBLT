#include "blt/core/allocator.h"
#include "blt/core/backend.h"

#include <errno.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#include <malloc.h>
#else
#include <unistd.h>
#endif

static void* aligned_malloc(size_t size, size_t alignment) {
#if defined(_WIN32)
    return _aligned_malloc(size, alignment);
#else
    void* ptr = NULL;
    if (posix_memalign(&ptr, alignment, size) != 0) {
        return NULL;
    }
    return ptr;
#endif
}

static void aligned_free(void* ptr) {
#if defined(_WIN32)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

blt_arena* blt_arena_create(size_t capacity_bytes, blt_backend backend) {
    blt_arena* arena = (blt_arena*)calloc(1, sizeof(*arena));
    if (!arena) {
        BLT_FATAL("failed to allocate arena struct");
    }

    arena->capacity = capacity_bytes;
    arena->backend = backend;

    if (backend == BLT_BACKEND_CUDA) {
        BLT_FATAL("CUDA backend is not supported in Phase 0");
    }

    arena->buffer = aligned_malloc(capacity_bytes, 64);
    if (!arena->buffer) {
        free(arena);
        BLT_FATAL("failed to allocate arena buffer");
    }

    return arena;
}

void blt_arena_destroy(blt_arena* arena) {
    if (!arena) {
        return;
    }
    aligned_free(arena->buffer);
    free(arena);
}

void blt_arena_reset(blt_arena* arena) {
    if (!arena) {
        return;
    }
    arena->offset = 0;
}

void* blt_arena_alloc(blt_arena* arena, size_t bytes, size_t alignment) {
    if (!arena || !arena->buffer) {
        BLT_FATAL("invalid arena");
    }

    if (alignment == 0) {
        alignment = 1;
    }

    size_t current = arena->offset;
    size_t aligned = (current + (alignment - 1)) & ~(alignment - 1);
    if (aligned + bytes > arena->capacity) {
        BLT_FATAL("arena allocation exceeded capacity");
    }

    void* ptr = (char*)arena->buffer + aligned;
    arena->offset = aligned + bytes;
    return ptr;
}

blt_tensor blt_tensor_create(blt_arena* arena, const size_t* shape, size_t ndim, blt_dtype dtype) {
    blt_tensor tensor;
    memset(&tensor, 0, sizeof(tensor));

    tensor.dtype = dtype;
    tensor.backend = arena ? arena->backend : BLT_BACKEND_CPU;
    tensor.is_view = false;
    tensor.ndim = ndim;

    for (size_t i = 0; i < BLT_MAX_NDIM; ++i) {
        tensor.shape[i] = (i < ndim) ? shape[i] : 0;
    }

    tensor.numel = blt_tensor_compute_numel(shape, ndim);
    blt_tensor_compute_row_major_strides(shape, ndim, tensor.strides);

    size_t bytes = blt_tensor_bytes(&tensor);
    tensor.data = blt_arena_alloc(arena, bytes, 64);
    if (tensor.data) {
        memset(tensor.data, 0, bytes);
    }

    return tensor;
}
