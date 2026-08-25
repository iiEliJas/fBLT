#include "test_helpers.h"
#include "blt/core/allocator.h"
#include "blt/core/backend.h"

#include <stdint.h>

int run_cuda_arena_transfer_tests(void) {
#ifndef BLT_WITH_CUDA
    // CPU-only build exposes no CUDA backend to exercise.
    return 1;
#else
    blt_arena* host = blt_arena_create(1 << 20, BLT_BACKEND_CPU);
    blt_arena* dev = blt_arena_create(1 << 20, BLT_BACKEND_CUDA);
    TEST_ASSERT(host != NULL && dev != NULL);
    TEST_ASSERT(dev->backend == BLT_BACKEND_CUDA && host->buffer != NULL && dev->buffer != NULL);

    const size_t shape[2] = {4, 8};
    blt_tensor t = blt_tensor_create(host, shape, 2, BLT_DTYPE_FP32);
    float* td = (float*)t.data;
    for (size_t i = 0; i < t.numel; i++) {
        td[i] = (float)(i % 17) * 0.5f - 3.0f;
    }

    // Device tensors must come up zero-initialized.
    blt_tensor z = blt_tensor_create(dev, shape, 2, BLT_DTYPE_FP32);
    TEST_ASSERT(z.backend == BLT_BACKEND_CUDA);
    blt_tensor zh = blt_tensor_to_host(&z, host);
    for (size_t i = 0; i < zh.numel; i++) {
        TEST_ASSERT(((const float*)zh.data)[i] == 0.0f);
    }

    // H2D -> D2H round trip is bit-exact and preserves layout metadata.
    blt_tensor d = blt_tensor_to_device(&t, dev);
    TEST_ASSERT(d.backend == BLT_BACKEND_CUDA);
    TEST_ASSERT(d.ndim == t.ndim && d.numel == t.numel &&
                d.shape[0] == t.shape[0] && d.shape[1] == t.shape[1]);
    TEST_ASSERT(((uintptr_t)d.data % 64) == 0);   // device bump allocator honors alignment
    blt_tensor back = blt_tensor_to_host(&d, host);
    for (size_t i = 0; i < back.numel; i++) {
        TEST_ASSERT(((const float*)back.data)[i] == td[i]);
    }

    // Reset frees capacity for reuse on the same slab.
    const size_t off_before = dev->offset;
    blt_tensor_create(dev, shape, 2, BLT_DTYPE_FP32);
    TEST_ASSERT(dev->offset > off_before);
    blt_arena_reset(dev);
    TEST_ASSERT(dev->offset == 0);
    blt_tensor d2 = blt_tensor_to_device(&t, dev);
    TEST_ASSERT(d2.backend == BLT_BACKEND_CUDA);

    blt_arena_destroy(dev);
    blt_arena_destroy(host);
    return 1;
#endif
}
