// CPU-vs-GPU parity for cuBLAS-backed matmul forward/backward.

#include "blt/core/allocator.h"
#include "blt/core/backend.h"
#include "blt/ops/matmul.h"

#include "test_helpers.h"
#include "test_suite.h"

#include <stdint.h>

#ifndef BLT_WITH_CUDA

int run_cuda_parity_matmul(void) { return 1; }

#else

static uint32_t prng_next(uint32_t *state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

static void fill_random(blt_tensor *t, uint32_t *state) {
    float *d = (float *)t->data;
    for (size_t i = 0; i < t->numel; i++) {
        d[i] = ((float)(prng_next(state) & 0xFFFF) / 32768.0f - 1.0f);
    }
}

static blt_tensor make_2d(blt_arena *arena, size_t rows, size_t cols, uint32_t *state) {
    size_t shape[2] = {rows, cols};
    blt_tensor t = blt_tensor_create(arena, shape, 2, BLT_DTYPE_FP32);
    if (state) {
        fill_random(&t, state);
    }
    return t;
}

int run_cuda_parity_matmul(void) {
    blt_arena *host = blt_arena_create(4 << 20, BLT_BACKEND_CPU);
    blt_arena *dev = blt_arena_create(4 << 20, BLT_BACKEND_CUDA);
    TEST_ASSERT(host != NULL && dev != NULL);

    uint32_t rng = 0xCAFEBABEu;

    const size_t cases[4][3] = {
        {1, 1, 1},      // degenerate single element
        {17, 33, 5},    // skinny, odd dims
        {64, 128, 256}, // typical transformer shapes
        {129, 65, 257}, // primes everywhere
    };

    for (size_t c = 0; c < sizeof(cases) / sizeof(cases[0]); c++) {
        const size_t M = cases[c][0];
        const size_t K = cases[c][1];
        const size_t N = cases[c][2];

        blt_tensor a = make_2d(host, M, K, &rng);
        blt_tensor b = make_2d(host, K, N, &rng);
        blt_tensor go = make_2d(host, M, N, &rng);

        // ---- forward ----
        size_t out_shape[2] = {M, N};
        blt_tensor exp_out = blt_tensor_create(host, out_shape, 2, BLT_DTYPE_FP32);
        blt_matmul(&a, &b, &exp_out);

        blt_tensor da = blt_tensor_to_device(&a, dev);
        blt_tensor db = blt_tensor_to_device(&b, dev);
        blt_tensor d_out = blt_tensor_create(dev, out_shape, 2, BLT_DTYPE_FP32);
        blt_matmul(&da, &db, &d_out);
        blt_tensor got_out = blt_tensor_to_host(&d_out, host);
        TEST_ASSERT_CLOSE(&got_out, &exp_out, 1e-4f);

        // ---- backward with both grads ----
        blt_tensor exp_ga = make_2d(host, M, K, NULL);
        blt_tensor exp_gb = make_2d(host, K, N, NULL);
        blt_matmul_backward(&a, &b, &go, &exp_ga, &exp_gb);

        blt_tensor dgo = blt_tensor_to_device(&go, dev);
        blt_tensor d_ga = blt_tensor_create(dev, a.shape, 2, BLT_DTYPE_FP32);
        blt_tensor d_gb = blt_tensor_create(dev, b.shape, 2, BLT_DTYPE_FP32);
        blt_matmul_backward(&da, &db, &dgo, &d_ga, &d_gb);
        blt_tensor got_ga = blt_tensor_to_host(&d_ga, host);
        blt_tensor got_gb = blt_tensor_to_host(&d_gb, host);
        TEST_ASSERT_CLOSE(&got_ga, &exp_ga, 1e-4f);
        TEST_ASSERT_CLOSE(&got_gb, &exp_gb, 1e-4f);

        // ---- backward with only grad_b (NULL grad_a path) ----
        blt_tensor exp_gb2 = make_2d(host, K, N, NULL);
        blt_matmul_backward(&a, &b, &go, NULL, &exp_gb2);
        d_gb = blt_tensor_create(dev, b.shape, 2, BLT_DTYPE_FP32);
        blt_matmul_backward(&da, &db, &dgo, NULL, &d_gb);
        got_gb = blt_tensor_to_host(&d_gb, host);
        TEST_ASSERT_CLOSE(&got_gb, &exp_gb2, 1e-4f);

        // ---- backward with only grad_a (NULL grad_b path) ----
        blt_tensor exp_ga2 = make_2d(host, M, K, NULL);
        blt_matmul_backward(&a, &b, &go, &exp_ga2, NULL);
        d_ga = blt_tensor_create(dev, a.shape, 2, BLT_DTYPE_FP32);
        blt_matmul_backward(&da, &db, &dgo, &d_ga, NULL);
        got_ga = blt_tensor_to_host(&d_ga, host);
        TEST_ASSERT_CLOSE(&got_ga, &exp_ga2, 1e-4f);
    }

    blt_arena_destroy(dev);
    blt_arena_destroy(host);
    return 1;
}

#endif // BLT_WITH_CUDA
