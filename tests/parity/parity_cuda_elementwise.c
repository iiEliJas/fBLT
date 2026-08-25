// CPU-vs-GPU parity harness for dispatched ops.
//
// Pattern per op: build deterministic inputs on a host arena, run the
// dispatched entry point on CPU tensors to get expected outputs, then copy
// the inputs into a device arena (blt_tensor_to_device), run the same entry
// point again so dispatch lands in the CUDA implementation, copy results
// back and compare within tolerance. Skips cleanly when built without CUDA.

#include "blt/core/allocator.h"
#include "blt/core/backend.h"
#include "blt/ops/elementwise.h"
#include "blt/ops/gelu.h"
#include "blt/ops/swiglu.h"

#include "test_helpers.h"
#include "test_suite.h"

#include <stdint.h>
#include <string.h>

#ifndef BLT_WITH_CUDA

int run_cuda_parity_elementwise(void) {
    return 1;
}

#else

// xorshift32 PRNG so tests never disturb global rand state.
static uint32_t prng_next(uint32_t* state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

static void fill_random(blt_tensor* t, uint32_t* state) {
    float* d = (float*)t->data;
    for (size_t i = 0; i < t->numel; i++) {
        d[i] = ((float)(prng_next(state) & 0xFFFF) / 32768.0f - 1.0f);
    }
}

static blt_tensor make_like(blt_arena* arena, const blt_tensor* src) {
    return blt_tensor_create(arena, src->shape, src->ndim, src->dtype);
}

#define CUDA_PARITY_COMMON \
    blt_arena* host = blt_arena_create(4 << 20, BLT_BACKEND_CPU); \
    blt_arena* dev = blt_arena_create(4 << 20, BLT_BACKEND_CUDA); \
    TEST_ASSERT(host != NULL && dev != NULL);

#define CUDA_PARITY_FINI \
    blt_arena_destroy(dev); \
    blt_arena_destroy(host);

int run_cuda_parity_elementwise(void) {
    CUDA_PARITY_COMMON

    const size_t shape[2] = {37, 129};
    uint32_t rng = 0x12345678u;

    blt_tensor a = blt_tensor_create(host, shape, 2, BLT_DTYPE_FP32);
    blt_tensor b = blt_tensor_create(host, shape, 2, BLT_DTYPE_FP32);
    fill_random(&a, &rng);
    fill_random(&b, &rng);

    // ---- add / mul ----
    blt_tensor exp_add = make_like(host, &a);
    blt_tensor exp_mul = make_like(host, &a);
    blt_add(&a, &b, &exp_add);
    blt_mul(&a, &b, &exp_mul);

    blt_tensor da = blt_tensor_to_device(&a, dev);
    blt_tensor db = blt_tensor_to_device(&b, dev);
    blt_tensor d_add = make_like(dev, &da);
    blt_tensor d_mul = make_like(dev, &da);
    blt_add(&da, &db, &d_add);
    blt_mul(&da, &db, &d_mul);

    blt_tensor got_add = blt_tensor_to_host(&d_add, host);
    blt_tensor got_mul = blt_tensor_to_host(&d_mul, host);
    TEST_ASSERT_CLOSE(&got_add, &exp_add, 1e-6f);
    TEST_ASSERT_CLOSE(&got_mul, &exp_mul, 1e-6f);

    // ---- scale (in place) ----
    blt_tensor s_cpu = make_like(host, &a);
    memcpy(s_cpu.data, a.data, blt_tensor_bytes(&a));
    blt_scale(&s_cpu, -2.5f);

    blt_tensor ds = blt_tensor_to_device(&a, dev);
    blt_scale(&ds, -2.5f);
    blt_tensor got_scale = blt_tensor_to_host(&ds, host);
    TEST_ASSERT_CLOSE(&got_scale, &s_cpu, 1e-6f);

    // ---- gelu forward / backward ----
    blt_tensor exp_gelu = make_like(host, &a);
    blt_tensor grad_out = make_like(host, &a);
    fill_random(&grad_out, &rng);
    blt_gelu_forward(&a, &exp_gelu);

    blt_tensor exp_ggx = make_like(host, &a);
    blt_gelu_backward(&grad_out, &a, &exp_ggx);

    blt_tensor dg_out = blt_tensor_to_device(&grad_out, dev);
    blt_tensor d_gelu = make_like(dev, &da);
    blt_tensor d_ggx = make_like(dev, &da);
    blt_gelu_forward(&da, &d_gelu);
    blt_gelu_backward(&dg_out, &da, &d_ggx);

    blt_tensor got_gelu = blt_tensor_to_host(&d_gelu, host);
    blt_tensor got_ggx = blt_tensor_to_host(&d_ggx, host);
    TEST_ASSERT_CLOSE(&got_gelu, &exp_gelu, 1e-5f);
    TEST_ASSERT_CLOSE(&got_ggx, &exp_ggx, 1e-5f);

    // ---- swiglu forward / backward ----
    blt_tensor gate_t = make_like(host, &a);
    memcpy(gate_t.data, b.data, blt_tensor_bytes(&b));
    blt_tensor exp_sw = make_like(host, &gate_t);
    blt_tensor exp_sgg = make_like(host, &gate_t);
    blt_tensor exp_sgu = make_like(host, &gate_t);
    blt_swiglu_forward(&gate_t, &a, &exp_sw);
    blt_swiglu_backward(&grad_out, &gate_t, &a, &exp_sgg, &exp_sgu);

    blt_tensor dgate = blt_tensor_to_device(&gate_t, dev);
    blt_tensor d_sw = make_like(dev, &da);
    blt_tensor d_sgg = make_like(dev, &da);
    blt_tensor d_sgu = make_like(dev, &da);
    blt_swiglu_forward(&dgate, &da, &d_sw);
    blt_swiglu_backward(&dg_out, &dgate, &da, &d_sgg, &d_sgu);

    blt_tensor got_sw = blt_tensor_to_host(&d_sw, host);
    blt_tensor got_sgg = blt_tensor_to_host(&d_sgg, host);
    blt_tensor got_sgu = blt_tensor_to_host(&d_sgu, host);
    TEST_ASSERT_CLOSE(&got_sw, &exp_sw, 1e-5f);
    TEST_ASSERT_CLOSE(&got_sgg, &exp_sgg, 1e-5f);
    TEST_ASSERT_CLOSE(&got_sgu, &exp_sgu, 1e-5f);

    CUDA_PARITY_FINI
    return 1;
}

#endif // BLT_WITH_CUDA
