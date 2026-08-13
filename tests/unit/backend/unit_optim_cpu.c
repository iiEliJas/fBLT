#include "test_helpers.h"
#include "test_suite.h"

#include "blt/ops/optim.h"
#include "blt/core/allocator.h"
#include "blt/core/tensor.h"


// Single-parameter check
static int test_sgd_step_known_value(void) {
    blt_arena* arena = blt_arena_create(64 * 1024, BLT_BACKEND_CPU);
    if (!arena) {
        return 0;
    }

    size_t shape[1] = { 4 };
    blt_tensor param = blt_tensor_create(arena, shape, 1, BLT_DTYPE_FP32);
    blt_tensor grad  = blt_tensor_create(arena, shape, 1, BLT_DTYPE_FP32);

    float* p = (float*)param.data;
    float* g = (float*)grad.data;
    p[0] = 1.0f; p[1] = -2.0f; p[2] = 0.5f; p[3] = 10.0f;
    g[0] = 0.1f; g[1] =  0.2f; g[2] = -1.0f; g[3] =  4.0f;

    float lr = 0.5f;
    blt_sgd_step(&param, &grad, lr);

    // expected[i] = p_before[i] - lr * g_before[i]
    float expected[4] = { 1.0f - 0.5f * 0.1f, -2.0f - 0.5f * 0.2f, 0.5f - 0.5f * -1.0f, 10.0f - 0.5f * 4.0f };

    int ok = 1;
    for (size_t i = 0; i < 4; ++i) {
        float diff = p[i] - expected[i];
        if (diff < 0) diff = -diff;
        if (diff > 1e-6f) {
            ok = 0;
        }
    }
    TEST_ASSERT(ok);

    blt_arena_destroy(arena);
    return ok;
}



// No-op check: grad = 0 leaves param unchanged
static int test_sgd_step_zero_grad(void) {
    blt_arena* arena = blt_arena_create(64 * 1024, BLT_BACKEND_CPU);
    if (!arena) {
        return 0;
    }

    size_t shape[1] = { 3 };
    blt_tensor param = blt_tensor_create(arena, shape, 1, BLT_DTYPE_FP32);
    blt_tensor grad  = blt_tensor_create(arena, shape, 1, BLT_DTYPE_FP32); // zero-init

    float* p = (float*)param.data;
    p[0] = 3.14f; p[1] = -7.0f; p[2] = 42.0f;
    float before[3] = { p[0], p[1], p[2] };

    blt_sgd_step(&param, &grad, 0.1f);

    int ok = (p[0] == before[0]) && (p[1] == before[1]) && (p[2] == before[2]);
    TEST_ASSERT(ok);

    blt_arena_destroy(arena);
    return ok;
}



// Loop-update drift check: repeated SGD steps toward a fixed pseudo-gradient
// dont accumulate error beyond floating-point tolerance
static int test_sgd_step_loop(void) {
    blt_arena* arena = blt_arena_create(64 * 1024, BLT_BACKEND_CPU);
    if (!arena) {
        return 0;
    }

    size_t shape[1] = { 1 };
    blt_tensor param = blt_tensor_create(arena, shape, 1, BLT_DTYPE_FP32);
    blt_tensor grad  = blt_tensor_create(arena, shape, 1, BLT_DTYPE_FP32);

    float* p = (float*)param.data;
    float* g = (float*)grad.data;
    p[0] = 100.0f;
    g[0] = 1.0f;
    float lr = 1.0f;

    for (int step = 0; step < 100; ++step) {
        blt_sgd_step(&param, &grad, lr);
    }

    // 100 steps of param -= 1.0 * 1.0 starting at 100.0 -> 0.0
    float diff = p[0] - 0.0f;
    if (diff < 0) diff = -diff;
    int ok = diff <= 1e-3f;
    TEST_ASSERT(ok);

    blt_arena_destroy(arena);
    return ok;
}


int run_optim_backend_tests(void) {
    int ok = 1;
    ok &= test_sgd_step_known_value();
    ok &= test_sgd_step_zero_grad();
    ok &= test_sgd_step_loop();
    return ok;
}