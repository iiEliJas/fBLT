// CPU-vs-GPU parity for the reductions family: softmax, cross-entropy,
// RoPE, RMSNorm, LayerNorm (forward). Same harness pattern as
// parity_cuda_elementwise.c.

#include "blt/core/allocator.h"
#include "blt/core/backend.h"
#include "blt/ops/softmax.h"
#include "blt/ops/cross_entropy.h"
#include "blt/ops/rope.h"
#include "blt/ops/rmsnorm.h"
#include "blt/ops/layernorm.h"

#include "test_helpers.h"
#include "test_suite.h"

#include <stdint.h>
#include <string.h>

#ifndef BLT_WITH_CUDA

int run_cuda_parity_reductions(void) { return 1; }

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

static blt_tensor make_like(blt_arena *arena, const blt_tensor *src) {
    return blt_tensor_create(arena, src->shape, src->ndim, src->dtype);
}

#define CUDA_PARITY_COMMON                                                                                             \
    blt_arena *host = blt_arena_create(4 << 20, BLT_BACKEND_CPU);                                                      \
    blt_arena *dev = blt_arena_create(4 << 20, BLT_BACKEND_CUDA);                                                      \
    TEST_ASSERT(host != NULL && dev != NULL);

#define CUDA_PARITY_FINI                                                                                               \
    blt_arena_destroy(dev);                                                                                            \
    blt_arena_destroy(host);

int run_cuda_parity_reductions(void) {
    CUDA_PARITY_COMMON

    const size_t seq_len = 33;
    const size_t vocab = 256; // byte vocab keeps every uint8 target in range
    const size_t embed_dim = 64;
    const size_t heads = 4;
    const size_t head_dim = 16;
    uint32_t rng = 0x87654321u;

    // ---- softmax forward / backward over [seq, vocab] rows ----
    const size_t shape2[2] = {seq_len, vocab};
    blt_tensor logits = blt_tensor_create(host, shape2, 2, BLT_DTYPE_FP32);
    fill_random(&logits, &rng);

    blt_tensor exp_sm = make_like(host, &logits);
    blt_softmax(&logits, &exp_sm);

    blt_tensor dlogits = blt_tensor_to_device(&logits, dev);
    blt_tensor d_sm = make_like(dev, &dlogits);
    blt_softmax(&dlogits, &d_sm);
    blt_tensor got_sm = blt_tensor_to_host(&d_sm, host);
    TEST_ASSERT_CLOSE(&got_sm, &exp_sm, 1e-6f);

    blt_tensor grad_out = make_like(host, &logits);
    fill_random(&grad_out, &rng);
    blt_tensor exp_smb = make_like(host, &logits);
    blt_softmax_backward(&grad_out, &exp_sm, &exp_smb);

    blt_tensor dgrad_out = blt_tensor_to_device(&grad_out, dev);
    blt_tensor d_smb = make_like(dev, &dlogits);
    blt_softmax_backward(&dgrad_out, &d_sm, &d_smb);
    blt_tensor got_smb = blt_tensor_to_host(&d_smb, host);
    TEST_ASSERT_CLOSE(&got_smb, &exp_smb, 1e-6f);

    // ---- cross-entropy forward / backward with uint8 targets ----
    size_t t_shape[1] = {seq_len};
    blt_tensor targets = blt_tensor_create(host, t_shape, 1, BLT_DTYPE_UINT8);
    unsigned char *td = (unsigned char *)targets.data;
    for (size_t i = 0; i < seq_len; i++) {
        td[i] = (unsigned char)(prng_next(&rng) & 0xFF);
    }

    size_t scalar_shape[1] = {1};
    blt_tensor exp_loss = blt_tensor_create(host, scalar_shape, 1, BLT_DTYPE_FP32);
    blt_cross_entropy_forward(&logits, &targets, &exp_loss);

    blt_tensor dtargets = blt_tensor_to_device(&targets, dev);
    blt_tensor d_loss = blt_tensor_create(dev, scalar_shape, 1, BLT_DTYPE_FP32);
    blt_cross_entropy_forward(&dlogits, &dtargets, &d_loss);
    blt_tensor got_loss = blt_tensor_to_host(&d_loss, host);
    TEST_ASSERT_CLOSE(&got_loss, &exp_loss, 1e-4f);

    blt_tensor exp_ceg = make_like(host, &logits);
    blt_cross_entropy_backward(&logits, &targets, &exp_ceg);
    blt_tensor d_ceg = make_like(dev, &dlogits);
    blt_cross_entropy_backward(&dlogits, &dtargets, &d_ceg);
    blt_tensor got_ceg = blt_tensor_to_host(&d_ceg, host);
    TEST_ASSERT_CLOSE(&got_ceg, &exp_ceg, 1e-5f);

    // ---- RoPE precompute + apply + apply-backward ----
    const size_t half_dim = head_dim / 2;
    size_t table_shape[2] = {seq_len, half_dim};
    blt_tensor cos_cpu = blt_tensor_create(host, table_shape, 2, BLT_DTYPE_FP32);
    blt_tensor sin_cpu = blt_tensor_create(host, table_shape, 2, BLT_DTYPE_FP32);
    blt_rope_config rope_cfg = {.theta = 10000.0f, .head_dim = head_dim};
    blt_rope_precompute(seq_len, &rope_cfg, &cos_cpu, &sin_cpu);

    blt_tensor cos_dev = blt_tensor_create(dev, table_shape, 2, BLT_DTYPE_FP32);
    blt_tensor sin_dev = blt_tensor_create(dev, table_shape, 2, BLT_DTYPE_FP32);
    blt_rope_precompute(seq_len, &rope_cfg, &cos_dev, &sin_dev);
    blt_tensor got_cos = blt_tensor_to_host(&cos_dev, host);
    blt_tensor got_sin = blt_tensor_to_host(&sin_dev, host);
    TEST_ASSERT_CLOSE(&got_cos, &cos_cpu, 1e-6f);
    TEST_ASSERT_CLOSE(&got_sin, &sin_cpu, 1e-6f);

    size_t rope_shape[3] = {seq_len, heads, head_dim};
    blt_tensor rx = blt_tensor_create(host, rope_shape, 3, BLT_DTYPE_FP32);
    fill_random(&rx, &rng);
    blt_tensor exp_rot = make_like(host, &rx);
    blt_rope_apply(&rx, &cos_cpu, &sin_cpu, &exp_rot);

    blt_tensor drx = blt_tensor_to_device(&rx, dev);
    blt_tensor d_rot = make_like(dev, &drx);
    blt_rope_apply(&drx, &cos_dev, &sin_dev, &d_rot);
    blt_tensor got_rot = blt_tensor_to_host(&d_rot, host);
    TEST_ASSERT_CLOSE(&got_rot, &exp_rot, 1e-6f);

    blt_tensor rgrad = make_like(host, &rx);
    fill_random(&rgrad, &rng);
    blt_tensor exp_rbwd = make_like(host, &rx);
    blt_rope_apply_backward(&rgrad, &cos_cpu, &sin_cpu, &exp_rbwd);

    blt_tensor drgrad = blt_tensor_to_device(&rgrad, dev);
    blt_tensor d_rbwd = make_like(dev, &drx);
    blt_rope_apply_backward(&drgrad, &cos_dev, &sin_dev, &d_rbwd);
    blt_tensor got_rbwd = blt_tensor_to_host(&d_rbwd, host);
    TEST_ASSERT_CLOSE(&got_rbwd, &exp_rbwd, 1e-6f);

    // ---- RMSNorm forward / backward ----
    size_t norm_shape[2] = {seq_len, embed_dim};
    blt_tensor nx = blt_tensor_create(host, norm_shape, 2, BLT_DTYPE_FP32);
    fill_random(&nx, &rng);
    size_t w_shape[1] = {embed_dim};
    blt_tensor nw = blt_tensor_create(host, w_shape, 1, BLT_DTYPE_FP32);
    fill_random(&nw, &rng);

    blt_tensor exp_rms = make_like(host, &nx);
    blt_rmsnorm_forward(&nx, &nw, &exp_rms);

    blt_tensor dnx = blt_tensor_to_device(&nx, dev);
    blt_tensor dnw = blt_tensor_to_device(&nw, dev);
    blt_tensor d_rms = make_like(dev, &dnx);
    blt_rmsnorm_forward(&dnx, &dnw, &d_rms);
    blt_tensor got_rms = blt_tensor_to_host(&d_rms, host);
    TEST_ASSERT_CLOSE(&got_rms, &exp_rms, 1e-6f);

    blt_tensor ngrad = make_like(host, &nx);
    fill_random(&ngrad, &rng);
    blt_tensor exp_gx = make_like(host, &nx);
    blt_tensor exp_gw = blt_tensor_create(host, w_shape, 1, BLT_DTYPE_FP32); // zeroed by create
    blt_rmsnorm_backward(&ngrad, &nx, &nw, &exp_gx, &exp_gw);

    blt_tensor dngrad = blt_tensor_to_device(&ngrad, dev);
    blt_tensor d_gx = make_like(dev, &dnx);
    blt_tensor d_gw = blt_tensor_create(dev, w_shape, 1, BLT_DTYPE_FP32);
    blt_rmsnorm_backward(&dngrad, &dnx, &dnw, &d_gx, &d_gw);
    blt_tensor got_gx = blt_tensor_to_host(&d_gx, host);
    blt_tensor got_gw = blt_tensor_to_host(&d_gw, host);
    TEST_ASSERT_CLOSE(&got_gx, &exp_gx, 1e-5f);
    TEST_ASSERT_CLOSE(&got_gw, &exp_gw, 1e-5f);

    // ---- LayerNorm forward ----
    blt_tensor nb = blt_tensor_create(host, w_shape, 1, BLT_DTYPE_FP32);
    fill_random(&nb, &rng);
    blt_tensor exp_ln = make_like(host, &nx);
    blt_layernorm_forward(&nx, &nw, &nb, &exp_ln, 1e-5f);

    blt_tensor dnb = blt_tensor_to_device(&nb, dev);
    blt_tensor d_ln = make_like(dev, &dnx);
    blt_layernorm_forward(&dnx, &dnw, &dnb, &d_ln, 1e-5f);
    blt_tensor got_ln = blt_tensor_to_host(&d_ln, host);
    TEST_ASSERT_CLOSE(&got_ln, &exp_ln, 1e-6f);

    CUDA_PARITY_FINI
    return 1;
}

#endif // BLT_WITH_CUDA
