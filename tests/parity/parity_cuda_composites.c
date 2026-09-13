// CPU-vs-GPU parity for composite ops (attention mask builders, patch
// pooling) and the AdamW optimizer. Same harness pattern as the other
// parity_cuda_* files.

#include "core/allocator.h"
#include "core/backend.h"
#include "ops/mask_builder.h"
#include "ops/patch_pool.h"
#include "ops/optim.h"

#include "test_helpers.h"
#include "test_suite.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

#ifndef BLT_WITH_CUDA

int run_cuda_parity_composites(void) { return 1; }

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

#define CUDA_PARITY_COMMON                                                                                             \
    blt_arena *host = blt_arena_create(4 << 20, BLT_BACKEND_CPU);                                                      \
    blt_arena *dev = blt_arena_create(4 << 20, BLT_BACKEND_CUDA);                                                      \
    TEST_ASSERT(host != NULL && dev != NULL);

#define CUDA_PARITY_FINI                                                                                               \
    blt_arena_destroy(dev);                                                                                            \
    blt_arena_destroy(host);

static int check_mask_configs(void) {
    CUDA_PARITY_COMMON
    uint32_t rng = 0xABCDEF01u;

    // Variant matrix: exercise every mask feature combination.
    const size_t sq = 19;
    const size_t skv = 23;

    size_t docs[3] = {7, 15, 15}; // last boundary must not split anything
    size_t q_groups[19];
    size_t kv_groups[23];
    // Cyclic ids guarantee every group exists on both sides, so no query
    // row ends up without any valid key (the builder rejects those).
    for (size_t i = 0; i < sq; i++) q_groups[i] = i % 3;
    for (size_t j = 0; j < skv; j++) kv_groups[j] = j % 3;
    (void)rng;

    const struct {
        bool is_causal;
        size_t window;
        bool use_docs;
        bool use_groups;
        bool bidir;
        const char *name;
    } variants[] = {
        {true, 0, false, false, false, "causal"},     {false, 5, false, false, false, "window"},
        {true, 4, true, false, false, "causal+docs"}, {false, 0, true, false, false, "docs"},
        {true, 0, false, true, true, "groups-bidir"}, {true, 0, false, true, false, "groups-causal"},
        {true, 6, true, true, true, "everything"},
    };

    for (size_t v = 0; v < sizeof(variants) / sizeof(variants[0]); v++) {
        blt_mask_config cfg;
        memset(&cfg, 0, sizeof(cfg));
        cfg.seq_len_q = sq;
        cfg.seq_len_kv = skv;
        cfg.is_causal = variants[v].is_causal;
        cfg.sliding_window = variants[v].window;
        if (variants[v].use_docs) {
            cfg.doc_boundaries = docs;
            cfg.num_docs = 4;
        }
        if (variants[v].use_groups) {
            cfg.query_group_ids = q_groups;
            cfg.kv_group_ids = kv_groups;
            cfg.bidirectional_within_group = variants[v].bidir;
        }

        blt_tensor m_cpu = {0};
        blt_build_attention_mask(&cfg, &m_cpu, host);
        blt_tensor m_dev = {0};
        blt_build_attention_mask(&cfg, &m_dev, dev);
        TEST_ASSERT(m_dev.backend == BLT_BACKEND_CUDA);

        blt_tensor m_host_copy = blt_tensor_to_host(&m_dev, host);
        TEST_ASSERT_CLOSE(&m_host_copy, &m_cpu, 0.0f);
    }

    // ---- block diffusion masks: TRAIN and INFER ----
    const size_t S = 24;
    const size_t N = 12;
    const size_t B = 2; // blocks of 2 tile [12, 24) exactly

    blt_block_diffusion_config bcfg;
    memset(&bcfg, 0, sizeof(bcfg));
    bcfg.seq_len = S;
    bcfg.num_clean = N;

    bcfg.mode = BLT_BDM_TRAIN;
    bcfg.block_size = B;
    blt_tensor t_cpu = {0};
    blt_build_block_diffusion_mask(&bcfg, &t_cpu, host);
    blt_tensor t_dev = {0};
    blt_build_block_diffusion_mask(&bcfg, &t_dev, dev);
    blt_tensor t_copy = blt_tensor_to_host(&t_dev, host);
    TEST_ASSERT_CLOSE(&t_copy, &t_cpu, 0.0f);

    bcfg.mode = BLT_BDM_INFER;
    blt_tensor i_cpu = {0};
    blt_build_block_diffusion_mask(&bcfg, &i_cpu, host);
    blt_tensor i_dev = {0};
    blt_build_block_diffusion_mask(&bcfg, &i_dev, dev);
    blt_tensor i_copy = blt_tensor_to_host(&i_dev, host);
    TEST_ASSERT_CLOSE(&i_copy, &i_cpu, 0.0f);

    CUDA_PARITY_FINI
    return 1;
}

static int check_patch_pool(void) {
    CUDA_PARITY_COMMON
    uint32_t rng = 0x55AA55AAu;

    const size_t seq_len = 40;
    const size_t embed_dim = 16;
    size_t shape[2] = {seq_len, embed_dim};

    blt_patch_info patches[7];
    const size_t lengths[7] = {1, 9, 2, 8, 3, 7, 10}; // tiles [0, 40) exactly
    size_t pos = 0;
    for (size_t j = 0; j < 7; j++) {
        patches[j].start_idx = pos;
        patches[j].length = lengths[j];
        patches[j].peak_entropy = 0.5f;
        pos += patches[j].length;
    }
    TEST_ASSERT(pos == seq_len);

    blt_tensor hidden = blt_tensor_create(host, shape, 2, BLT_DTYPE_FP32);
    fill_random(&hidden, &rng);

    size_t out_shape[2] = {7, embed_dim};
    blt_tensor grad_out = blt_tensor_create(host, out_shape, 2, BLT_DTYPE_FP32);
    fill_random(&grad_out, &rng);

    for (size_t p = 0; p < 2; p++) {
        const blt_patch_pool_type pool = (p == 0) ? BLT_POOL_MEAN : BLT_POOL_MAX;

        blt_tensor exp_fwd = blt_tensor_create(host, out_shape, 2, BLT_DTYPE_FP32);
        blt_patch_pool_forward(&hidden, patches, 7, pool, &exp_fwd);

        blt_tensor d_hidden = blt_tensor_to_device(&hidden, dev);
        blt_tensor d_fwd = blt_tensor_create(dev, out_shape, 2, BLT_DTYPE_FP32);
        blt_patch_pool_forward(&d_hidden, patches, 7, pool, &d_fwd);
        blt_tensor got_fwd = blt_tensor_to_host(&d_fwd, host);
        TEST_ASSERT_CLOSE(&got_fwd, &exp_fwd, pool == BLT_POOL_MEAN ? 1e-6f : 0.0f);

        blt_tensor exp_gh = blt_tensor_create(host, shape, 2, BLT_DTYPE_FP32);
        blt_patch_pool_backward(&grad_out, &hidden, patches, 7, pool, &exp_gh);

        blt_tensor d_grad_out = blt_tensor_to_device(&grad_out, dev);
        blt_tensor d_gh = blt_tensor_create(dev, shape, 2, BLT_DTYPE_FP32);
        blt_patch_pool_backward(&d_grad_out, &d_hidden, patches, 7, pool, &d_gh);
        blt_tensor got_gh = blt_tensor_to_host(&d_gh, host);
        TEST_ASSERT_CLOSE(&got_gh, &exp_gh, 1e-6f);
    }

    CUDA_PARITY_FINI
    return 1;
}

static int check_optimizers(void) {
    CUDA_PARITY_COMMON
    uint32_t rng = 0x0BADF00Du;

    const size_t n = 257;
    size_t shape[1] = {n};

    blt_tensor p0 = blt_tensor_create(host, shape, 1, BLT_DTYPE_FP32);
    fill_random(&p0, &rng);
    blt_tensor g = blt_tensor_create(host, shape, 1, BLT_DTYPE_FP32);
    fill_random(&g, &rng);

    // ---- sgd ----
    blt_tensor cpu_p = blt_tensor_create(host, shape, 1, BLT_DTYPE_FP32);
    memcpy(cpu_p.data, p0.data, blt_tensor_bytes(&cpu_p));
    blt_sgd_step(&cpu_p, &g, 0.05f);

    blt_tensor d_p = blt_tensor_to_device(&p0, dev);
    blt_tensor d_g = blt_tensor_to_device(&g, dev);
    blt_sgd_step(&d_p, &d_g, 0.05f);
    blt_tensor got_p = blt_tensor_to_host(&d_p, host);
    TEST_ASSERT_CLOSE(&got_p, &cpu_p, 1e-7f);

    // ---- adamw over several steps with decoupled weight decay ----
    // Start from pristine params: d_p above was consumed by the sgd check.
    d_p = blt_tensor_to_device(&p0, dev);

    blt_adamw_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.lr = 0.01f;
    cfg.beta1 = 0.9f;
    cfg.beta2 = 0.999f;
    cfg.eps = 1e-8f;
    cfg.weight_decay = 0.1f;

    blt_tensor c_m = blt_tensor_create(host, shape, 1, BLT_DTYPE_FP32); // zeroed
    blt_tensor c_v = blt_tensor_create(host, shape, 1, BLT_DTYPE_FP32);
    blt_tensor c_p = blt_tensor_create(host, shape, 1, BLT_DTYPE_FP32);
    memcpy(c_p.data, p0.data, blt_tensor_bytes(&c_p));

    blt_tensor d_m = blt_tensor_create(dev, shape, 1, BLT_DTYPE_FP32);
    blt_tensor d_v = blt_tensor_create(dev, shape, 1, BLT_DTYPE_FP32);

    float max_abs_diff = 0.0f;
    for (size_t step = 1; step <= 5; step++) {
        cfg.step = step;
        blt_adamw_step(&c_p, &g, &c_m, &c_v, &cfg);
        blt_adamw_step(&d_p, &d_g, &d_m, &d_v, &cfg);
    }
    got_p = blt_tensor_to_host(&d_p, host);
    const float *a = (const float *)got_p.data;
    const float *b = (const float *)c_p.data;
    for (size_t i = 0; i < n; i++) {
        const float diff = fabsf(a[i] - b[i]);
        if (diff > max_abs_diff) max_abs_diff = diff;
    }
    TEST_ASSERT(max_abs_diff <= 1e-5f);

    CUDA_PARITY_FINI
    return 1;
}

int run_cuda_parity_composites(void) {
    TEST_ASSERT(check_mask_configs());
    TEST_ASSERT(check_patch_pool());
    TEST_ASSERT(check_optimizers());
    return 1;
}

#endif // BLT_WITH_CUDA
