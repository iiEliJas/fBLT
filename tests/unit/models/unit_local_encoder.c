#include <stdlib.h>
#include <math.h>

#include "test_suite.h"
#include "test_helpers.h"
#include "blt/ops/mask_builder.h"
#include "blt/models/local_encoder.h"
#include "blt/ops/matmul.h"
#include "blt/ops/optim.h"


static size_t rand_range(size_t lo, size_t hi) {
    if (lo >= hi) return lo;
    return lo + (size_t)(rand() % (hi - lo + 1));
}

// Makes rand, increasing doc boundaries in [1, seq_len)
static size_t make_doc_boundaries(size_t seq_len, size_t* out, size_t max_docs) {
    if (seq_len <= 1) return 0;

    size_t target_docs = 1 + (size_t)(rand() % max_docs);
    if (target_docs > seq_len) target_docs = seq_len;
    size_t target_boundaries = target_docs - 1;

    size_t placed = 0;
    size_t last = 0;

    while (placed < target_boundaries) {
        size_t remaining_boundaries = target_boundaries - placed;
        size_t max_candidate = seq_len - remaining_boundaries;
        size_t min_candidate = last + 1;

        if (min_candidate > max_candidate) break;

        size_t candidate = rand_range(min_candidate, max_candidate);
        out[placed++] = candidate;
        last = candidate;
    }

    return placed;
}



static size_t doc_of(size_t pos, const size_t* doc_boundaries, size_t num_boundaries) {
    size_t doc = 0;
    for (size_t d = 0; d < num_boundaries; d++) {
        if (pos >= doc_boundaries[d]) doc = d + 1;
    }
    return doc;
}



int run_lencoder_mask_model_test(void) {
    const int NUM_TRIALS = 200;
    blt_arena* arena = blt_arena_create(4 * 1024 * 1024, BLT_BACKEND_CPU);
    if (!arena) return 0;

    for (int trial = 0; trial < NUM_TRIALS; trial++) {
        blt_arena_reset(arena);

        size_t seq_len = rand_range(4, 64);
        size_t window = (rand() % 4 == 0) ? 0 : rand_range(1, seq_len);

        size_t doc_boundaries[16];
        
        for (int i = 0; i < 16; i++) {
            doc_boundaries[i] = seq_len;
        }

        size_t num_boundaries = make_doc_boundaries(seq_len, doc_boundaries, 6);

        blt_mask_config cfg = {0};
        cfg.seq_len_q = seq_len;
        cfg.seq_len_kv = seq_len;
        cfg.sliding_window = window;
        cfg.doc_boundaries = doc_boundaries;
        cfg.num_docs = num_boundaries + 1;
        cfg.is_causal = true;
        cfg.query_group_ids = NULL;
        cfg.kv_group_ids = NULL;

        blt_tensor mask = {0};
        blt_build_attention_mask(&cfg, &mask, arena);

        TEST_ASSERT(mask.ndim == 2);
        TEST_ASSERT(mask.shape[0] == seq_len);
        TEST_ASSERT(mask.shape[1] == seq_len);

        const float* data = (const float*)mask.data;

        for (size_t i = 0; i < seq_len; i++) {
            size_t di = doc_of(i, doc_boundaries, num_boundaries);
            for (size_t j = 0; j < seq_len; j++) {
                size_t dj = doc_of(j, doc_boundaries, num_boundaries);
                float v = data[i * seq_len + j];
                bool same_doc = (di == dj);
                bool causal_ok = (j <= i);
                bool window_ok = (window == 0) || (i >= j && (i - j) < window);
                bool expect_allowed = same_doc && causal_ok && window_ok;

                if (!same_doc) {
                    TEST_ASSERT(isinf(v) && v < 0.0f);
                } else if (expect_allowed) {
                    TEST_ASSERT(v == 0.0f);
                } else {
                    TEST_ASSERT(isinf(v) && v < 0.0f);
                }
            }
        }
    }

    blt_arena_destroy(arena);
    return 1;
}



//------------------------------------------------------------------------------
// Overfit test

static unsigned next_rand(unsigned* seed) {
    *seed = *seed * 1103515245u + 12345u;
    return (*seed / 65536u) % 32768u;
}

static void fill_small_uniform(blt_tensor* t, unsigned* seed) {
    float* d = (float*)t->data;
    for (size_t i = 0; i < t->numel; i++) {
        d[i] = ((float)(next_rand(seed) % 2000) / 1000.0f - 1.0f) * 0.05f;
    }
}

int run_lencoder_overfit_test(void) {
    unsigned seed = 1234;
    // arena for model and gradients
    blt_arena* persistent_arena = blt_arena_create(16 * 1024 * 1024, BLT_BACKEND_CPU);
    // arena for intermediate tensors
    blt_arena* compute_arena = blt_arena_create(16 * 1024 * 1024, BLT_BACKEND_CPU);
    
    if (!persistent_arena || !compute_arena) return 0;

    const size_t embed_dim = 16;
    uint8_t raw_bytes[12] = {'f','a','s','t','_','b','l','t','!','?','_','x'};
    size_t seq_len = 12;

    blt_tensor bytes_in = {0};
    size_t shape1[1] = {seq_len};
    bytes_in = blt_tensor_create(persistent_arena, shape1, 1, BLT_DTYPE_UINT8);
    memcpy(bytes_in.data, raw_bytes, seq_len);

    blt_patch_info patches[3] = {
        {0, 4, 0.0f}, {4, 4, 0.0f}, {8, 4, 0.0f}
    };
    size_t num_patches = 3;

    blt_local_encoder_config cfg = {0};
    cfg.embed_dim = embed_dim;
    cfg.num_layers = 2;
    cfg.hidden_dim = 32;
    cfg.num_heads = 4;
    cfg.cross_attn_heads = 4;
    cfg.local_window = 4;
    cfg.cross_attn_all_layers = true;
    cfg.pool_type = BLT_POOL_MEAN;
    cfg.rope_theta = 500000.0f;
    cfg.max_seq_len = 64;
    cfg.ngram_config.ngram_sizes[0] = 3;
    cfg.ngram_config.num_ngram_sizes = 1;
    cfg.ngram_config.per_ngram_vocab = 256;
    cfg.ngram_config.hash_prime = 1000000007ULL;
    cfg.ngram_config.normalize = true;
    cfg.ngram_config.embed_dim = embed_dim;

    blt_local_encoder* model = blt_local_encoder_create(persistent_arena, &cfg);
    TEST_ASSERT(model != NULL);

    fill_small_uniform(&model->byte_embedding_weight, &seed);
    for (size_t t = 0; t < model->ngram_weights.num_tables; t++) {
        fill_small_uniform(&model->ngram_weights.tables[t], &seed);
    }
    for (size_t l = 0; l < cfg.num_layers; l++) {
        blt_local_encoder_layer_storage* ls = &model->layers[l];
        fill_small_uniform(&ls->norm1_weight, &seed);
        fill_small_uniform(&ls->attn_qkv_w, &seed);
        fill_small_uniform(&ls->attn_proj_w, &seed);
        fill_small_uniform(&ls->norm2_weight, &seed);
        fill_small_uniform(&ls->ffn_up_w, &seed);
        fill_small_uniform(&ls->ffn_gate_w, &seed);
        fill_small_uniform(&ls->ffn_down_w, &seed);
        fill_small_uniform(&ls->cross_norm_weight, &seed);
        fill_small_uniform(&ls->cross_weight_q, &seed);
        fill_small_uniform(&ls->cross_weight_k, &seed);
        fill_small_uniform(&ls->cross_weight_v, &seed);
        fill_small_uniform(&ls->cross_weight_proj, &seed);
    }

    // byte_hidden_out [seq_len, embed_dim] -> pred [seq_len, 1]
    blt_tensor head_w = {0};
    size_t head_shape[2] = {embed_dim, 1};
    head_w = blt_tensor_create(persistent_arena, head_shape, 2, BLT_DTYPE_FP32);
    fill_small_uniform(&head_w, &seed);

    blt_tensor target = {0};
    size_t target_shape[2] = {seq_len, 1};
    target = blt_tensor_create(persistent_arena, target_shape, 2, BLT_DTYPE_FP32);
    float* tgt_d = (float*)target.data;
    for (size_t i = 0; i < seq_len; i++) tgt_d[i] = ((float)(i % 3)) - 1.0f; /* synthetic */

    blt_local_encoder_grad* grad = blt_local_encoder_grad_create(persistent_arena, model);

    size_t patch_shape[2] = {num_patches, embed_dim};
    size_t hidden_shape[2] = {seq_len, embed_dim};
    size_t pred_shape[2] = {seq_len, 1};

    float first_loss = -1.0f, last_loss = -1.0f;
    const int num_steps = 30;
    const float lr = 0.05f;

    for (int step = 0; step < num_steps; step++) {
        blt_arena_reset(compute_arena);

        blt_tensor patch_out = blt_tensor_create(compute_arena, patch_shape, 2, BLT_DTYPE_FP32);
        blt_tensor hidden_out = blt_tensor_create(compute_arena, hidden_shape, 2, BLT_DTYPE_FP32);

        blt_local_encoder_forward(model, &bytes_in, patches, num_patches,
                                   NULL, 0, &patch_out, &hidden_out, compute_arena);

        blt_tensor pred = blt_tensor_create(compute_arena, pred_shape, 2, BLT_DTYPE_FP32);
        blt_matmul(&hidden_out, &head_w, &pred);

        float* pred_d = (float*)pred.data;
        float loss = 0.0f;
        blt_tensor grad_pred = blt_tensor_create(compute_arena, pred_shape, 2, BLT_DTYPE_FP32);
        float* gpred_d = (float*)grad_pred.data;
        
        for (size_t i = 0; i < seq_len; i++) {
            float diff = pred_d[i] - tgt_d[i];
            loss += diff * diff;
            gpred_d[i] = 2.0f * diff / (float)seq_len;
        }
        loss /= (float)seq_len;
        if (step == 0) first_loss = loss;
        last_loss = loss;

        blt_tensor grad_hidden = blt_tensor_create(compute_arena, hidden_shape, 2, BLT_DTYPE_FP32);
        blt_tensor grad_head_w = blt_tensor_create(compute_arena, head_shape, 2, BLT_DTYPE_FP32);
        blt_matmul_backward(&hidden_out, &head_w, &grad_pred, &grad_hidden, &grad_head_w);

        blt_tensor grad_patch_out = blt_tensor_create(compute_arena, patch_shape, 2, BLT_DTYPE_FP32);

        blt_local_encoder_backward(model, &bytes_in, patches, num_patches, NULL, 0,
                                    &grad_patch_out, &grad_hidden, grad, compute_arena);

        blt_sgd_step(&head_w, &grad_head_w, lr);
        blt_sgd_step(&model->byte_embedding_weight, &grad->embedding_grad, lr);
        for (size_t t = 0; t < model->ngram_weights.num_tables; t++) {
            blt_sgd_step(&model->ngram_weights.tables[t], &grad->ngram_grads.tables[t], lr);
        }
        for (size_t l = 0; l < cfg.num_layers; l++) {
            blt_local_encoder_layer_storage* ls = &model->layers[l];
            blt_local_encoder_layer_grad* lg = &grad->layer_grads[l];
            blt_sgd_step(&ls->norm1_weight, &lg->norm1_weight, lr);
            blt_sgd_step(&ls->attn_qkv_w, &lg->attn_qkv_w, lr);
            blt_sgd_step(&ls->attn_proj_w, &lg->attn_proj_w, lr);
            blt_sgd_step(&ls->norm2_weight, &lg->norm2_weight, lr);
            blt_sgd_step(&ls->ffn_up_w, &lg->ffn_up_w, lr);
            blt_sgd_step(&ls->ffn_gate_w, &lg->ffn_gate_w, lr);
            blt_sgd_step(&ls->ffn_down_w, &lg->ffn_down_w, lr);
            blt_sgd_step(&ls->cross_norm_weight, &lg->cross_norm_weight, lr);
            blt_sgd_step(&ls->cross_weight_q, &lg->cross_weight_q, lr);
            blt_sgd_step(&ls->cross_weight_k, &lg->cross_weight_k, lr);
            blt_sgd_step(&ls->cross_weight_v, &lg->cross_weight_v, lr);
            blt_sgd_step(&ls->cross_weight_proj, &lg->cross_weight_proj, lr);
        }
    }

    TEST_ASSERT(last_loss < first_loss);

    blt_arena_destroy(persistent_arena);
    blt_arena_destroy(compute_arena);
    return 1;
}



//----------------------------------------------------------------------------
// Smoke test

static void fill_ones(blt_tensor* t) {
    float* d = (float*)t->data;
    for (size_t i = 0; i < t->numel; i++) d[i] = 1.0f;
}

// confirms forward+backward run without crashing
int run_local_encoder_smoke_test(void) {
    const size_t num_layers_opts[] = {1, 3};
    const bool cross_attn_opts[] = {true, false};

    for (size_t li = 0; li < 2; li++) {
        for (size_t ci = 0; ci < 2; ci++) {
            blt_arena* arena = blt_arena_create(8 * 1024 * 1024, BLT_BACKEND_CPU);
            if (!arena) return 0;

            const size_t embed_dim = 16;
            uint8_t raw_bytes[16];
            for (int i = 0; i < 16; i++) raw_bytes[i] = (uint8_t)('a' + (i % 20));
            size_t seq_len = 16;

            blt_tensor bytes_in = {0};
            size_t shape1[1] = {seq_len};
            bytes_in = blt_tensor_create(arena, shape1, 1, BLT_DTYPE_UINT8);
            memcpy(bytes_in.data, raw_bytes, seq_len);

            blt_patch_info patches[4] = {
                {0, 4, 0.0f}, {4, 4, 0.0f}, {8, 4, 0.0f}, {12, 4, 0.0f}
            };
            size_t num_patches = 4;

            blt_local_encoder_config cfg = {0};
            cfg.embed_dim = embed_dim;
            cfg.num_layers = num_layers_opts[li];
            cfg.hidden_dim = 32;
            cfg.num_heads = 4;
            cfg.cross_attn_heads = 4;
            cfg.local_window = 4;
            cfg.cross_attn_all_layers = cross_attn_opts[ci];
            cfg.pool_type = BLT_POOL_MEAN;
            cfg.rope_theta = 500000.0f;
            cfg.max_seq_len = 64;
            cfg.ngram_config.ngram_sizes[0] = 3;
            cfg.ngram_config.ngram_sizes[1] = 4;
            cfg.ngram_config.num_ngram_sizes = 2;
            cfg.ngram_config.per_ngram_vocab = 256;
            cfg.ngram_config.hash_prime = 1000000007ULL;
            cfg.ngram_config.normalize = true;
            cfg.ngram_config.embed_dim = embed_dim;

            blt_local_encoder* model = blt_local_encoder_create(arena, &cfg);
            TEST_ASSERT(model != NULL);

            fill_ones(&model->byte_embedding_weight);
            for (size_t t = 0; t < model->ngram_weights.num_tables; t++) {
                fill_ones(&model->ngram_weights.tables[t]);
            }
            for (size_t l = 0; l < cfg.num_layers; l++) {
                blt_local_encoder_layer_storage* ls = &model->layers[l];
                fill_ones(&ls->norm1_weight);
                fill_ones(&ls->attn_qkv_w);
                fill_ones(&ls->attn_proj_w);
                fill_ones(&ls->norm2_weight);
                fill_ones(&ls->ffn_up_w);
                fill_ones(&ls->ffn_gate_w);
                fill_ones(&ls->ffn_down_w);
                fill_ones(&ls->cross_norm_weight);
                fill_ones(&ls->cross_weight_q);
                fill_ones(&ls->cross_weight_k);
                fill_ones(&ls->cross_weight_v);
                fill_ones(&ls->cross_weight_proj);
            }

            size_t patch_shape[2] = {num_patches, embed_dim};
            size_t hidden_shape[2] = {seq_len, embed_dim};
            blt_tensor patch_out = blt_tensor_create(arena, patch_shape, 2, BLT_DTYPE_FP32);
            blt_tensor hidden_out = blt_tensor_create(arena, hidden_shape, 2, BLT_DTYPE_FP32);

            blt_local_encoder_forward(model, &bytes_in, patches, num_patches,
                                       NULL, 0, &patch_out, &hidden_out, arena);

            TEST_ASSERT(patch_out.shape[0] == num_patches);
            TEST_ASSERT(patch_out.shape[1] == embed_dim);
            TEST_ASSERT(hidden_out.shape[0] == seq_len);
            TEST_ASSERT(hidden_out.shape[1] == embed_dim);

            blt_local_encoder_grad* grad = blt_local_encoder_grad_create(arena, model);

            blt_tensor grad_patch_out = blt_tensor_create(arena, patch_shape, 2, BLT_DTYPE_FP32);
            blt_tensor grad_hidden_out = blt_tensor_create(arena, hidden_shape, 2, BLT_DTYPE_FP32);
            fill_ones(&grad_patch_out);
            fill_ones(&grad_hidden_out);

            blt_local_encoder_backward(model, &bytes_in, patches, num_patches, NULL, 0,
                                        &grad_patch_out, &grad_hidden_out, grad, arena);

            blt_arena_destroy(arena);
        }
    }

    return 1;
}