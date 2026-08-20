#include "test_suite.h"
#include "test_helpers.h"
#include "blt/core/allocator.h"
#include "blt/models/patcher.h"
#include "blt/ops/mask_builder.h"
#include "blt/models/global_transformer.h"
#include "blt/ops/matmul.h"
#include "blt/ops/cross_entropy.h"
#include "blt/ops/optim.h"

#include <stdlib.h>
#include <math.h>
#include <time.h>



//-------------------------------------------------------------------------------
// Document boundary mapping test
//
// Test that the global transformer can correctly map document boundaries


// Converts byte-indexed document boundaries into patch-indexed ones by
// requiring an exact match against some patch start_idx
static int map_byte_boundaries_to_patch_boundaries(
    const size_t* byte_doc_boundaries, size_t num_docs,
    const blt_patch_info* patches, size_t num_patches,
    size_t* patch_doc_boundaries_out
) {
    for (size_t d = 0; d < num_docs; d++) {
        size_t byte_idx = byte_doc_boundaries[d];
        bool found = false;
        for (size_t p = 0; p < num_patches; p++) {
            if (patches[p].start_idx == byte_idx) {
                patch_doc_boundaries_out[d] = p;
                found = true;
                break;
            }
        }
        if (!found) {
            return 0;   // byte boundary landed mid-patch -- invariant violated
        }
    }
    return 1;
}



int run_global_transformer_doc_boundary(void) {
    blt_patch_info patches[6];
    size_t starts[]  = {0, 5, 9, 14, 20, 25};
    size_t lengths[] = {5, 4, 5, 6, 5, 3};
    for (size_t i = 0; i < 6; i++) {
        patches[i].start_idx = starts[i];
        patches[i].length = lengths[i];
        patches[i].peak_entropy = 0.0f;
    }
    size_t num_patches = 6;

    // Doc 0 starts implicitly at byte 0 (patch 0) -- only the starts of
    // doc 1 and doc 2 are listed, matching blt_build_attention_mask's
    // convention.
    size_t byte_doc_boundaries[] = { 9, 20 };   // = patch starts of patches 2, 4
    size_t num_boundaries = 2;
    size_t patch_doc_boundaries[2] = {0};

    TEST_ASSERT(map_byte_boundaries_to_patch_boundaries(
        byte_doc_boundaries, num_boundaries, patches, num_patches, patch_doc_boundaries));

    TEST_ASSERT(patch_doc_boundaries[0] == 2);
    TEST_ASSERT(patch_doc_boundaries[1] == 4);

    // Confirm the invariant would actually catch a violation: shift one
    // boundary into the middle of patch 2 (bytes [9, 14)).
    size_t bad_boundaries[] = { 10, 20 };
    size_t bad_out[2] = {0};
    TEST_ASSERT(map_byte_boundaries_to_patch_boundaries(
        bad_boundaries, num_boundaries, patches, num_patches, bad_out) == 0);

    return 1;
}



//-------------------------------------------------------------------------------
// Causal masking fuzz test#
//
// Tests that the global transformer can correctly apply causal masking with
// document boundaries. This is a fuzz test that generates random numbers of
// patches and random document boundaries, then checks that the resulting
// attention mask is correct.

static size_t make_doc_boundaries(size_t num_patches, size_t* out, size_t max_docs) {
    if (num_patches <= 1) return 0;

    size_t target_docs = 1 + (size_t)(rand() % max_docs);
    if (target_docs > num_patches) target_docs = num_patches;
    size_t target_boundaries = target_docs - 1;

    size_t placed = 0;
    size_t last = 0;

    while (placed < target_boundaries) {
        size_t remaining = target_boundaries - placed;
        size_t max_candidate = num_patches - remaining;
        size_t min_candidate = last + 1;
        if (min_candidate > max_candidate) break;

        size_t span = max_candidate - min_candidate + 1;
        size_t candidate = min_candidate + (rand() % span);
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



int run_global_transformer_causal_mask(void) {
    srand(1234);
    const int num_trials = 50;

    for (int trial = 0; trial < num_trials; trial++) {
        size_t num_patches = 2 + (rand() % 30);   // [2, 31]

        size_t doc_boundaries[16];
        size_t num_boundaries = make_doc_boundaries(num_patches, doc_boundaries, 6);

        blt_arena* arena = blt_arena_create(1024 * 1024, BLT_BACKEND_CPU);
        if (!arena) return 0;

        blt_mask_config cfg = {0};
        cfg.seq_len_q = num_patches;
        cfg.seq_len_kv = num_patches;
        cfg.sliding_window = 0;
        cfg.doc_boundaries = doc_boundaries;
        cfg.num_docs = num_boundaries + 1;
        cfg.is_causal = true;

        blt_tensor mask = {0};
        blt_build_attention_mask(&cfg, &mask, arena);
        TEST_ASSERT(mask.ndim == 2 && mask.shape[0] == num_patches && mask.shape[1] == num_patches);

        const float* data = (const float*)mask.data;
        for (size_t i = 0; i < num_patches; i++) {
            size_t doc_i = doc_of(i, doc_boundaries, num_boundaries);
            for (size_t j = 0; j < num_patches; j++) {
                size_t doc_j = doc_of(j, doc_boundaries, num_boundaries);
                float v = data[i * num_patches + j];
                bool should_allow = (j <= i) && (doc_i == doc_j);
                if (should_allow) {
                    TEST_ASSERT(v > -INFINITY);
                } else {
                    TEST_ASSERT(v <= -INFINITY);
                }
            }
        }

        blt_arena_destroy(arena);
    }

    return 1;
}



//-------------------------------------------------------------------------------
// Global transformer overfit test
//
// Tests that the global transformer can overfit a single batch of random
// data. This is a smoke test to ensure that the forward and backward passes
// are wired up correctly and that the model can learn.


static void fill_random(blt_tensor* t, float scale) {
    float* d = (float*)t->data;
    for (size_t i = 0; i < t->numel; i++) {
        d[i] = scale * (((float)(rand() % 2000) / 1000.0f) - 1.0f);
    }
}



static void sgd_update_global_transformer(blt_global_transformer* model, const blt_global_transformer_grad* grad, float lr) {
    for (size_t l = 0; l < model->stack.num_layers; l++) {
        blt_transformer_layer_storage* s = &model->stack.layer_storage[l];
        const blt_transformer_layer_grad* g = &grad->stack_grad->layer_grads[l];

        blt_sgd_step(&s->norm1_weight, &g->norm1_weight, lr);
        blt_sgd_step(&s->attn_qkv_w, &g->attn_qkv_w, lr);
        blt_sgd_step(&s->attn_proj_w, &g->attn_proj_w, lr);
        blt_sgd_step(&s->norm2_weight, &g->norm2_weight, lr);
        blt_sgd_step(&s->ffn_up_w, &g->ffn_up_w, lr);
        blt_sgd_step(&s->ffn_gate_w, &g->ffn_gate_w, lr);
        blt_sgd_step(&s->ffn_down_w, &g->ffn_down_w, lr);
    }
}



int run_global_transformer_overfit(void) {
    blt_arena* arena = blt_arena_create(16 * 1024 * 1024, BLT_BACKEND_CPU);
    if (!arena) return 0;

    srand(7);

    size_t num_patches = 6;
    size_t probe_dim = 8;

    blt_global_transformer_config cfg = {0};
    cfg.num_layers = 2;
    cfg.embed_dim = 16;
    cfg.hidden_dim = 32;
    cfg.num_heads = 4;
    cfg.max_seq_len = 32;
    cfg.rope_theta = 500000.0f;

    blt_global_transformer* model = blt_global_transformer_create(arena, &cfg);
    TEST_ASSERT(model != NULL);

    for (size_t l = 0; l < cfg.num_layers; l++) {
        blt_transformer_layer_storage* s = &model->stack.layer_storage[l];
        fill_random(&s->norm1_weight, 1.0f);
        fill_random(&s->attn_qkv_w, 0.1f);
        fill_random(&s->attn_proj_w, 0.1f);
        fill_random(&s->norm2_weight, 1.0f);
        fill_random(&s->ffn_up_w, 0.1f);
        fill_random(&s->ffn_gate_w, 0.1f);
        fill_random(&s->ffn_down_w, 0.1f);
    }

    blt_global_transformer_grad* grad = blt_global_transformer_grad_create(arena, model);
    TEST_ASSERT(grad != NULL);

    size_t patch_shape[2] = { num_patches, cfg.embed_dim };
    blt_tensor patch_in = blt_tensor_create(arena, patch_shape, 2, BLT_DTYPE_FP32);
    fill_random(&patch_in, 1.0f);

    size_t probe_shape[2] = { cfg.embed_dim, probe_dim };
    blt_tensor probe_w = blt_tensor_create(arena, probe_shape, 2, BLT_DTYPE_FP32);
    fill_random(&probe_w, 0.1f);
    blt_tensor probe_grad = blt_tensor_create(arena, probe_shape, 2, BLT_DTYPE_FP32);

    size_t target_shape[1] = { num_patches };
    blt_tensor targets = blt_tensor_create(arena, target_shape, 1, BLT_DTYPE_UINT8);
    for (size_t i = 0; i < num_patches; i++) {
        ((uint8_t*)targets.data)[i] = (uint8_t)(rand() % probe_dim);
    }

    size_t doc_boundaries[] = { 0 };
    size_t num_docs = 1;

    float first_loss = 0.0f, last_loss = 0.0f;
    const int num_steps = 200;
    const float lr = 0.05f;

    for (int step = 0; step < num_steps; step++) {
        blt_tensor patch_out = blt_tensor_create(arena, patch_shape, 2, BLT_DTYPE_FP32);
        blt_global_transformer_forward(model, &patch_in, doc_boundaries, num_docs, &patch_out, arena);

        size_t logits_shape[2] = { num_patches, probe_dim };
        blt_tensor logits = blt_tensor_create(arena, logits_shape, 2, BLT_DTYPE_FP32);
        blt_matmul(&patch_out, &probe_w, &logits);

        size_t loss_shape[1] = { 1 };
        blt_tensor loss = blt_tensor_create(arena, loss_shape, 1, BLT_DTYPE_FP32);
        blt_cross_entropy_forward(&logits, &targets, &loss);

        float loss_val = ((float*)loss.data)[0];
        if (step == 0) first_loss = loss_val;
        last_loss = loss_val;

        blt_tensor grad_logits = blt_tensor_create(arena, logits_shape, 2, BLT_DTYPE_FP32);
        blt_cross_entropy_backward(&logits, &targets, &grad_logits);

        blt_tensor grad_patch_out = blt_tensor_create(arena, patch_shape, 2, BLT_DTYPE_FP32);
        blt_matmul_backward(&patch_out, &probe_w, &grad_logits, &grad_patch_out, &probe_grad);

        blt_tensor grad_patch_in = blt_tensor_create(arena, patch_shape, 2, BLT_DTYPE_FP32);
        blt_global_transformer_backward(model, &patch_in, doc_boundaries, num_docs,
                                         &grad_patch_out, &grad_patch_in, grad, arena);

        sgd_update_global_transformer(model, grad, lr);
        blt_sgd_step(&probe_w, &probe_grad, lr);
    }

    TEST_ASSERT(last_loss < first_loss * 0.5f);

    blt_arena_destroy(arena);
    return 1;
}