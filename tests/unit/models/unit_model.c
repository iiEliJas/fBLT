#include <ctype.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "test_helpers.h"
#include "test_suite.h"

#include "core/allocator.h"
#include "ops/elementwise.h"
#include "ops/optim.h"
#include "models/local_encoder.h"
#include "models/global_transformer.h"
#include "models/local_decoder.h"
#include "models/transformer_stack.h"
#include "models/entropy_lm.h"
#include "models/patcher.h"
#include "models/model.h"
#include "core/generate_greedy.h"

//----------------------------------------------------------------------
// Helpers

static void fill_small_uniform(blt_tensor *t, float scale) {
    float *data = (float *)t->data;
    for (size_t i = 0; i < t->numel; i++) {
        float r = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
        data[i] = r * scale;
    }
}

static void fill_constant(blt_tensor *t, float value) {
    float *data = (float *)t->data;
    for (size_t i = 0; i < t->numel; i++) {
        data[i] = value;
    }
}

static void make_small_model_config(blt_model_config *config) {
    const size_t embed_dim = 16;
    const size_t hidden_dim = 32;
    const size_t max_seq_len = 64;

    memset(config, 0, sizeof(*config));

    config->encoder_config.embed_dim = embed_dim;
    config->encoder_config.patch_dim = 0;
    config->encoder_config.num_layers = 1;
    config->encoder_config.hidden_dim = hidden_dim;
    config->encoder_config.num_heads = 2;
    config->encoder_config.cross_attn_heads = 2;
    config->encoder_config.local_window = 0;
    config->encoder_config.cross_attn_all_layers = true;
    config->encoder_config.pool_type = BLT_POOL_MEAN;
    config->encoder_config.rope_theta = 10000.0f;
    config->encoder_config.max_seq_len = max_seq_len;
    config->encoder_config.ngram_config.ngram_sizes[0] = 3;
    config->encoder_config.ngram_config.ngram_sizes[1] = 4;
    config->encoder_config.ngram_config.num_ngram_sizes = 2;
    config->encoder_config.ngram_config.per_ngram_vocab = 64;
    config->encoder_config.ngram_config.hash_prime = 1000000007ULL;
    config->encoder_config.ngram_config.normalize = true;
    config->encoder_config.ngram_config.embed_dim = embed_dim;

    config->global_config.embed_dim = embed_dim;
    config->global_config.num_layers = 1;
    config->global_config.hidden_dim = hidden_dim;
    config->global_config.num_heads = 2;
    config->global_config.rope_theta = 10000.0f;
    config->global_config.max_seq_len = max_seq_len;

    config->decoder_config.embed_dim = embed_dim;
    config->decoder_config.patch_dim = 0;
    config->decoder_config.num_layers = 1;
    config->decoder_config.hidden_dim = hidden_dim;
    config->decoder_config.num_heads = 2;
    config->decoder_config.cross_attn_heads = 2;
    config->decoder_config.local_window = 0;
    config->decoder_config.cross_attn_all_layers = true;
    config->decoder_config.rope_theta = 10000.0f;
    config->decoder_config.max_seq_len = max_seq_len;
    config->decoder_config.vocab_size = 256;
}

static void init_transformer_layer(blt_transformer_layer_storage *l, float scale) {
    fill_constant(&l->norm1_weight, 1.0f);
    fill_small_uniform(&l->attn_qkv_w, scale);
    fill_small_uniform(&l->attn_proj_w, scale);
    fill_constant(&l->norm2_weight, 1.0f);
    fill_small_uniform(&l->ffn_up_w, scale);
    fill_small_uniform(&l->ffn_gate_w, scale);
    fill_small_uniform(&l->ffn_down_w, scale);
}

static void init_encoder_layer(blt_local_encoder_layer_storage *l, float scale) {
    fill_constant(&l->norm1_weight, 1.0f);
    fill_small_uniform(&l->attn_qkv_w, scale);
    fill_small_uniform(&l->attn_proj_w, scale);
    fill_constant(&l->norm2_weight, 1.0f);
    fill_small_uniform(&l->ffn_up_w, scale);
    fill_small_uniform(&l->ffn_gate_w, scale);
    fill_small_uniform(&l->ffn_down_w, scale);
    fill_constant(&l->cross_norm_weight, 1.0f);
    fill_small_uniform(&l->cross_weight_q, scale);
    fill_small_uniform(&l->cross_weight_k, scale);
    fill_small_uniform(&l->cross_weight_v, scale);
    fill_small_uniform(&l->cross_weight_proj, scale);
}

static void init_decoder_layer(blt_local_decoder_layer_storage *l, float scale) {
    fill_constant(&l->cross_norm_weight, 1.0f);
    fill_small_uniform(&l->cross_weight_q, scale);
    fill_small_uniform(&l->cross_weight_k, scale);
    fill_small_uniform(&l->cross_weight_v, scale);
    fill_small_uniform(&l->cross_weight_proj, scale);
    fill_constant(&l->norm1_weight, 1.0f);
    fill_small_uniform(&l->attn_qkv_w, scale);
    fill_small_uniform(&l->attn_proj_w, scale);
    fill_constant(&l->norm2_weight, 1.0f);
    fill_small_uniform(&l->ffn_up_w, scale);
    fill_small_uniform(&l->ffn_gate_w, scale);
    fill_small_uniform(&l->ffn_down_w, scale);
}

static void random_init_model(blt_model *model, float scale) {
    // byte_embedding_weight: small random values, uniform in [-scale, scale]
    fill_small_uniform(&model->encoder->byte_embedding_weight, scale);
    for (size_t i = 0; i < model->encoder->config.num_layers; i++) {
        init_encoder_layer(&model->encoder->layers[i], scale);
    }

    for (size_t i = 0; i < model->global->stack.num_layers; i++) {
        init_transformer_layer(&model->global->stack.layer_storage[i], scale);
    }

    for (size_t i = 0; i < model->decoder->config.num_layers; i++) {
        init_decoder_layer(&model->decoder->layers[i], scale);
    }
    fill_small_uniform(&model->decoder->lm_head_weight, scale);
}

static void random_init_entropy_lm(blt_entropy_lm *lm, float scale) {
    fill_small_uniform(&lm->embedding_weight, scale);
    for (size_t i = 0; i < lm->stack.num_layers; i++) {
        init_transformer_layer(&lm->stack.layer_storage[i], scale);
    }
    fill_small_uniform(&lm->lm_head_weight, scale);
}

// Global-norm gradient clipping, applied just before the SGD step.
// Plain unclipped SGD on a freshly-initialized toy model reliably
// diverges to inf/nan within a few dozen steps at any usable lr, since
// nothing here normalizes logit magnitude.

static float tensor_sq_norm(const blt_tensor *t) {
    const float *data = (const float *)t->data;
    float sum = 0.0f;
    for (size_t i = 0; i < t->numel; i++) {
        sum += data[i] * data[i];
    }
    return sum;
}

static float transformer_layer_grad_sq_norm(const blt_transformer_layer_grad *g) {
    return tensor_sq_norm(&g->norm1_weight) + tensor_sq_norm(&g->attn_qkv_w) + tensor_sq_norm(&g->attn_proj_w) +
           tensor_sq_norm(&g->norm2_weight) + tensor_sq_norm(&g->ffn_up_w) + tensor_sq_norm(&g->ffn_gate_w) +
           tensor_sq_norm(&g->ffn_down_w);
}

static float encoder_layer_grad_sq_norm(const blt_local_encoder_layer_grad *g) {
    return tensor_sq_norm(&g->norm1_weight) + tensor_sq_norm(&g->attn_qkv_w) + tensor_sq_norm(&g->attn_proj_w) +
           tensor_sq_norm(&g->norm2_weight) + tensor_sq_norm(&g->ffn_up_w) + tensor_sq_norm(&g->ffn_gate_w) +
           tensor_sq_norm(&g->ffn_down_w) + tensor_sq_norm(&g->cross_norm_weight) + tensor_sq_norm(&g->cross_weight_q) +
           tensor_sq_norm(&g->cross_weight_k) + tensor_sq_norm(&g->cross_weight_v) +
           tensor_sq_norm(&g->cross_weight_proj);
}

static float decoder_layer_grad_sq_norm(const blt_local_decoder_layer_grad *g) {
    return tensor_sq_norm(&g->cross_norm_weight) + tensor_sq_norm(&g->cross_weight_q) +
           tensor_sq_norm(&g->cross_weight_k) + tensor_sq_norm(&g->cross_weight_v) +
           tensor_sq_norm(&g->cross_weight_proj) + tensor_sq_norm(&g->norm1_weight) + tensor_sq_norm(&g->attn_qkv_w) +
           tensor_sq_norm(&g->attn_proj_w) + tensor_sq_norm(&g->norm2_weight) + tensor_sq_norm(&g->ffn_up_w) +
           tensor_sq_norm(&g->ffn_gate_w) + tensor_sq_norm(&g->ffn_down_w);
}

static float compute_model_grad_global_norm(blt_model_grad *grad, const blt_model *model) {
    float total = 0.0f;

    total += tensor_sq_norm(&grad->encoder_grad->embedding_grad);
    for (size_t i = 0; i < model->encoder->ngram_weights.num_tables; i++) {
        total += tensor_sq_norm(&grad->encoder_grad->ngram_grads.tables[i]);
    }
    for (size_t i = 0; i < model->encoder->config.num_layers; i++) {
        total += encoder_layer_grad_sq_norm(&grad->encoder_grad->layer_grads[i]);
    }
    for (size_t i = 0; i < model->global->stack.num_layers; i++) {
        total += transformer_layer_grad_sq_norm(&grad->global_grad->stack_grad->layer_grads[i]);
    }
    for (size_t i = 0; i < model->decoder->config.num_layers; i++) {
        total += decoder_layer_grad_sq_norm(&grad->decoder_grad->layer_grads[i]);
    }
    total += tensor_sq_norm(&grad->decoder_grad->lm_head_grad);

    return sqrtf(total);
}

static void scale_transformer_layer_grad(blt_transformer_layer_grad *g, float scale) {
    blt_scale(&g->norm1_weight, scale);
    blt_scale(&g->attn_qkv_w, scale);
    blt_scale(&g->attn_proj_w, scale);
    blt_scale(&g->norm2_weight, scale);
    blt_scale(&g->ffn_up_w, scale);
    blt_scale(&g->ffn_gate_w, scale);
    blt_scale(&g->ffn_down_w, scale);
}

static void scale_encoder_layer_grad(blt_local_encoder_layer_grad *g, float scale) {
    blt_scale(&g->norm1_weight, scale);
    blt_scale(&g->attn_qkv_w, scale);
    blt_scale(&g->attn_proj_w, scale);
    blt_scale(&g->norm2_weight, scale);
    blt_scale(&g->ffn_up_w, scale);
    blt_scale(&g->ffn_gate_w, scale);
    blt_scale(&g->ffn_down_w, scale);
    blt_scale(&g->cross_norm_weight, scale);
    blt_scale(&g->cross_weight_q, scale);
    blt_scale(&g->cross_weight_k, scale);
    blt_scale(&g->cross_weight_v, scale);
    blt_scale(&g->cross_weight_proj, scale);
}

static void scale_decoder_layer_grad(blt_local_decoder_layer_grad *g, float scale) {
    blt_scale(&g->cross_norm_weight, scale);
    blt_scale(&g->cross_weight_q, scale);
    blt_scale(&g->cross_weight_k, scale);
    blt_scale(&g->cross_weight_v, scale);
    blt_scale(&g->cross_weight_proj, scale);
    blt_scale(&g->norm1_weight, scale);
    blt_scale(&g->attn_qkv_w, scale);
    blt_scale(&g->attn_proj_w, scale);
    blt_scale(&g->norm2_weight, scale);
    blt_scale(&g->ffn_up_w, scale);
    blt_scale(&g->ffn_gate_w, scale);
    blt_scale(&g->ffn_down_w, scale);
}

// Scales every gradient tensor in place so the global L2 norm across all
// of them is <= max_norm. No-op if already within bounds.
static void clip_model_grad_global_norm(blt_model_grad *grad, const blt_model *model, float max_norm) {
    float norm = compute_model_grad_global_norm(grad, model);
    if (norm <= max_norm || norm == 0.0f) {
        return;
    }
    float scale = max_norm / norm;

    blt_scale(&grad->encoder_grad->embedding_grad, scale);
    for (size_t i = 0; i < model->encoder->ngram_weights.num_tables; i++) {
        blt_scale(&grad->encoder_grad->ngram_grads.tables[i], scale);
    }
    for (size_t i = 0; i < model->encoder->config.num_layers; i++) {
        scale_encoder_layer_grad(&grad->encoder_grad->layer_grads[i], scale);
    }
    for (size_t i = 0; i < model->global->stack.num_layers; i++) {
        scale_transformer_layer_grad(&grad->global_grad->stack_grad->layer_grads[i], scale);
    }
    for (size_t i = 0; i < model->decoder->config.num_layers; i++) {
        scale_decoder_layer_grad(&grad->decoder_grad->layer_grads[i], scale);
    }
    blt_scale(&grad->decoder_grad->lm_head_grad, scale);
}

static void sgd_step_transformer_layer(blt_transformer_layer_storage *w, blt_transformer_layer_grad *g, float lr) {
    blt_sgd_step(&w->norm1_weight, &g->norm1_weight, lr);
    blt_sgd_step(&w->attn_qkv_w, &g->attn_qkv_w, lr);
    blt_sgd_step(&w->attn_proj_w, &g->attn_proj_w, lr);
    blt_sgd_step(&w->norm2_weight, &g->norm2_weight, lr);
    blt_sgd_step(&w->ffn_up_w, &g->ffn_up_w, lr);
    blt_sgd_step(&w->ffn_gate_w, &g->ffn_gate_w, lr);
    blt_sgd_step(&w->ffn_down_w, &g->ffn_down_w, lr);
}

static void sgd_step_encoder_layer(blt_local_encoder_layer_storage *w, blt_local_encoder_layer_grad *g, float lr) {
    blt_sgd_step(&w->norm1_weight, &g->norm1_weight, lr);
    blt_sgd_step(&w->attn_qkv_w, &g->attn_qkv_w, lr);
    blt_sgd_step(&w->attn_proj_w, &g->attn_proj_w, lr);
    blt_sgd_step(&w->norm2_weight, &g->norm2_weight, lr);
    blt_sgd_step(&w->ffn_up_w, &g->ffn_up_w, lr);
    blt_sgd_step(&w->ffn_gate_w, &g->ffn_gate_w, lr);
    blt_sgd_step(&w->ffn_down_w, &g->ffn_down_w, lr);
    blt_sgd_step(&w->cross_norm_weight, &g->cross_norm_weight, lr);
    blt_sgd_step(&w->cross_weight_q, &g->cross_weight_q, lr);
    blt_sgd_step(&w->cross_weight_k, &g->cross_weight_k, lr);
    blt_sgd_step(&w->cross_weight_v, &g->cross_weight_v, lr);
    blt_sgd_step(&w->cross_weight_proj, &g->cross_weight_proj, lr);
}

static void sgd_step_decoder_layer(blt_local_decoder_layer_storage *w, blt_local_decoder_layer_grad *g, float lr) {
    blt_sgd_step(&w->cross_norm_weight, &g->cross_norm_weight, lr);
    blt_sgd_step(&w->cross_weight_q, &g->cross_weight_q, lr);
    blt_sgd_step(&w->cross_weight_k, &g->cross_weight_k, lr);
    blt_sgd_step(&w->cross_weight_v, &g->cross_weight_v, lr);
    blt_sgd_step(&w->cross_weight_proj, &g->cross_weight_proj, lr);
    blt_sgd_step(&w->norm1_weight, &g->norm1_weight, lr);
    blt_sgd_step(&w->attn_qkv_w, &g->attn_qkv_w, lr);
    blt_sgd_step(&w->attn_proj_w, &g->attn_proj_w, lr);
    blt_sgd_step(&w->norm2_weight, &g->norm2_weight, lr);
    blt_sgd_step(&w->ffn_up_w, &g->ffn_up_w, lr);
    blt_sgd_step(&w->ffn_gate_w, &g->ffn_gate_w, lr);
    blt_sgd_step(&w->ffn_down_w, &g->ffn_down_w, lr);
}

// embedding_grad and n-gram tables are scatter-add targets and must be zeroed before every
//  backward call
static void zero_scatter_grads(blt_model_grad *grad, const blt_local_encoder *enc) {
    zero_tensor(&grad->encoder_grad->embedding_grad);
    for (size_t i = 0; i < enc->ngram_weights.num_tables; i++) {
        zero_tensor(&grad->encoder_grad->ngram_grads.tables[i]);
    }
}

static void apply_sgd_to_model(blt_model *model, blt_model_grad *grad, float lr) {
    blt_local_encoder *enc = model->encoder;
    blt_local_encoder_grad *enc_grad = grad->encoder_grad;
    blt_sgd_step(&enc->byte_embedding_weight, &enc_grad->embedding_grad, lr);
    for (size_t i = 0; i < enc->ngram_weights.num_tables; i++) {
        blt_sgd_step(&enc->ngram_weights.tables[i], &enc_grad->ngram_grads.tables[i], lr);
    }
    for (size_t i = 0; i < enc->config.num_layers; i++) {
        sgd_step_encoder_layer(&enc->layers[i], &enc_grad->layer_grads[i], lr);
    }

    blt_transformer_stack *stack = &model->global->stack;
    blt_transformer_stack_grad *stack_grad = grad->global_grad->stack_grad;
    for (size_t i = 0; i < stack->num_layers; i++) {
        sgd_step_transformer_layer(&stack->layer_storage[i], &stack_grad->layer_grads[i], lr);
    }

    blt_local_decoder *dec = model->decoder;
    blt_local_decoder_grad *dec_grad = grad->decoder_grad;
    for (size_t i = 0; i < dec->config.num_layers; i++) {
        sgd_step_decoder_layer(&dec->layers[i], &dec_grad->layer_grads[i], lr);
    }
    blt_sgd_step(&dec->lm_head_weight, &dec_grad->lm_head_grad, lr);
}

// Fixed-stride patching so the overfit test
// isolates the encoder/global/decoder pipeline itself
static size_t build_fixed_patches(size_t seq_len, size_t patch_len, blt_patch_info *patches_out) {
    size_t num_patches = 0;
    size_t start = 0;
    while (start < seq_len) {
        size_t len = (start + patch_len <= seq_len) ? patch_len : (seq_len - start);
        patches_out[num_patches].start_idx = start;
        patches_out[num_patches].length = len;
        patches_out[num_patches].peak_entropy = 0.0f;
        num_patches++;
        start += len;
    }
    return num_patches;
}

//----------------------------------------------------------------------
// Multi-document model bridge test
//
// The model API stores document starts including the implicit start at byte 0.
// Mask consumers store only later starts. This test verifies the conversion
// across encoder, global, and decoder paths without involving cross-attention.

int run_model_multi_doc_isolation(void) {
    srand(91);

    blt_arena *model_arena = blt_arena_create(4 * 1024 * 1024, BLT_BACKEND_CPU);
    blt_arena *scratch_a = blt_arena_create(8 * 1024 * 1024, BLT_BACKEND_CPU);
    blt_arena *scratch_b = blt_arena_create(8 * 1024 * 1024, BLT_BACKEND_CPU);
    TEST_ASSERT(model_arena && scratch_a && scratch_b);

    blt_model_config config;
    make_small_model_config(&config);
    config.decoder_config.cross_attn_placement = BLT_XATTN_NONE;

    blt_model *model = blt_model_create(model_arena, &config);
    blt_model_grad *grad = blt_model_grad_create(model_arena, model);
    TEST_ASSERT(model != NULL && grad != NULL);
    random_init_model(model, 0.1f);

    const size_t seq_len = 32;
    const size_t vocab_size = config.decoder_config.vocab_size;
    const size_t boundary = 16;
    uint8_t bytes_a_data[32];
    uint8_t bytes_b_data[32];
    for (size_t i = 0; i < seq_len; i++) {
        bytes_a_data[i] = (uint8_t)(i + 17);
        bytes_b_data[i] = bytes_a_data[i];
    }
    for (size_t i = 0; i < 4; i++) {
        bytes_b_data[i] ^= 0x5a;
    }

    blt_patch_info patches[8];
    const size_t num_patches = build_fixed_patches(seq_len, 4, patches);
    TEST_ASSERT(num_patches == 8);
    const size_t doc_boundaries[2] = {0, boundary};

    size_t bytes_shape[1] = {seq_len};
    size_t logits_shape[2] = {seq_len, vocab_size};
    size_t loss_shape[1] = {1};
    blt_tensor bytes_a = blt_tensor_create(scratch_a, bytes_shape, 1, BLT_DTYPE_UINT8);
    blt_tensor bytes_b = blt_tensor_create(scratch_b, bytes_shape, 1, BLT_DTYPE_UINT8);
    memcpy(bytes_a.data, bytes_a_data, sizeof(bytes_a_data));
    memcpy(bytes_b.data, bytes_b_data, sizeof(bytes_b_data));

    blt_tensor logits_a = blt_tensor_create(scratch_a, logits_shape, 2, BLT_DTYPE_FP32);
    blt_tensor logits_b = blt_tensor_create(scratch_b, logits_shape, 2, BLT_DTYPE_FP32);
    blt_tensor loss_a = blt_tensor_create(scratch_a, loss_shape, 1, BLT_DTYPE_FP32);
    blt_tensor loss_b = blt_tensor_create(scratch_b, loss_shape, 1, BLT_DTYPE_FP32);
    blt_model_forward(model, &bytes_a, NULL, patches, num_patches, doc_boundaries, 2, &logits_a, &loss_a, scratch_a);
    blt_model_forward(model, &bytes_b, NULL, patches, num_patches, doc_boundaries, 2, &logits_b, &loss_b, scratch_b);

    const float *out_a = (const float *)logits_a.data;
    const float *out_b = (const float *)logits_b.data;
    bool first_doc_changed = false;
    for (size_t i = 0; i < boundary * vocab_size; i++) {
        if (out_a[i] != out_b[i]) {
            first_doc_changed = true;
            break;
        }
    }
    TEST_ASSERT(first_doc_changed);
    for (size_t i = boundary * vocab_size; i < seq_len * vocab_size; i++) {
        TEST_ASSERT(out_a[i] == out_b[i]);
    }

    // Exercise the same boundary conversion in the recompute-and-backward path.
    blt_model_backward(model, &bytes_a, NULL, patches, num_patches, doc_boundaries, 2, grad, scratch_a);
    TEST_ASSERT(isfinite(compute_model_grad_global_norm(grad, model)));

    blt_arena_destroy(scratch_b);
    blt_arena_destroy(scratch_a);
    blt_arena_destroy(model_arena);
    return 1;
}

//----------------------------------------------------------------------
// Test 1: Overfit-one-batch
//
// A handful of short code snippets, run through the full encoder ->
// global -> decoder model, with a plain per-sample SGD update after
// every forward/backward

int run_blt_model_overfit(void) {
    srand(7);

    blt_arena *model_arena = blt_arena_create(4 * 1024 * 1024, BLT_BACKEND_CPU);
    blt_arena *scratch_arena = blt_arena_create(8 * 1024 * 1024, BLT_BACKEND_CPU);
    if (!model_arena || !scratch_arena) {
        return 0;
    }

    blt_model_config config;
    make_small_model_config(&config);

    blt_model *model = blt_model_create(model_arena, &config);
    blt_model_grad *grad = blt_model_grad_create(model_arena, model);
    TEST_ASSERT(model != NULL && grad != NULL);

    random_init_model(model, 0.1f);

    const char *snippets[] = {
        "int x=1;\n",
        "return 0;\n",
        "for(;;){}\n",
    };
    const size_t num_snippets = sizeof(snippets) / sizeof(snippets[0]);
    const size_t patch_len = 4;
    const float lr = 0.1f;
    const size_t max_steps = 3000;
    const float loss_threshold = 0.05f;
    const float max_grad_norm = 5.0f;

    float avg_loss = 1e9f;

    for (size_t step = 0; step < max_steps; step++) {
        avg_loss = 0.0f;

        for (size_t s = 0; s < num_snippets; s++) {
            blt_arena_reset(scratch_arena);

            size_t seq_len = strlen(snippets[s]);
            TEST_ASSERT(seq_len >= 2 && seq_len <= config.encoder_config.max_seq_len);

            size_t bytes_shape[1] = {seq_len};
            blt_tensor bytes_in = blt_tensor_create(scratch_arena, bytes_shape, 1, BLT_DTYPE_UINT8);
            memcpy(bytes_in.data, snippets[s], seq_len);

            blt_patch_info patches[32];
            size_t num_patches = build_fixed_patches(seq_len, patch_len, patches);
            size_t doc_boundaries[1] = {0};

            size_t logits_shape[2] = {seq_len, config.decoder_config.vocab_size};
            blt_tensor logits = blt_tensor_create(scratch_arena, logits_shape, 2, BLT_DTYPE_FP32);
            size_t scalar_shape[1] = {1};
            blt_tensor loss = blt_tensor_create(scratch_arena, scalar_shape, 1, BLT_DTYPE_FP32);

            blt_model_forward(model, &bytes_in, NULL, patches, num_patches, doc_boundaries, 1, &logits, &loss,
                              scratch_arena);

            avg_loss += ((float *)loss.data)[0];

            zero_scatter_grads(grad, model->encoder);
            blt_model_backward(model, &bytes_in, NULL, patches, num_patches, doc_boundaries, 1, grad, scratch_arena);

            clip_model_grad_global_norm(grad, model, max_grad_norm);
            apply_sgd_to_model(model, grad, lr);
        }

        avg_loss /= (float)num_snippets;

        if (avg_loss < loss_threshold) {
            break;
        }
    }

    printf("    final avg loss = %f\n", (double)avg_loss);
    TEST_ASSERT(avg_loss < loss_threshold);

    blt_arena_destroy(scratch_arena);
    blt_arena_destroy(model_arena);
    return 1;
}

//----------------------------------------------------------------------
// Test 2: Greedy generation sanity check
//
// Not automated pass/fail (qualitative sampling, informal): lightly bias a
// fresh model toward the same snippets, then greedily generate from a
// small fixed prompt set and print the completions for a human to read.
// The only hard assertions are structural (non-NULL arenas, correct
// output length) -- this test always returns success once generation
// runs without crashing.

int run_blt_model_generate_sanity(void) {
    srand(42);

    blt_arena *model_arena = blt_arena_create(4 * 1024 * 1024, BLT_BACKEND_CPU);
    blt_arena *scratch_arena = blt_arena_create(8 * 1024 * 1024, BLT_BACKEND_CPU);
    if (!model_arena || !scratch_arena) {
        return 0;
    }

    blt_model_config config;
    make_small_model_config(&config);

    blt_model *model = blt_model_create(model_arena, &config);
    blt_model_grad *grad = blt_model_grad_create(model_arena, model);
    TEST_ASSERT(model != NULL && grad != NULL);

    random_init_model(model, 0.1f);

    // short warmup
    const char *snippets[] = {
        "int x=1;\n",
        "return 0;\n",
        "for(;i<x;i++){}\n",
    };

    const size_t num_snippets = sizeof(snippets) / sizeof(snippets[0]);
    const size_t patch_len = 4;
    const float lr = 0.1f;
    const size_t warmup_steps = 300;
    const float max_grad_norm = 5.0f;

    for (size_t step = 0; step < warmup_steps; step++) {
        for (size_t s = 0; s < num_snippets; s++) {
            blt_arena_reset(scratch_arena);

            size_t seq_len = strlen(snippets[s]);
            size_t bytes_shape[1] = {seq_len};
            blt_tensor bytes_in = blt_tensor_create(scratch_arena, bytes_shape, 1, BLT_DTYPE_UINT8);
            memcpy(bytes_in.data, snippets[s], seq_len);

            blt_patch_info patches[32];
            size_t num_patches = build_fixed_patches(seq_len, patch_len, patches);
            size_t doc_boundaries[1] = {0};

            size_t logits_shape[2] = {seq_len, config.decoder_config.vocab_size};
            blt_tensor logits = blt_tensor_create(scratch_arena, logits_shape, 2, BLT_DTYPE_FP32);
            size_t scalar_shape[1] = {1};
            blt_tensor loss = blt_tensor_create(scratch_arena, scalar_shape, 1, BLT_DTYPE_FP32);

            blt_model_forward(model, &bytes_in, NULL, patches, num_patches, doc_boundaries, 1, &logits, &loss,
                              scratch_arena);

            zero_scatter_grads(grad, model->encoder);
            blt_model_backward(model, &bytes_in, NULL, patches, num_patches, doc_boundaries, 1, grad, scratch_arena);

            clip_model_grad_global_norm(grad, model, max_grad_norm);
            apply_sgd_to_model(model, grad, lr);
        }
    }

    // Small entropy LM to drive the patcher during generation.
    blt_entropy_lm_config entropy_cfg = {0};
    entropy_cfg.embed_dim = 16;
    entropy_cfg.num_layers = 1;
    entropy_cfg.hidden_dim = 32;
    entropy_cfg.num_heads = 2;
    entropy_cfg.max_seq_len = 64;
    entropy_cfg.rope_theta = 10000.0f;

    blt_entropy_lm *entropy_model = blt_entropy_lm_create(model_arena, &entropy_cfg);
    TEST_ASSERT(entropy_model != NULL);
    random_init_entropy_lm(entropy_model, 0.1f);

    blt_patcher_config patcher_cfg = {0};
    patcher_cfg.threshold_global = 1.0f;
    patcher_cfg.threshold_monotonic = 0.5f;
    patcher_cfg.max_patch_length = 8;
    patcher_cfg.rule = BLT_PATCH_RULE_GLOBAL;
    patcher_cfg.reset_on_newline = false;

    const char *prompts[] = {"int ", "for(", "ret"};
    const size_t num_prompts = sizeof(prompts) / sizeof(prompts[0]);
    const size_t max_new_bytes = 16;

    printf("    check generations:\n");

    for (size_t p = 0; p < num_prompts; p++) {
        blt_arena_reset(scratch_arena);

        size_t prompt_len = strlen(prompts[p]);
        size_t out_len = prompt_len + max_new_bytes;
        uint8_t *output = (uint8_t *)malloc(out_len + 1);
        TEST_ASSERT(output != NULL);

        blt_generate_greedy(model, entropy_model, &patcher_cfg, (const uint8_t *)prompts[p], prompt_len, max_new_bytes,
                            output, scratch_arena);

        printf("        prompt=\"%s\" -> \"", prompts[p]);
        for (size_t i = 0; i < out_len; i++) {
            uint8_t c = output[i];
            putchar(isprint(c) ? (int)c : '.');
        }
        printf("\"\n");

        free(output);
    }

    blt_arena_destroy(scratch_arena);
    blt_arena_destroy(model_arena);
    return 1;
}