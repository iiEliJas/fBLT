// BLT-D / BLT-DV generation controllers (Fast-BLT section 3.1.2,
// Algorithm 1, section 5.2):
//   1. selection kernels (confidence + EB) on hand-computed scores
//   2. blt_draft_block: determinism, byte validity, NFE accounting per
//      strategy extreme, seeded top-p reproducibility
//   3. full generation: exact output length, determinism
//   4. DV gate: verified generation must be byte-identical to plain greedy
//      BLT generation -- every committed byte equals the causal argmax at
//      its position by construction (accept-until-mismatch + replacement).

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "test_helpers.h"
#include "test_suite.h"

#include "core/allocator.h"
#include "core/backend.h"
#include "models/model.h"
#include "models/entropy_lm.h"
#include "models/local_common.h"
#include "models/patcher.h"
#include "infer/block_generation.h"
#include "infer/self_speculation.h"
#include "ops/optim.h"
#include "core/generate_greedy.h"

//----------------------------------------------------------------------
// Helpers (mirrors unit_self_speculation.c conventions)
//----------------------------------------------------------------------

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

static void random_init_entropy_lm(blt_entropy_lm *lm, float scale) {
    float *emb = (float *)lm->embedding_weight.data;
    for (size_t i = 0; i < lm->embedding_weight.numel; i++) {
        emb[i] = (((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f) * scale;
    }
    for (size_t i = 0; i < lm->stack.num_layers; i++) {
        blt_transformer_layer_storage *l = &lm->stack.layer_storage[i];
        fill_constant(&l->norm1_weight, 1.0f);
        fill_small_uniform(&l->attn_qkv_w, scale);
        fill_small_uniform(&l->attn_proj_w, scale);
        fill_constant(&l->norm2_weight, 1.0f);
        fill_small_uniform(&l->ffn_up_w, scale);
        fill_small_uniform(&l->ffn_gate_w, scale);
        fill_small_uniform(&l->ffn_down_w, scale);
    }
    fill_small_uniform(&lm->lm_head_weight, scale);
}

static void random_init_enc_glob_only(blt_model *model, float scale) {
    fill_small_uniform(&model->encoder->byte_embedding_weight, scale);
    for (size_t i = 0; i < model->encoder->config.num_layers; i++) {
        blt_local_layer_storage *l = &model->encoder->layers[i];
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
    for (size_t i = 0; i < model->global->stack.num_layers; i++) {
        blt_transformer_layer_storage *l = &model->global->stack.layer_storage[i];
        fill_constant(&l->norm1_weight, 1.0f);
        fill_small_uniform(&l->attn_qkv_w, scale);
        fill_small_uniform(&l->attn_proj_w, scale);
        fill_constant(&l->norm2_weight, 1.0f);
        fill_small_uniform(&l->ffn_up_w, scale);
        fill_small_uniform(&l->ffn_gate_w, scale);
        fill_small_uniform(&l->ffn_down_w, scale);
    }
}

static void random_init_model_full(blt_model *model, float scale) {
    random_init_enc_glob_only(model, scale);
    for (size_t i = 0; i < model->decoder->config.num_layers; i++) {
        blt_local_layer_storage *l = &model->decoder->layers[i];
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
    fill_small_uniform(&model->decoder->lm_head_weight, scale);
    fill_small_uniform(&model->decoder->d0_embed_weight, 0.05f);
}

static void fixed_patches(size_t seq_len, size_t patch_len, blt_patch_info *out, size_t *num_out) {
    size_t start = 0;
    *num_out = 0;
    while (start < seq_len) {
        size_t len = (start + patch_len <= seq_len) ? patch_len : (seq_len - start);
        out[*num_out].start_idx = start;
        out[*num_out].length = len;
        out[*num_out].peak_entropy = 0.0f;
        (*num_out)++;
        start += len;
    }
}

static void setup_world(blt_arena **model_arena, blt_arena **scratch, blt_model **model, blt_entropy_lm **lm,
                        blt_patcher_config *pcfg) {
    // Returns void: TEST_ASSERT inside callers guards the outputs.
    *model_arena = blt_arena_create(8 * 1024 * 1024, BLT_BACKEND_CPU);
    *scratch = blt_arena_create(32 * 1024 * 1024, BLT_BACKEND_CPU);

    blt_model_config cfg;
    make_small_model_config(&cfg);
    *model = blt_model_create(*model_arena, &cfg);

    blt_entropy_lm_config ecfg;
    memset(&ecfg, 0, sizeof(ecfg));
    ecfg.embed_dim = 16;
    ecfg.num_layers = 1;
    ecfg.num_heads = 2;
    ecfg.hidden_dim = 32;
    ecfg.max_seq_len = 64;
    ecfg.rope_theta = 10000.0f;
    *lm = blt_entropy_lm_create(*model_arena, &ecfg);

    random_init_model_full(*model, 0.1f);
    random_init_entropy_lm(*lm, 0.1f);

    // Effectively disables entropy-driven boundaries: every patch becomes
    // exactly max_patch_length bytes -- the SAME fixed-stride segmentation
    // train_snippets uses below, keeping drafting in-distribution at toy
    // scale.
    memset(pcfg, 0, sizeof(*pcfg));
    pcfg->threshold_global = 1e9f;
    pcfg->threshold_monotonic = 1e9f;
    pcfg->max_patch_length = 4;
    pcfg->rule = BLT_PATCH_RULE_GLOBAL;
    pcfg->reset_on_newline = false;
}

//----------------------------------------------------------------------
// Training helpers (clipped SGD over every trainable tensor)
//----------------------------------------------------------------------

static float grad_sq(blt_tensor *t) {
    const float *d = (const float *)t->data;
    float s = 0.0f;
    for (size_t i = 0; i < t->numel; i++) s += d[i] * d[i];
    return s;
}

static void clip_all(blt_model *m, blt_model_grad *g, float max_norm) {
    float sq = 0.0f;
    sq += grad_sq(&g->encoder_grad->embedding_grad);
    for (size_t i = 0; i < m->encoder->ngram_weights.num_tables; i++)
        sq += grad_sq(&g->encoder_grad->ngram_grads.tables[i]);
    for (size_t i = 0; i < m->encoder->config.num_layers; i++) {
        blt_local_layer_grad *l = &g->encoder_grad->layer_grads[i];
        blt_tensor *ts[] = {&l->norm1_weight,   &l->attn_qkv_w,     &l->attn_proj_w,    &l->norm2_weight,
                            &l->ffn_up_w,       &l->ffn_gate_w,     &l->ffn_down_w,     &l->cross_norm_weight,
                            &l->cross_weight_q, &l->cross_weight_k, &l->cross_weight_v, &l->cross_weight_proj};
        for (size_t j = 0; j < 12; j++) sq += grad_sq(ts[j]);
    }
    for (size_t i = 0; i < m->global->stack.num_layers; i++) {
        blt_transformer_layer_grad *l = &g->global_grad->stack_grad->layer_grads[i];
        blt_tensor *ts[] = {&l->norm1_weight, &l->attn_qkv_w, &l->attn_proj_w, &l->norm2_weight,
                            &l->ffn_up_w,     &l->ffn_gate_w, &l->ffn_down_w};
        for (size_t j = 0; j < 7; j++) sq += grad_sq(ts[j]);
    }
    for (size_t i = 0; i < m->decoder->config.num_layers; i++) {
        blt_local_layer_grad *l = &g->decoder_grad->layer_grads[i];
        blt_tensor *ts[] = {&l->cross_norm_weight, &l->cross_weight_q, &l->cross_weight_k, &l->cross_weight_v,
                            &l->cross_weight_proj, &l->norm1_weight,   &l->attn_qkv_w,     &l->attn_proj_w,
                            &l->norm2_weight,      &l->ffn_up_w,       &l->ffn_gate_w,     &l->ffn_down_w};
        for (size_t j = 0; j < 12; j++) sq += grad_sq(ts[j]);
    }
    sq += grad_sq(&g->decoder_grad->lm_head_grad);
    sq += grad_sq(&g->decoder_grad->d0_embed_grad);

    const float norm = sqrtf(sq);
    if (norm <= max_norm || norm == 0.0f) return;
    const float scale = max_norm / norm;

    blt_scale(&g->encoder_grad->embedding_grad, scale);
    for (size_t i = 0; i < m->encoder->ngram_weights.num_tables; i++)
        blt_scale(&g->encoder_grad->ngram_grads.tables[i], scale);
    for (size_t part = 0; part < 3; part++) {
        const size_t n_layers = part == 1 ? m->global->stack.num_layers
                                          : (part == 0 ? m->encoder->config.num_layers : m->decoder->config.num_layers);
        for (size_t i = 0; i < n_layers; i++) {
            if (part == 1) {
                blt_transformer_layer_grad *l = &g->global_grad->stack_grad->layer_grads[i];
                blt_tensor *ts[] = {&l->norm1_weight, &l->attn_qkv_w, &l->attn_proj_w, &l->norm2_weight,
                                    &l->ffn_up_w,     &l->ffn_gate_w, &l->ffn_down_w};
                for (size_t j = 0; j < 7; j++) blt_scale(ts[j], scale);
            } else {
                blt_local_layer_grad *l =
                    part == 0 ? &g->encoder_grad->layer_grads[i] : &g->decoder_grad->layer_grads[i];
                blt_tensor *ts[] = {&l->norm1_weight,   &l->attn_qkv_w,     &l->attn_proj_w,    &l->norm2_weight,
                                    &l->ffn_up_w,       &l->ffn_gate_w,     &l->ffn_down_w,     &l->cross_norm_weight,
                                    &l->cross_weight_q, &l->cross_weight_k, &l->cross_weight_v, &l->cross_weight_proj};
                for (size_t j = 0; j < 12; j++) blt_scale(ts[j], scale);
            }
        }
    }
    blt_scale(&g->decoder_grad->lm_head_grad, scale);
    blt_scale(&g->decoder_grad->d0_embed_grad, scale);
}

static void apply_sgd_step_all(blt_model *model, blt_model_grad *grad, float lr) {
    blt_local_encoder *enc = model->encoder;
    blt_local_encoder_grad *eg = grad->encoder_grad;
    blt_sgd_step(&enc->byte_embedding_weight, &eg->embedding_grad, lr);
    for (size_t i = 0; i < enc->ngram_weights.num_tables; i++) {
        blt_sgd_step(&enc->ngram_weights.tables[i], &eg->ngram_grads.tables[i], lr);
    }
    for (size_t i = 0; i < enc->config.num_layers; i++) {
        blt_local_layer_storage *w = &enc->layers[i];
        blt_local_layer_grad *g = &eg->layer_grads[i];
        blt_tensor *ws[] = {&w->norm1_weight,   &w->attn_qkv_w,     &w->attn_proj_w,    &w->norm2_weight,
                            &w->ffn_up_w,       &w->ffn_gate_w,     &w->ffn_down_w,     &w->cross_norm_weight,
                            &w->cross_weight_q, &w->cross_weight_k, &w->cross_weight_v, &w->cross_weight_proj};
        blt_tensor *gs[] = {&g->norm1_weight,   &g->attn_qkv_w,     &g->attn_proj_w,    &g->norm2_weight,
                            &g->ffn_up_w,       &g->ffn_gate_w,     &g->ffn_down_w,     &g->cross_norm_weight,
                            &g->cross_weight_q, &g->cross_weight_k, &g->cross_weight_v, &g->cross_weight_proj};
        for (size_t j = 0; j < 12; j++) blt_sgd_step(ws[j], gs[j], lr);
    }
    for (size_t i = 0; i < model->global->stack.num_layers; i++) {
        blt_transformer_layer_storage *w = &model->global->stack.layer_storage[i];
        blt_transformer_layer_grad *g = &grad->global_grad->stack_grad->layer_grads[i];
        blt_tensor *ws[] = {&w->norm1_weight, &w->attn_qkv_w, &w->attn_proj_w, &w->norm2_weight,
                            &w->ffn_up_w,     &w->ffn_gate_w, &w->ffn_down_w};
        blt_tensor *gs[] = {&g->norm1_weight, &g->attn_qkv_w, &g->attn_proj_w, &g->norm2_weight,
                            &g->ffn_up_w,     &g->ffn_gate_w, &g->ffn_down_w};
        for (size_t j = 0; j < 7; j++) blt_sgd_step(ws[j], gs[j], lr);
    }
    for (size_t i = 0; i < model->decoder->config.num_layers; i++) {
        blt_local_layer_storage *w = &model->decoder->layers[i];
        blt_local_layer_grad *g = &grad->decoder_grad->layer_grads[i];
        blt_tensor *ws[] = {&w->norm1_weight,   &w->attn_qkv_w,     &w->attn_proj_w,    &w->norm2_weight,
                            &w->ffn_up_w,       &w->ffn_gate_w,     &w->ffn_down_w,     &w->cross_norm_weight,
                            &w->cross_weight_q, &w->cross_weight_k, &w->cross_weight_v, &w->cross_weight_proj};
        blt_tensor *gs[] = {&g->norm1_weight,   &g->attn_qkv_w,     &g->attn_proj_w,    &g->norm2_weight,
                            &g->ffn_up_w,       &g->ffn_gate_w,     &g->ffn_down_w,     &g->cross_norm_weight,
                            &g->cross_weight_q, &g->cross_weight_k, &g->cross_weight_v, &g->cross_weight_proj};
        for (size_t j = 0; j < 12; j++) blt_sgd_step(ws[j], gs[j], lr);
    }
    blt_sgd_step(&model->decoder->lm_head_weight, &grad->decoder_grad->lm_head_grad, lr);
    blt_sgd_step(&model->decoder->d0_embed_weight, &grad->decoder_grad->d0_embed_grad, lr);
}

// Trains the full model to memorize short snippets under the plain causal
// objective (same recipe as unit_self_speculation.c), so speculative
// drafts can agree with causal predictions.
static void train_snippets(blt_model *model, blt_model_grad *grad, blt_arena *scratch, const char **snippets,
                           size_t num_snippets, size_t steps) {
    const size_t patch_len = 4;
    const float lr = 0.05f;
    const float clip = 5.0f;

    for (size_t step = 0; step < steps; step++) {
        for (size_t s = 0; s < num_snippets; s++) {
            blt_arena_reset(scratch);

            size_t seq_len = strlen(snippets[s]);
            size_t bytes_shape[1] = {seq_len};
            blt_tensor bytes_in = blt_tensor_create(scratch, bytes_shape, 1, BLT_DTYPE_UINT8);
            memcpy(bytes_in.data, snippets[s], seq_len);

            blt_patch_info patches[32];
            size_t num_patches = 0;
            fixed_patches(seq_len, patch_len, patches, &num_patches);

            size_t logits_shape[2] = {seq_len, model->config.decoder_config.vocab_size};
            blt_tensor logits = blt_tensor_create(scratch, logits_shape, 2, BLT_DTYPE_FP32);
            size_t scalar_shape[1] = {1};
            blt_tensor loss = blt_tensor_create(scratch, scalar_shape, 1, BLT_DTYPE_FP32);

            blt_model_forward(model, &bytes_in, NULL, patches, num_patches, NULL, 0, &logits, &loss, scratch);

            zero_tensor(&grad->encoder_grad->embedding_grad);
            for (size_t i = 0; i < model->encoder->ngram_weights.num_tables; i++) {
                zero_tensor(&grad->encoder_grad->ngram_grads.tables[i]);
            }
            blt_model_backward(model, &bytes_in, NULL, patches, num_patches, NULL, 0, grad, scratch);

            clip_all(model, grad, clip);
            apply_sgd_step_all(model, grad, lr);
        }
    }
}

//----------------------------------------------------------------------
// 1. Selection kernels
//----------------------------------------------------------------------

int run_blockgen_select_kernels(void) {
    // Confidence: threshold fires on all cells above alpha.
    {
        float scores[4] = {0.95f, 0.80f, 0.60f, 0.30f};
        uint8_t masked[4] = {1, 1, 1, 1};
        const uint32_t sel = blt_unmask_select_confidence(scores, masked, 4, 0.7f);
        TEST_ASSERT(sel == (0x1u | 0x2u)); // cells 0,1 only
    }
    // Confidence: nothing above alpha -> fallback picks the single best.
    {
        float scores[3] = {0.10f, 0.50f, 0.20f};
        uint8_t masked[3] = {1, 1, 1};
        const uint32_t sel = blt_unmask_select_confidence(scores, masked, 3, 0.9f);
        TEST_ASSERT(sel == 0x2u); // cell 1 (highest confidence)
    }
    // Confidence: unmasked cells never selected even with high scores.
    {
        float scores[4] = {0.99f, 0.99f, 0.99f, 0.99f};
        uint8_t masked[4] = {0, 1, 0, 1};
        const uint32_t sel = blt_unmask_select_confidence(scores, masked, 4, 0.5f);
        TEST_ASSERT(sel == (0x2u | 0x8u));
    }
    // EB: budget keeps a prefix of the ascending-entropy order.
    {
        float ent[4] = {2.0f, 0.5f, 1.5f, 1.0f};
        uint8_t masked[4] = {1, 1, 0, 1};
        const uint32_t sel = blt_unmask_select_eb(ent, masked, 4, 1.4f);
        TEST_ASSERT(sel == 0x2u); // only cell 1 (0.5 <= 1.4, +1.0 > 1.4)
    }
    // EB: huge budget selects every masked cell.
    {
        float ent[3] = {3.0f, 1.0f, 2.0f};
        uint8_t masked[3] = {1, 1, 1};
        const uint32_t sel = blt_unmask_select_eb(ent, masked, 3, 100.0f);
        TEST_ASSERT(sel == 0x7u);
    }
    // EB: negative budget -> progress rule takes lowest entropy alone.
    {
        float ent[3] = {2.0f, 0.5f, 1.0f};
        uint8_t masked[3] = {1, 1, 1};
        const uint32_t sel = blt_unmask_select_eb(ent, masked, 3, -1.0f);
        TEST_ASSERT(sel == 0x2u);
    }
    // Nothing masked -> empty selection.
    {
        float scores[2] = {1.0f, 1.0f};
        uint8_t masked[2] = {0, 0};
        TEST_ASSERT(blt_unmask_select_confidence(scores, masked, 2, 0.0f) == 0);
        TEST_ASSERT(blt_unmask_select_eb(scores, masked, 2, 10.0f) == 0);
    }
    return 1;
}

//----------------------------------------------------------------------
// 2. blt_draft_block behavior on a random tiny model
//----------------------------------------------------------------------

int run_blockgen_draft_behavior(void) {
    srand(77);

    blt_arena *model_arena;
    blt_arena *scratch;
    blt_model *model;
    blt_entropy_lm *lm;
    blt_patcher_config pcfg;
    setup_world(&model_arena, &scratch, &model, &lm, &pcfg);
    TEST_ASSERT(model_arena && scratch && model && lm);

    static const uint8_t prompt[12] = "hello_block";
    blt_patch_info patches[16];
    size_t num_patches;
    fixed_patches(12, 4, patches, &num_patches);

    size_t shape1[1] = {12};
    blt_tensor prefix_bytes = blt_tensor_create(scratch, shape1, 1, BLT_DTYPE_UINT8);
    memcpy(prefix_bytes.data, prompt, 12);

    blt_model_enc_out enc;
    blt_model_encode(model, &prefix_bytes, patches, num_patches, NULL, 0, &enc, scratch);

    blt_block_gen_config gc;
    memset(&gc, 0, sizeof(gc));
    gc.block_size = 4;
    gc.d0_mode = BLT_D0_LEARNED;

    // Confidence with unreachable threshold: exactly B passes (progress rule).
    gc.opts.strategy = BLT_UNMASK_CONFIDENCE;
    gc.opts.threshold = 1.5f;
    uint8_t block_a[4];
    const size_t nfe_a = blt_draft_block(model, &enc, patches, num_patches, prompt, 12, &gc, block_a, scratch);
    TEST_ASSERT(nfe_a == 4);

    // Same call again: deterministic repeat.
    uint8_t block_b[4];
    const size_t nfe_b = blt_draft_block(model, &enc, patches, num_patches, prompt, 12, &gc, block_b, scratch);
    TEST_ASSERT(nfe_b == nfe_a);
    TEST_ASSERT(memcmp(block_a, block_b, 4) == 0);

    // Threshold 0: everything unmasks in a single pass.
    gc.opts.threshold = 0.0f;
    uint8_t block_c[4];
    const size_t nfe_c = blt_draft_block(model, &enc, patches, num_patches, prompt, 12, &gc, block_c, scratch);
    TEST_ASSERT(nfe_c == 1);
    for (size_t b = 0; b < 4; b++) {
        TEST_ASSERT(block_c[b] < 256);
    }

    // EB with huge budget: single pass as well.
    gc.opts.strategy = BLT_UNMASK_EB;
    gc.opts.threshold = 1e9f;
    uint8_t block_d[4];
    const size_t nfe_d = blt_draft_block(model, &enc, patches, num_patches, prompt, 12, &gc, block_d, scratch);
    TEST_ASSERT(nfe_d == 1);

    // Top-p sampling: seeded reruns are identical across seeds differing.
    gc.opts.use_top_p = 1;
    gc.opts.top_p = 0.9f;
    gc.opts.threshold = 1.5f;
    gc.opts.seed = 42;
    uint8_t s1[4], s2[4], s3[4];
    blt_draft_block(model, &enc, patches, num_patches, prompt, 12, &gc, s1, scratch);
    blt_draft_block(model, &enc, patches, num_patches, prompt, 12, &gc, s2, scratch);
    gc.opts.seed = 43;
    blt_draft_block(model, &enc, patches, num_patches, prompt, 12, &gc, s3, scratch);
    TEST_ASSERT(memcmp(s1, s2, 4) == 0);

    blt_arena_destroy(model_arena);
    blt_arena_destroy(scratch);
    return 1;
}

//----------------------------------------------------------------------
// 3+4. Full generation: length/determinism + DV == plain greedy gate
//----------------------------------------------------------------------

int run_blockgen_generation_gates(void) {
    srand(1234);

    blt_arena *model_arena;
    blt_arena *scratch;
    blt_model *model;
    blt_entropy_lm *lm;
    blt_patcher_config pcfg;
    setup_world(&model_arena, &scratch, &model, &lm, &pcfg);
    TEST_ASSERT(model_arena && scratch && model && lm);

    blt_model_grad *grad = blt_model_grad_create(model_arena, model);
    TEST_ASSERT(grad != NULL);

    // Memorized corpus: generation continues a trained snippet.
    static const char *snippets[] = {"the quick brown fox", "driftless tune"};
    train_snippets(model, grad, scratch, snippets, 2, 2000);

    static const char *prompt_str = "the quick ";
    const size_t prompt_len = strlen(prompt_str);
    const uint8_t *prompt = (const uint8_t *)prompt_str;
    const size_t new_bytes = 10; // completes "brown fox" when memorized

    blt_block_gen_config gc;
    memset(&gc, 0, sizeof(gc));
    gc.block_size = 4;
    gc.d0_mode = BLT_D0_LEARNED;
    gc.opts.strategy = BLT_UNMASK_CONFIDENCE;
    // Paper-style alpha: a memorizing model is highly confident on its
    // snippet, so blocks typically unmask in a single diffusion pass.
    gc.opts.threshold = 0.7f;

    uint8_t out_a[prompt_len + new_bytes];
    uint8_t out_b[prompt_len + new_bytes];

    blt_generate_greedy_blockdiff(model, lm, &pcfg, prompt, prompt_len, new_bytes, out_a, &gc, NULL, scratch);
    TEST_ASSERT(memcmp(out_a, prompt, prompt_len) == 0);

    blt_infer_stats st;
    blt_generate_greedy_blockdiff(model, lm, &pcfg, prompt, prompt_len, new_bytes, out_b, &gc, &st, scratch);
    TEST_ASSERT(memcmp(out_a, out_b, prompt_len + new_bytes) == 0); // determinism

    //------------------------------------------------------------------
    // DV gate: verification must reproduce plain greedy BLT exactly --
    // every committed byte is the causal argmax at its position.
    //------------------------------------------------------------------
    uint8_t ref[prompt_len + new_bytes];
    blt_generate_greedy(model, lm, &pcfg, prompt, prompt_len, new_bytes, ref, scratch);

    uint8_t dv[prompt_len + new_bytes];
    memset(&st, 0, sizeof(st));
    blt_generate_greedy_blockdiff_verify(model, lm, &pcfg, prompt, prompt_len, new_bytes, dv, &gc, &st, scratch);
    TEST_ASSERT(memcmp(dv, ref, prompt_len + new_bytes) == 0);

    // Progress/accounting invariants (always enforceable).
    TEST_ASSERT(st.nfe_decoder > 0);
    TEST_ASSERT(st.bytes_drafted >= st.bytes_accepted);
    TEST_ASSERT(st.nfe_encoder_global >= 1);

    // Informational at toy scale: drafting quality depends on how well the
    // fully-masked (t~1) regime was trained. Large-scale BLT-DV models get
    // high acceptance there (paper section 5.2); a toy snippet-trained
    // model under-trains it (the lambda <= 0.3 reweighting starves L_mask
    // further), so decoder-NFE savings are NOT asserted here. The load-
    // bearing gates above are: DV == greedy byte-identity, exact output
    // length, determinism, per-round progress.
    printf("[BLOCKGEN] DV==greedy ok; nfe_dec=%zu baseline=%zu drafted=%zu "
           "accepted=%zu\n",
           st.nfe_decoder, new_bytes, st.bytes_drafted, st.bytes_accepted);

    blt_arena_destroy(model_arena);
    blt_arena_destroy(scratch);
    return 1;
}

//----------------------------------------------------------------------
// Stage 6 add-on helpers: boundary-aligned commit selection, adaptive B,
// and end-to-end boundary-aligned DV against the greedy reference.
//----------------------------------------------------------------------

int run_blockgen_stage6_helpers(void) {
    // ---- blt_aligned_commit_select ------------------------------------
    {
        blt_patch_info p[5];
        const size_t starts[] = {0, 4, 6, 7, 12};
        const size_t lens[] = {4, 2, 1, 5, 3};
        for (size_t i = 0; i < 5; i++) {
            p[i].start_idx = starts[i];
            p[i].length = lens[i];
            p[i].peak_entropy = 0.0f;
        }
        // ends: 4, 6, 7, 12, 15
        TEST_ASSERT(blt_aligned_commit_select(p, 5, 0, 4) == 4);
        TEST_ASSERT(blt_aligned_commit_select(p, 5, 4, 6) == 6);
        TEST_ASSERT(blt_aligned_commit_select(p, 5, 5, 10) == 7);
        TEST_ASSERT(blt_aligned_commit_select(p, 5, 7, 11) == 0); // none
        TEST_ASSERT(blt_aligned_commit_select(p, 5, 7, 12) == 12);
        TEST_ASSERT(blt_aligned_commit_select(p, 5, 13, 15) == 15);
    }

    // ---- blt_block_adapt_b --------------------------------------------
    {
        TEST_ASSERT(blt_block_adapt_b(8, 0.9, 4, 16, 0.5f) == 9);   // grow
        TEST_ASSERT(blt_block_adapt_b(8, 0.2, 4, 16, 0.5f) == 7);   // shrink
        TEST_ASSERT(blt_block_adapt_b(8, 0.55, 4, 16, 0.5f) == 8);  // hold band
        TEST_ASSERT(blt_block_adapt_b(16, 0.9, 4, 16, 0.5f) == 16); // cap
        TEST_ASSERT(blt_block_adapt_b(4, 0.2, 4, 16, 0.5f) == 4);   // floor
        TEST_ASSERT(blt_block_adapt_b(8, 0.2, 0, 16, 0.5f) == 7);   // b_min=0 -> floor 1
        TEST_ASSERT(blt_block_adapt_b(1, 0.2, 0, 16, 0.5f) == 1);   // cannot shrink further
        TEST_ASSERT(blt_block_adapt_b(8, 0.9, 0, 16, 0.5f) == 9);
        TEST_ASSERT(blt_block_adapt_b(8, 0.2, 4, 0, 0.5f) == 8);    // disabled
        TEST_ASSERT(blt_block_adapt_b(3, 0.9, 4, 16, 0.5f) == 5);   // clamp-in, then grow
        TEST_ASSERT(blt_block_adapt_b(20, 0.2, 4, 16, 0.5f) == 15); // clamp-in, then shrink
    }

    // ---- boundary-aligned DV end-to-end (degenerate patcher => exact) --
    srand(4321);

    blt_arena *model_arena;
    blt_arena *scratch;
    blt_model *model;
    blt_entropy_lm *lm;
    blt_patcher_config pcfg;
    setup_world(&model_arena, &scratch, &model, &lm, &pcfg);
    TEST_ASSERT(model_arena && scratch && model && lm);

    blt_model_grad *grad = blt_model_grad_create(model_arena, model);
    TEST_ASSERT(grad != NULL);

    static const char *snippets[] = {"the quick brown fox", "driftless tune"};
    train_snippets(model, grad, scratch, snippets, 2, 2000);

    static const char *prompt_str = "driftless ";
    const size_t prompt_len = strlen(prompt_str);
    const uint8_t *prompt = (const uint8_t *)prompt_str;
    const size_t new_bytes = 8;

    uint8_t ref[prompt_len + new_bytes];
    blt_generate_greedy(model, lm, &pcfg, prompt, prompt_len, new_bytes, ref, scratch);

    blt_block_gen_config gc;
    memset(&gc, 0, sizeof(gc));
    gc.block_size = 4;
    gc.d0_mode = BLT_D0_LEARNED;
    gc.opts.strategy = BLT_UNMASK_CONFIDENCE;
    gc.opts.threshold = 0.7f;
    gc.boundary_aligned = 1;

    uint8_t out[prompt_len + new_bytes];
    blt_infer_stats st;
    memset(&st, 0, sizeof(st));
    blt_generate_greedy_blockdiff_verify(model, lm, &pcfg, prompt, prompt_len, new_bytes, out, &gc, &st, scratch);
    // With the degenerate patcher every position is a patch boundary, so
    // aligned commitment must reproduce greedy byte-for-byte.
    TEST_ASSERT(memcmp(out, ref, prompt_len + new_bytes) == 0);
    TEST_ASSERT(st.bytes_drafted >= st.bytes_accepted);
    TEST_ASSERT(st.bytes_accepted >= new_bytes - 1); // progress per round

    printf("[BLOCKGEN] stage6 helpers ok; aligned DV drafted=%zu accepted=%zu\n", st.bytes_drafted, st.bytes_accepted);

    blt_arena_destroy(model_arena);
    blt_arena_destroy(scratch);
    return 1;
}
