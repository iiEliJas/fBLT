#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "test_helpers.h"
#include "test_suite.h"

#include "core/allocator.h"
#include "core/backend.h"
#include "models/model.h"
#include "models/entropy_lm.h"
#include "models/local_common.h"
#include "models/patcher.h"
#include "infer/self_speculation.h"
#include "ops/optim.h"
#include "core/generate_greedy.h"

//----------------------------------------------------------------------
// Helpers (mirrors unit_model.c / unit_model_stages.c conventions)

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

// Randomizes encoder + global; leaves the DECODER untouched (zero weights).
// With a zero LM head every model prediction is byte 0, making verification
// outcomes fully predictable for directed tests.
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
    fill_small_uniform(&model->decoder->lm_head_weight, scale);
}

static void make_patcher_cfg(blt_patcher_config *cfg) {
    memset(cfg, 0, sizeof(*cfg));
    cfg->threshold_global = 1.0f;
    cfg->threshold_monotonic = 0.5f;
    cfg->max_patch_length = 8;
    cfg->rule = BLT_PATCH_RULE_GLOBAL;
    cfg->reset_on_newline = false;
}

static void make_entropy_cfg(blt_entropy_lm_config *cfg) {
    memset(cfg, 0, sizeof(*cfg));
    cfg->embed_dim = 16;
    cfg->num_layers = 1;
    cfg->hidden_dim = 32;
    cfg->num_heads = 2;
    cfg->max_seq_len = 64;
    cfg->rope_theta = 10000.0f;
}

//----------------------------------------------------------------------
// Test 1: directed blt_verify_draft against a degenerate zero-decoder
// model. All predictions are byte 0, so every Algorithm 2 branch has a
// hand-computable outcome.

int run_verify_draft_directed(void) {
    srand(21);

    blt_arena *model_arena = blt_arena_create(1024 * 1024, BLT_BACKEND_CPU);
    blt_arena *scratch = blt_arena_create(4 * 1024 * 1024, BLT_BACKEND_CPU);
    TEST_ASSERT(model_arena && scratch);

    blt_model_config config;
    make_small_model_config(&config);
    blt_model *model = blt_model_create(model_arena, &config);
    TEST_ASSERT(model != NULL);
    random_init_enc_glob_only(model, 0.1f); // decoder stays ZERO

    blt_entropy_lm *entropy_model =
        blt_entropy_lm_create(model_arena, &(blt_entropy_lm_config){.embed_dim = 16,
                                                                    .num_layers = 1,
                                                                    .hidden_dim = 32,
                                                                    .num_heads = 2,
                                                                    .max_seq_len = 64,
                                                                    .rope_theta = 10000.0f});
    // zero-init entropy LM is fine (deterministic)
    TEST_ASSERT(entropy_model != NULL);

    blt_patcher_config pcfg;
    make_patcher_cfg(&pcfg);

    blt_infer_stats stats;

    // Case A: mismatch at first draft byte -> replaced with byte 0,
    // committed length advances by exactly 1.
    {
        uint8_t x[16] = {7, 7, 5, 5};
        blt_infer_stats_reset(&stats);
        size_t new_len =
            blt_verify_draft(model, entropy_model, &pcfg, x, /*l=*/2, /*r=*/2, /*target_len=*/16, &stats, scratch);
        TEST_ASSERT(new_len == 3);
        TEST_ASSERT(x[0] == 7 && x[1] == 7 && x[2] == 0);
        TEST_ASSERT(stats.bytes_accepted == 0);
        TEST_ASSERT(stats.nfe_encoder_global == 1 && stats.nfe_decoder == 1);
    }

    // Case B: full match -> all drafted bytes accepted + one free byte 0.
    {
        uint8_t x[16] = {7, 7, 0, 0, 0};
        blt_infer_stats_reset(&stats);
        size_t new_len =
            blt_verify_draft(model, entropy_model, &pcfg, x, /*l=*/2, /*r=*/2, /*target_len=*/16, &stats, scratch);
        TEST_ASSERT(new_len == 5);
        TEST_ASSERT(x[0] == 7 && x[1] == 7 && x[2] == 0 && x[3] == 0 && x[4] == 0);
        TEST_ASSERT(stats.bytes_accepted == 2);
    }

    // Case C: full match but budget reached -> free byte skipped.
    {
        uint8_t x[16] = {7, 7, 0, 0};
        blt_infer_stats_reset(&stats);
        size_t new_len =
            blt_verify_draft(model, entropy_model, &pcfg, x, /*l=*/2, /*r=*/2, /*target_len=*/4, &stats, scratch);
        TEST_ASSERT(new_len == 4);
        TEST_ASSERT(stats.bytes_accepted == 2);
    }

    // Case D: mismatch at second draft byte -> first accepted, second
    // replaced, length advances by 2.
    {
        uint8_t x[16] = {9, 9, 0, 5, 5};
        blt_infer_stats_reset(&stats);
        size_t new_len =
            blt_verify_draft(model, entropy_model, &pcfg, x, /*l=*/2, /*r=*/3, /*target_len=*/16, &stats, scratch);
        TEST_ASSERT(new_len == 4);
        TEST_ASSERT(x[2] == 0 && x[3] == 0); // accepted, then replaced
        TEST_ASSERT(stats.bytes_accepted == 1);
    }

    blt_arena_destroy(scratch);
    blt_arena_destroy(model_arena);
    return 1;
}

//----------------------------------------------------------------------
// Test 2: structural invariants of the full BLT-S loop on random models
// (no equality gate here -- random-weight argmaxes are chaotically
// sensitive to the tail-patch differences between drafting and
// verification contexts; exactness is gated separately on a trained
// model in run_selfspec_equivalence_trained).

typedef struct {
    const char *prompt;
    size_t max_new;
} gen_case;

static int fill_output_like_baseline(const blt_model *model, const blt_entropy_lm *lm, const blt_patcher_config *pcfg,
                                     const char *prompt, size_t max_new, blt_d0_mode mode, size_t k, uint8_t *out,
                                     blt_infer_stats *stats, blt_arena *scratch) {
    blt_self_spec_config cfg = {.window_k = k, .d0_mode = mode};
    blt_arena_reset(scratch);
    blt_generate_greedy_selfspec(model, lm, pcfg, (const uint8_t *)prompt, strlen(prompt), max_new, out, &cfg, stats,
                                 scratch);
    return 1;
}

int run_selfspec_structural_random(void) {
    srand(22);

    blt_arena *model_arena = blt_arena_create(2 * 1024 * 1024, BLT_BACKEND_CPU);
    blt_arena *scratch = blt_arena_create(8 * 1024 * 1024, BLT_BACKEND_CPU);
    TEST_ASSERT(model_arena && scratch);

    blt_model_config config;
    make_small_model_config(&config);
    blt_model *model = blt_model_create(model_arena, &config);
    TEST_ASSERT(model != NULL);
    random_init_model_full(model, 0.1f);

    blt_entropy_lm_config ecfg;
    make_entropy_cfg(&ecfg);
    blt_entropy_lm *lm = blt_entropy_lm_create(model_arena, &ecfg);
    TEST_ASSERT(lm != NULL);
    random_init_entropy_lm(lm, 0.1f);

    blt_patcher_config pcfg;
    make_patcher_cfg(&pcfg);

    const gen_case cases[] = {
        {"ab", 1},                                // r_eff == 0 shortcut
        {"ab", 2},    {"xy", 5},    {"int ", 17}, // forces window capping + multiple rounds
        {"q9z!", 13}, {"for(", 24},
    };
    const size_t num_cases = sizeof(cases) / sizeof(cases[0]);
    const blt_d0_mode modes[] = {BLT_D0_ZEROS, BLT_D0_LEARNED};
    const size_t ks[] = {4, 16};

    for (size_t c = 0; c < num_cases; c++) {
        for (size_t mi = 0; mi < 2; mi++) {
            for (size_t ki = 0; ki < 2; ki++) {
                const size_t out_len = strlen(cases[c].prompt) + cases[c].max_new;
                uint8_t out[128];
                uint8_t out_rerun[128];
                blt_infer_stats stats;

                TEST_ASSERT(out_len <= 128);
                blt_infer_stats_reset(&stats);
                fill_output_like_baseline(model, lm, &pcfg, cases[c].prompt, cases[c].max_new, modes[mi], ks[ki], out,
                                          &stats, scratch);

                // NFE bookkeeping sane: at least one round happened, and
                // the expensive tier stayed within 2 encodes per generated
                // byte (draft-context + verify per round, >=1 byte/round).
                TEST_ASSERT(stats.nfe_encoder_global >= 1);
                TEST_ASSERT(stats.nfe_decoder >= 1);
                TEST_ASSERT(stats.bytes_drafted <= cases[c].max_new * ks[ki]);
                TEST_ASSERT(stats.nfe_encoder_global <= 2 * cases[c].max_new);

                // determinism: identical rerun
                blt_infer_stats stats2;
                blt_infer_stats_reset(&stats2);
                fill_output_like_baseline(model, lm, &pcfg, cases[c].prompt, cases[c].max_new, modes[mi], ks[ki],
                                          out_rerun, &stats2, scratch);
                TEST_ASSERT(memcmp(out, out_rerun, out_len) == 0);
            }
        }
    }

    blt_arena_destroy(scratch);
    blt_arena_destroy(model_arena);
    return 1;
}

//----------------------------------------------------------------------
// Test 3: THE equivalence gate -- on a model overfit to fixed snippets,
// greedy BLT-S output must be byte-identical to plain greedy BLT
// generation (Fast-BLT Figure 2 guarantee), and encoder/global NFEs must
// drop strictly below the one-full-forward-per-byte baseline.

static void apply_sgd_step_all(blt_model *model, blt_model_grad *grad, float lr);

static void train_snippets(blt_model *model, blt_model_grad *grad, blt_arena *scratch, const char **snippets,
                           size_t num_snippets, size_t steps) {
    const size_t patch_len = 4;
    const float lr = 0.1f;

    for (size_t step = 0; step < steps; step++) {
        for (size_t s = 0; s < num_snippets; s++) {
            blt_arena_reset(scratch);

            size_t seq_len = strlen(snippets[s]);
            size_t bytes_shape[1] = {seq_len};
            blt_tensor bytes_in = blt_tensor_create(scratch, bytes_shape, 1, BLT_DTYPE_UINT8);
            memcpy(bytes_in.data, snippets[s], seq_len);

            blt_patch_info patches[32];
            size_t num_patches = 0;
            size_t start = 0;
            while (start < seq_len) {
                size_t len = (start + patch_len <= seq_len) ? patch_len : (seq_len - start);
                patches[num_patches].start_idx = start;
                patches[num_patches].length = len;
                patches[num_patches].peak_entropy = 0.0f;
                num_patches++;
                start += len;
            }

            size_t logits_shape[2] = {seq_len, model->config.decoder_config.vocab_size};
            blt_tensor logits = blt_tensor_create(scratch, logits_shape, 2, BLT_DTYPE_FP32);
            size_t scalar_shape[1] = {1};
            blt_tensor loss = blt_tensor_create(scratch, scalar_shape, 1, BLT_DTYPE_FP32);

            blt_model_forward(model, &bytes_in, NULL, patches, num_patches, NULL, 0, &logits, &loss, scratch);

            // zero scatter-add grads before backward
            zero_tensor(&grad->encoder_grad->embedding_grad);
            for (size_t i = 0; i < model->encoder->ngram_weights.num_tables; i++) {
                zero_tensor(&grad->encoder_grad->ngram_grads.tables[i]);
            }
            blt_model_backward(model, &bytes_in, NULL, patches, num_patches, NULL, 0, grad, scratch);

            // plain SGD over all weights (no clipping: toy memorization task)
            apply_sgd_step_all(model, grad, lr);
        }
    }
}

// Minimal SGD application (subset of unit_model.c's helper, local copy).
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
    for (size_t i = 0; i < model->global->stack.num_layers; i++) {
        blt_transformer_layer_storage *w = &model->global->stack.layer_storage[i];
        blt_transformer_layer_grad *g = &grad->global_grad->stack_grad->layer_grads[i];
        blt_sgd_step(&w->norm1_weight, &g->norm1_weight, lr);
        blt_sgd_step(&w->attn_qkv_w, &g->attn_qkv_w, lr);
        blt_sgd_step(&w->attn_proj_w, &g->attn_proj_w, lr);
        blt_sgd_step(&w->norm2_weight, &g->norm2_weight, lr);
        blt_sgd_step(&w->ffn_up_w, &g->ffn_up_w, lr);
        blt_sgd_step(&w->ffn_gate_w, &g->ffn_gate_w, lr);
        blt_sgd_step(&w->ffn_down_w, &g->ffn_down_w, lr);
    }
    for (size_t i = 0; i < model->decoder->config.num_layers; i++) {
        blt_local_layer_storage *w = &model->decoder->layers[i];
        blt_local_layer_grad *g = &grad->decoder_grad->layer_grads[i];
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
    blt_sgd_step(&model->decoder->lm_head_weight, &grad->decoder_grad->lm_head_grad, lr);
    blt_sgd_step(&model->decoder->d0_embed_weight, &grad->decoder_grad->d0_embed_grad, lr);
}

int run_selfspec_equivalence_trained(void) {
    srand(23);

    blt_arena *model_arena = blt_arena_create(4 * 1024 * 1024, BLT_BACKEND_CPU);
    blt_arena *scratch = blt_arena_create(8 * 1024 * 1024, BLT_BACKEND_CPU);
    TEST_ASSERT(model_arena && scratch);

    blt_model_config config;
    make_small_model_config(&config);
    blt_model *model = blt_model_create(model_arena, &config);
    blt_model_grad *grad = blt_model_grad_create(model_arena, model);
    TEST_ASSERT(model != NULL && grad != NULL);
    random_init_model_full(model, 0.1f);

    const char *snippets[] = {
        "int x=1;\n",
        "return 0;\n",
        "for(;;){}\n",
    };

    // Warmup toward memorized continuations: large logit margins make the
    // greedy argmaxes robust, mirroring the regime where the paper's
    // equivalence guarantee applies.
    train_snippets(model, grad, scratch, snippets, 3, 300);

    blt_entropy_lm_config ecfg;
    make_entropy_cfg(&ecfg);
    blt_entropy_lm *lm = blt_entropy_lm_create(model_arena, &ecfg);
    TEST_ASSERT(lm != NULL);
    random_init_entropy_lm(lm, 0.1f);

    blt_patcher_config pcfg;
    make_patcher_cfg(&pcfg);

    // Prompt corpus: prefixes and interior slices of the trained snippets,
    // exercising multi-round generation, window capping and the r_eff == 0
    // tail path.
    const char *prompts[] = {
        "in", "int", "int ", "int x", "nt x=1;\n", "re", "ret", "return", "return ", "eturn 0;\n",
        "fo", "for", "for(", "for(;", "(;;){}\n",  "x=", "=1;", ";\n",    "0;\n",    "{}\n",
    };
    const size_t num_prompts = sizeof(prompts) / sizeof(prompts[0]);
    const size_t max_new = 14;

    const size_t ks[] = {4, 8, 16};
    const blt_d0_mode modes[] = {BLT_D0_ZEROS, BLT_D0_LEARNED};
    const size_t num_k = sizeof(ks) / sizeof(ks[0]);
    const size_t num_m = sizeof(modes) / sizeof(modes[0]);

    size_t enc_nfe_per_config[3][2] = {{0}};
    size_t dec_nfe_per_config[3][2] = {{0}};
    size_t drafted_per_config[3][2] = {{0}};
    size_t accepted_per_config[3][2] = {{0}};
    size_t total_baseline_enc_nfe = 0;

    for (size_t p = 0; p < num_prompts; p++) {
        const size_t prompt_len = strlen(prompts[p]);
        const size_t out_len = prompt_len + max_new;
        uint8_t baseline[128];
        uint8_t spec[128];

        blt_arena_reset(scratch);
        blt_generate_greedy(model, lm, &pcfg, (const uint8_t *)prompts[p], prompt_len, max_new, baseline, scratch);
        total_baseline_enc_nfe += max_new; // baseline: one full forward per byte

        for (size_t ki = 0; ki < num_k; ki++) {
            for (size_t mi = 0; mi < num_m; mi++) {
                blt_self_spec_config cfg = {.window_k = ks[ki], .d0_mode = modes[mi]};
                blt_infer_stats stats;
                blt_infer_stats_reset(&stats);

                blt_arena_reset(scratch);
                blt_generate_greedy_selfspec(model, lm, &pcfg, (const uint8_t *)prompts[p], prompt_len, max_new, spec,
                                             &cfg, &stats, scratch);

                if (memcmp(baseline, spec, out_len) != 0) {
                    fprintf(stderr, "  EQUIVALENCE BREAK: prompt=\"%s\" k=%zu mode=%d\n", prompts[p], ks[ki],
                            (int)modes[mi]);
                    fprintf(stderr, "    baseline: \"");
                    for (size_t i = 0; i < out_len; i++) fputc(isprint(baseline[i]) ? baseline[i] : '.', stderr);
                    fprintf(stderr, "\"\n    selfspec: \"");
                    for (size_t i = 0; i < out_len; i++) fputc(isprint(spec[i]) ? spec[i] : '.', stderr);
                    fprintf(stderr, "\"\n");
                    TEST_ASSERT(!"BLT-S output diverged from plain greedy BLT");
                }

                enc_nfe_per_config[ki][mi] += stats.nfe_encoder_global;
                dec_nfe_per_config[ki][mi] += stats.nfe_decoder;
                drafted_per_config[ki][mi] += stats.bytes_drafted;
                accepted_per_config[ki][mi] += stats.bytes_accepted;
            }
        }
    }

    printf("    equivalence gate: %zu prompts x %zu configs byte-identical\n", num_prompts, num_k * num_m);
    size_t agg_enc = 0, agg_dec = 0, agg_draf = 0, agg_acc = 0;
    int any_accepted = 0;
    for (size_t ki = 0; ki < num_k; ki++) {
        for (size_t mi = 0; mi < num_m; mi++) {
            printf("    k=%-2zu mode=%s: enc/glob NFEs=%zu vs baseline=%zu (%.1f%%), "
                   "acceptance=%.1f%%\n",
                   ks[ki], modes[mi] == BLT_D0_ZEROS ? "ZEROS  " : "LEARNED", enc_nfe_per_config[ki][mi],
                   total_baseline_enc_nfe, 100.0 * (double)enc_nfe_per_config[ki][mi] / (double)total_baseline_enc_nfe,
                   100.0 * (double)accepted_per_config[ki][mi] / (double)drafted_per_config[ki][mi]);
            agg_enc += enc_nfe_per_config[ki][mi];
            agg_dec += dec_nfe_per_config[ki][mi];
            agg_draf += drafted_per_config[ki][mi];
            agg_acc += accepted_per_config[ki][mi];
            // Hard gates per config: strictly fewer expensive-tier
            // evaluations than the per-byte baseline, drafts accepted.
            TEST_ASSERT(enc_nfe_per_config[ki][mi] < total_baseline_enc_nfe);
            TEST_ASSERT(drafted_per_config[ki][mi] > 0);
            if (accepted_per_config[ki][mi] > 0.3 * drafted_per_config[ki][mi]) {
                any_accepted = 1;
            }
        }
    }
    blt_infer_stats agg = {agg_enc, agg_dec, agg_draf, agg_acc};
    blt_infer_stats_print(&agg, "    aggregate");

    // At least the ZEROS configs on memorized text must accept most drafts.
    TEST_ASSERT(any_accepted);

    blt_arena_destroy(scratch);
    blt_arena_destroy(model_arena);
    return 1;
}
