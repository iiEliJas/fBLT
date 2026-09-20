#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "test_helpers.h"
#include "test_suite.h"

#include "core/allocator.h"
#include "core/backend.h"
#include "models/local_decoder.h"
#include "models/local_common.h"
#include "models/block_diffusion.h"
#include "ops/mask_builder.h"

//----------------------------------------------------------------------
// Shared helpers

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

typedef struct {
    size_t embed_dim;
    size_t hidden_dim;
    size_t vocab;
    size_t max_seq;
} tiny_dims;

static blt_local_decoder *make_random_decoder(blt_arena *arena, const tiny_dims *d, float scale) {
    blt_local_decoder_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.embed_dim = d->embed_dim;
    cfg.patch_dim = 0;
    cfg.num_layers = 1;
    cfg.hidden_dim = d->hidden_dim;
    cfg.num_heads = 2;
    cfg.cross_attn_heads = 2;
    cfg.local_window = 0;
    cfg.cross_attn_all_layers = true;
    cfg.rope_theta = 10000.0f;
    cfg.max_seq_len = d->max_seq;
    cfg.vocab_size = d->vocab;

    blt_local_decoder *dec = blt_local_decoder_create(arena, &cfg);
    TEST_ASSERT(dec != NULL);
    blt_local_layer_storage *l = &dec->layers[0];
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
    fill_small_uniform(&dec->lm_head_weight, scale);
    return dec;
}

static size_t fixed_patches(size_t seq_len, size_t patch_len, blt_patch_info *out) {
    size_t np = 0, start = 0;
    while (start < seq_len) {
        size_t len = (start + patch_len <= seq_len) ? patch_len : (seq_len - start);
        out[np].start_idx = start;
        out[np].length = len;
        out[np].peak_entropy = 0.0f;
        np++;
        start += len;
    }
    return np;
}

//----------------------------------------------------------------------
// Test 1: Figure 5 fixture. The TRAIN mask must reproduce the paper's
// matrix exactly (N=6, B=4, S=14); INFER must give clean-causal +
// fully-open block region.

int run_block_diffusion_mask_fixture(void) {
    blt_arena *arena = blt_arena_create(1024 * 1024, BLT_BACKEND_CPU);
    TEST_ASSERT(arena != NULL);

    // ---- TRAIN: Fast-BLT Figure 5 matrix verbatim ----
    const char *fig5[14] = {
        "10000000000000", "11000000000000", "11100000000000", "11110000000000", "11111000000000",
        "11111100000000", "11111110000000", "11111111000000", "11111111100000", "11111111110000",
        "11111111111000", "11111111111100", "11111111111110", "11111111111111",
    };

    blt_block_diffusion_config mc = {
        .mode = BLT_BDM_TRAIN,
        .seq_len = 14,
        .num_clean = 6,
        .block_size = 4,
    };
    blt_tensor m;
    blt_build_block_diffusion_mask(&mc, &m, arena);

    const float *md = (const float *)m.data;
    for (size_t i = 0; i < 14; i++) {
        for (size_t j = 0; j < 14; j++) {
            const int want = fig5[i][j] - '0';
            const float got = md[i * 14 + j];
            if (want && !(got == 0.0f)) {
                TEST_ASSERT(!"Fig5: expected allowed (0.0f)");
            }
            if (!want && !isinf(got)) {
                TEST_ASSERT(!"Fig5: expected blocked (-inf)");
            }
        }
    }

    // ---- INFER: clean causal + single live bidirectional block ----
    blt_block_diffusion_config mi = {
        .mode = BLT_BDM_INFER,
        .seq_len = 7,
        .num_clean = 3,
        .block_size = 0,
    };
    blt_tensor mi_m;
    blt_build_block_diffusion_mask(&mi, &mi_m, arena);
    const float *id = (const float *)mi_m.data;
    for (size_t i = 0; i < 7; i++) {
        for (size_t j = 0; j < 7; j++) {
            const bool want = (i < 3) ? (j <= i) : true;
            const float got = id[i * 7 + j];
            if (want && !(got == 0.0f)) TEST_ASSERT(!"INFER: expected allowed");
            if (!want && !isinf(got)) TEST_ASSERT(!"INFER: expected blocked");
        }
    }

    blt_arena_destroy(arena);
    return 1;
}

//----------------------------------------------------------------------
// Gradcheck harness: builds a tiny scenario, returns loss + fills analytic
// grads. Used by the numeric check below.

typedef struct {
    const uint8_t *bytes;
    size_t N;
    blt_patch_info patches[8];
    size_t num_patches;
    blt_block_batch batch;
    blt_tensor h, pin;
} gc_scenario;

static void gc_setup(blt_arena *arena, const tiny_dims *dims, gc_scenario *sc) {
    srand(41);
    // byte values must stay below the toy vocab (32) for the CE targets
    static const uint8_t seq[10] = {1, 5, 9, 2, 7, 4, 8, 3, 6, 0};
    sc->bytes = seq;
    sc->N = 10;
    sc->num_patches = fixed_patches(sc->N, 3, sc->patches); // M=4 -> 3 blocks

    size_t h_shape[2] = {sc->N, dims->embed_dim};
    sc->h = blt_tensor_create(arena, h_shape, 2, BLT_DTYPE_FP32);
    fill_small_uniform(&sc->h, 0.5f);

    size_t pi_shape[2] = {sc->num_patches, dims->embed_dim};
    sc->pin = blt_tensor_create(arena, pi_shape, 2, BLT_DTYPE_FP32);
    fill_small_uniform(&sc->pin, 0.5f);

    blt_block_batch_build(&sc->batch, arena, sc->bytes, sc->N, sc->patches, sc->num_patches, /*B=*/3, /*seed=*/42);
}

static float gc_loss(blt_arena *arena, blt_local_decoder *dec, const gc_scenario *sc, blt_d0_mode mode) {
    size_t S = sc->N + sc->batch.n_block_rows;
    size_t lg_shape[2] = {S, dec->config.vocab_size};
    blt_tensor logits = blt_tensor_create(arena, lg_shape, 2, BLT_DTYPE_FP32);
    size_t sc_shape[1] = {1};
    blt_tensor loss = blt_tensor_create(arena, sc_shape, 1, BLT_DTYPE_FP32);
    blt_local_decoder_forward_diffusion(dec, &sc->h, &sc->pin, sc->patches, sc->num_patches, sc->bytes, NULL,
                                        &sc->batch, mode, &logits, &loss, arena);
    return ((const float *)loss.data)[0];
}

//----------------------------------------------------------------------
// Test 2: numeric gradient check of blt_local_decoder_backward_diffusion.
// Central finite differences on sampled weight/input entries must match
// the analytic gradients.

int run_block_diffusion_gradcheck(void) {
    srand(40);

    blt_arena *model_arena = blt_arena_create(1024 * 1024, BLT_BACKEND_CPU);
    blt_arena *scratch = blt_arena_create(64 * 1024 * 1024, BLT_BACKEND_CPU);
    TEST_ASSERT(model_arena && scratch);

    tiny_dims dims = {.embed_dim = 8, .hidden_dim = 16, .vocab = 32, .max_seq = 48};
    blt_local_decoder *dec = make_random_decoder(model_arena, &dims, 0.2f);
    // deterministic nonzero D_0 table for the LEARNED branch
    float *tbl = (float *)dec->d0_embed_weight.data;
    for (size_t i = 0; i < dec->d0_embed_weight.numel; i++) {
        tbl[i] = (((i * 13) % 17) - 8.0f) * 0.05f;
    }

    gc_scenario sc;
    gc_setup(scratch, &dims, &sc);

    const blt_d0_mode mode = BLT_D0_LEARNED;

    blt_local_decoder_grad *grad = blt_local_decoder_grad_create(model_arena, dec);
    zero_tensor(&grad->lm_head_grad);
    zero_tensor(&grad->d0_embed_grad);
    for (size_t i = 0; i < dec->config.num_layers; i++) {
        blt_local_layer_grad *g = &grad->layer_grads[i];
        zero_tensor(&g->norm1_weight);
        zero_tensor(&g->attn_qkv_w);
        zero_tensor(&g->attn_proj_w);
        zero_tensor(&g->norm2_weight);
        zero_tensor(&g->ffn_up_w);
        zero_tensor(&g->ffn_gate_w);
        zero_tensor(&g->ffn_down_w);
        zero_tensor(&g->cross_norm_weight);
        zero_tensor(&g->cross_weight_q);
        zero_tensor(&g->cross_weight_k);
        zero_tensor(&g->cross_weight_v);
        zero_tensor(&g->cross_weight_proj);
    }

    size_t gbh_shape[2] = {sc.N, dims.embed_dim};
    blt_tensor grad_h = blt_tensor_create(scratch, gbh_shape, 2, BLT_DTYPE_FP32);
    size_t gp_shape[2] = {sc.num_patches, dims.embed_dim};
    blt_tensor grad_p = blt_tensor_create(scratch, gp_shape, 2, BLT_DTYPE_FP32);

    blt_local_decoder_backward_diffusion(dec, &sc.h, &sc.pin, sc.patches, sc.num_patches, sc.bytes, NULL, &sc.batch,
                                         mode, &grad_h, &grad_p, grad, scratch);

    TEST_ASSERT(sc.batch.t > 0.01f && sc.batch.t < 0.99f); // meaningful masking
    size_t masked_cells = 0;
    for (size_t r = 0; r < sc.batch.n_block_rows; r++) masked_cells += sc.batch.cell_masked[r];
    TEST_ASSERT(masked_cells > 0);

    // central-difference probe: (tensor, flat idx, analytic value)
    struct probe {
        blt_tensor *t;
        size_t idx;
        float analytic;
        const char *name;
    };
    struct probe probes[] = {
        {&dec->lm_head_weight, 0, 0, "lm_head[0,0]"},
        {&dec->layers[0].attn_qkv_w, 0, 0, "qkv[0,0]"},
        {&dec->layers[0].attn_proj_w, 9, 0, "proj[1,1]"},
        {&dec->layers[0].ffn_down_w, 0, 0, "ffn_down[0,0]"},
        {&dec->layers[0].ffn_gate_w, 19, 0, "ffn_gate[3,3]"},
        {&dec->layers[0].norm1_weight, 2, 0, "norm1[2]"},
        {&dec->layers[0].norm2_weight, 5, 0, "norm2[5]"},
        {&dec->layers[0].cross_weight_q, 0, 0, "xq[0,0]"},
        {&dec->layers[0].cross_norm_weight, 1, 0, "xnorm[1]"},
        {&dec->d0_embed_weight, 3 * dims.embed_dim + 4, 0, "d0_table[3,4]"},
    };

    // fill analytic values from the right grad tensors
    const blt_tensor *grad_map[] = {
        &grad->lm_head_grad,
        &grad->layer_grads[0].attn_qkv_w,
        &grad->layer_grads[0].attn_proj_w,
        &grad->layer_grads[0].ffn_down_w,
        &grad->layer_grads[0].ffn_gate_w,
        &grad->layer_grads[0].norm1_weight,
        &grad->layer_grads[0].norm2_weight,
        &grad->layer_grads[0].cross_weight_q,
        &grad->layer_grads[0].cross_norm_weight,
        &grad->d0_embed_grad,
    };
    for (size_t i = 0; i < sizeof(probes) / sizeof(probes[0]); i++) {
        probes[i].analytic = ((const float *)grad_map[i]->data)[probes[i].idx];
    }

    const float eps = 1e-3f;
    size_t failures = 0;
    for (size_t i = 0; i < sizeof(probes) / sizeof(probes[0]); i++) {
        float *w = (float *)probes[i].t->data;
        const float saved = w[probes[i].idx];

        w[probes[i].idx] = saved + eps;
        const float lp = gc_loss(scratch, dec, &sc, mode);
        w[probes[i].idx] = saved - eps;
        const float lm = gc_loss(scratch, dec, &sc, mode);
        w[probes[i].idx] = saved;

        const float numeric = (lp - lm) / (2.0f * eps);
        const float analytic = probes[i].analytic;
        const float denom = fmaxf(1.0f, fmaxf(fabsf(numeric), fabsf(analytic)));
        const float rel = fabsf(numeric - analytic) / denom;

        printf("    %-14s analytic=%+.6f numeric=%+.6f rel=%.2e\n", probes[i].name, (double)analytic, (double)numeric,
               (double)rel);
        if (rel > 2e-2f) failures++;
    }

    // input gradients: h[2][1] and patch_in[1][2]
    {
        const size_t hi = 2 * dims.embed_dim + 1;
        float *hp = (float *)sc.h.data;
        const float saved = hp[hi];
        hp[hi] = saved + eps;
        float lp = gc_loss(scratch, dec, &sc, mode);
        hp[hi] = saved - eps;
        float lm = gc_loss(scratch, dec, &sc, mode);
        hp[hi] = saved;
        float numeric = (lp - lm) / (2 * eps);
        float analytic = ((const float *)grad_h.data)[hi];
        float rel = fabsf(numeric - analytic) / fmaxf(1.0f, fmaxf(fabsf(numeric), fabsf(analytic)));
        printf("    %-14s analytic=%+.6f numeric=%+.6f rel=%.2e\n", "d/dh[2,1]", (double)analytic, (double)numeric,
               (double)rel);
        if (rel > 2e-2f) failures++;

        const size_t pi_idx = 1 * dims.embed_dim + 2;
        float *pp = (float *)sc.pin.data;
        const float sp = pp[pi_idx];
        pp[pi_idx] = sp + eps;
        lp = gc_loss(scratch, dec, &sc, mode);
        pp[pi_idx] = sp - eps;
        lm = gc_loss(scratch, dec, &sc, mode);
        pp[pi_idx] = sp;
        numeric = (lp - lm) / (2 * eps);
        analytic = ((const float *)grad_p.data)[pi_idx];
        rel = fabsf(numeric - analytic) / fmaxf(1.0f, fmaxf(fabsf(numeric), fabsf(analytic)));
        printf("    %-14s analytic=%+.6f numeric=%+.6f rel=%.2e\n", "d/dpin[1,2]", (double)analytic, (double)numeric,
               (double)rel);
        if (rel > 2e-2f) failures++;
    }

    TEST_ASSERT(failures == 0);

    blt_arena_destroy(scratch);
    blt_arena_destroy(model_arena);
    return 1;
}

//----------------------------------------------------------------------
// Test 3: fast overfit gate. A tiny full pipeline (encoder -> global ->
// diffusion decoder) trained with plain SGD must drive the combined
// L_clean + L_mask/t loss down sharply on memorized snippets. This proves
// the whole block-diffusion machinery (masks, preprocessing, forward, backward,
// D_0 table gradient) learns end-to-end. The real corpus-scale training
// comparison against plain BLT happens via bin/train_blt_d (Phase D exit).

#include "models/local_encoder.h"
#include "models/global_transformer.h"
#include "models/model.h"
#include "ops/optim.h"

static void fill_t_layer(blt_transformer_layer_storage *l, float s) {
    fill_constant(&l->norm1_weight, 1.0f);
    fill_small_uniform(&l->attn_qkv_w, s);
    fill_small_uniform(&l->attn_proj_w, s);
    fill_constant(&l->norm2_weight, 1.0f);
    fill_small_uniform(&l->ffn_up_w, s);
    fill_small_uniform(&l->ffn_gate_w, s);
    fill_small_uniform(&l->ffn_down_w, s);
}

static float grad_sq(blt_tensor *t) {
    const float *d = (const float *)t->data;
    float s = 0.0f;
    for (size_t i = 0; i < t->numel; i++) s += d[i] * d[i];
    return s;
}

// Global-norm clip across every gradient tensor (incl. d0 table).
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
    for (size_t i = 0; i < m->encoder->config.num_layers; i++) {
        blt_local_layer_grad *l = &g->encoder_grad->layer_grads[i];
        blt_tensor *ts[] = {&l->norm1_weight,   &l->attn_qkv_w,     &l->attn_proj_w,    &l->norm2_weight,
                            &l->ffn_up_w,       &l->ffn_gate_w,     &l->ffn_down_w,     &l->cross_norm_weight,
                            &l->cross_weight_q, &l->cross_weight_k, &l->cross_weight_v, &l->cross_weight_proj};
        for (size_t j = 0; j < 12; j++) blt_scale(ts[j], scale);
    }
    for (size_t i = 0; i < m->global->stack.num_layers; i++) {
        blt_transformer_layer_grad *l = &g->global_grad->stack_grad->layer_grads[i];
        blt_tensor *ts[] = {&l->norm1_weight, &l->attn_qkv_w, &l->attn_proj_w, &l->norm2_weight,
                            &l->ffn_up_w,     &l->ffn_gate_w, &l->ffn_down_w};
        for (size_t j = 0; j < 7; j++) blt_scale(ts[j], scale);
    }
    for (size_t i = 0; i < m->decoder->config.num_layers; i++) {
        blt_local_layer_grad *l = &g->decoder_grad->layer_grads[i];
        blt_tensor *ts[] = {&l->cross_norm_weight, &l->cross_weight_q, &l->cross_weight_k, &l->cross_weight_v,
                            &l->cross_weight_proj, &l->norm1_weight,   &l->attn_qkv_w,     &l->attn_proj_w,
                            &l->norm2_weight,      &l->ffn_up_w,       &l->ffn_gate_w,     &l->ffn_down_w};
        for (size_t j = 0; j < 12; j++) blt_scale(ts[j], scale);
    }
    blt_scale(&g->decoder_grad->lm_head_grad, scale);
    blt_scale(&g->decoder_grad->d0_embed_grad, scale);
}

static void sgd_all(blt_model *m, blt_model_grad *g, float lr) {
    blt_local_encoder *enc = m->encoder;
    blt_local_encoder_grad *eg = g->encoder_grad;
    blt_sgd_step(&enc->byte_embedding_weight, &eg->embedding_grad, lr);
    for (size_t i = 0; i < enc->ngram_weights.num_tables; i++)
        blt_sgd_step(&enc->ngram_weights.tables[i], &eg->ngram_grads.tables[i], lr);
    for (size_t i = 0; i < enc->config.num_layers; i++) {
        blt_local_layer_storage *w = &enc->layers[i];
        blt_local_layer_grad *gg = &eg->layer_grads[i];
        blt_tensor *ws[] = {&w->norm1_weight,   &w->attn_qkv_w,     &w->attn_proj_w,    &w->norm2_weight,
                            &w->ffn_up_w,       &w->ffn_gate_w,     &w->ffn_down_w,     &w->cross_norm_weight,
                            &w->cross_weight_q, &w->cross_weight_k, &w->cross_weight_v, &w->cross_weight_proj};
        blt_tensor *gs[] = {&gg->norm1_weight,   &gg->attn_qkv_w,     &gg->attn_proj_w,    &gg->norm2_weight,
                            &gg->ffn_up_w,       &gg->ffn_gate_w,     &gg->ffn_down_w,     &gg->cross_norm_weight,
                            &gg->cross_weight_q, &gg->cross_weight_k, &gg->cross_weight_v, &gg->cross_weight_proj};
        for (size_t j = 0; j < 12; j++) blt_sgd_step(ws[j], gs[j], lr);
    }
    for (size_t i = 0; i < m->global->stack.num_layers; i++) {
        blt_transformer_layer_storage *w = &m->global->stack.layer_storage[i];
        blt_transformer_layer_grad *gg = &g->global_grad->stack_grad->layer_grads[i];
        blt_tensor *ws[] = {&w->norm1_weight, &w->attn_qkv_w, &w->attn_proj_w, &w->norm2_weight,
                            &w->ffn_up_w,     &w->ffn_gate_w, &w->ffn_down_w};
        blt_tensor *gs[] = {&gg->norm1_weight, &gg->attn_qkv_w, &gg->attn_proj_w, &gg->norm2_weight,
                            &gg->ffn_up_w,     &gg->ffn_gate_w, &gg->ffn_down_w};
        for (size_t j = 0; j < 7; j++) blt_sgd_step(ws[j], gs[j], lr);
    }
    for (size_t i = 0; i < m->decoder->config.num_layers; i++) {
        blt_local_layer_storage *w = &m->decoder->layers[i];
        blt_local_layer_grad *gg = &g->decoder_grad->layer_grads[i];
        blt_tensor *ws[] = {&w->cross_norm_weight, &w->cross_weight_q, &w->cross_weight_k, &w->cross_weight_v,
                            &w->cross_weight_proj, &w->norm1_weight,   &w->attn_qkv_w,     &w->attn_proj_w,
                            &w->norm2_weight,      &w->ffn_up_w,       &w->ffn_gate_w,     &w->ffn_down_w};
        blt_tensor *gs[] = {&gg->cross_norm_weight, &gg->cross_weight_q, &gg->cross_weight_k, &gg->cross_weight_v,
                            &gg->cross_weight_proj, &gg->norm1_weight,   &gg->attn_qkv_w,     &gg->attn_proj_w,
                            &gg->norm2_weight,      &gg->ffn_up_w,       &gg->ffn_gate_w,     &gg->ffn_down_w};
        for (size_t j = 0; j < 12; j++) blt_sgd_step(ws[j], gs[j], lr);
    }
    blt_sgd_step(&m->decoder->lm_head_weight, &g->decoder_grad->lm_head_grad, lr);
    blt_sgd_step(&m->decoder->d0_embed_weight, &g->decoder_grad->d0_embed_grad, lr);
}

int run_block_diffusion_overfit_gate(void) {
    srand(43);

    blt_arena *model_arena = blt_arena_create(4 * 1024 * 1024, BLT_BACKEND_CPU);
    blt_arena *scratch = blt_arena_create(32 * 1024 * 1024, BLT_BACKEND_CPU);
    TEST_ASSERT(model_arena && scratch);

    // full tiny model (same recipe as unit_model.c)
    blt_model_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    const size_t E = 16, HID = 32, MS = 64;
    cfg.encoder_config.embed_dim = E;
    cfg.encoder_config.patch_dim = 0;
    cfg.encoder_config.num_layers = 1;
    cfg.encoder_config.hidden_dim = HID;
    cfg.encoder_config.num_heads = 2;
    cfg.encoder_config.cross_attn_heads = 2;
    cfg.encoder_config.local_window = 0;
    cfg.encoder_config.cross_attn_all_layers = true;
    cfg.encoder_config.pool_type = BLT_POOL_MEAN;
    cfg.encoder_config.rope_theta = 10000.0f;
    cfg.encoder_config.max_seq_len = MS;
    cfg.encoder_config.ngram_config.ngram_sizes[0] = 3;
    cfg.encoder_config.ngram_config.ngram_sizes[1] = 4;
    cfg.encoder_config.ngram_config.num_ngram_sizes = 2;
    cfg.encoder_config.ngram_config.per_ngram_vocab = 64;
    cfg.encoder_config.ngram_config.hash_prime = 1000000007ULL;
    cfg.encoder_config.ngram_config.normalize = true;
    cfg.encoder_config.ngram_config.embed_dim = E;
    cfg.global_config.embed_dim = E;
    cfg.global_config.num_layers = 1;
    cfg.global_config.hidden_dim = HID;
    cfg.global_config.num_heads = 2;
    cfg.global_config.rope_theta = 10000.0f;
    cfg.global_config.max_seq_len = MS;
    cfg.decoder_config.embed_dim = E;
    cfg.decoder_config.patch_dim = 0;
    cfg.decoder_config.num_layers = 1;
    cfg.decoder_config.hidden_dim = HID;
    cfg.decoder_config.num_heads = 2;
    cfg.decoder_config.cross_attn_heads = 2;
    cfg.decoder_config.local_window = 0;
    cfg.decoder_config.cross_attn_all_layers = true;
    cfg.decoder_config.rope_theta = 10000.0f;
    cfg.decoder_config.max_seq_len = MS;
    cfg.decoder_config.vocab_size = 256;

    blt_model *model = blt_model_create(model_arena, &cfg);
    blt_model_grad *grad = blt_model_grad_create(model_arena, model);
    TEST_ASSERT(model && grad);

    // random init everything
    fill_small_uniform(&model->encoder->byte_embedding_weight, 0.1f);
    for (size_t i = 0; i < model->encoder->config.num_layers; i++) {
        blt_local_layer_storage *l = &model->encoder->layers[i];
        fill_constant(&l->norm1_weight, 1.0f);
        fill_small_uniform(&l->attn_qkv_w, 0.1f);
        fill_small_uniform(&l->attn_proj_w, 0.1f);
        fill_constant(&l->norm2_weight, 1.0f);
        fill_small_uniform(&l->ffn_up_w, 0.1f);
        fill_small_uniform(&l->ffn_gate_w, 0.1f);
        fill_small_uniform(&l->ffn_down_w, 0.1f);
        fill_constant(&l->cross_norm_weight, 1.0f);
        fill_small_uniform(&l->cross_weight_q, 0.1f);
        fill_small_uniform(&l->cross_weight_k, 0.1f);
        fill_small_uniform(&l->cross_weight_v, 0.1f);
        fill_small_uniform(&l->cross_weight_proj, 0.1f);
    }
    for (size_t i = 0; i < model->global->stack.num_layers; i++)
        fill_t_layer(&model->global->stack.layer_storage[i], 0.1f);
    for (size_t i = 0; i < model->decoder->config.num_layers; i++) {
        blt_local_layer_storage *l = &model->decoder->layers[i];
        fill_constant(&l->cross_norm_weight, 1.0f);
        fill_small_uniform(&l->cross_weight_q, 0.1f);
        fill_small_uniform(&l->cross_weight_k, 0.1f);
        fill_small_uniform(&l->cross_weight_v, 0.1f);
        fill_small_uniform(&l->cross_weight_proj, 0.1f);
        fill_constant(&l->norm1_weight, 1.0f);
        fill_small_uniform(&l->attn_qkv_w, 0.1f);
        fill_small_uniform(&l->attn_proj_w, 0.1f);
        fill_constant(&l->norm2_weight, 1.0f);
        fill_small_uniform(&l->ffn_up_w, 0.1f);
        fill_small_uniform(&l->ffn_gate_w, 0.1f);
        fill_small_uniform(&l->ffn_down_w, 0.1f);
    }
    fill_small_uniform(&model->decoder->lm_head_weight, 0.1f);
    {
        float *tbl = (float *)model->decoder->d0_embed_weight.data;
        for (size_t i = 0; i < model->decoder->d0_embed_weight.numel; i++) tbl[i] = (((i * 13) % 17) - 8.0f) * 0.05f;
    }

    const char *snippets[] = {"int x=1;\n", "return 0;\n"};
    const size_t num_snippets = 2;

    float first_total = -1.0f, last_total = 0.0f;
    const size_t STEPS = 1200;
    const float LR = 0.1f;

    for (size_t step = 0; step < STEPS; step++) {
        float avg = 0.0f;
        for (size_t sn = 0; sn < num_snippets; sn++) {
            blt_arena_reset(scratch);
            const uint8_t *text = (const uint8_t *)snippets[sn];
            const size_t N = strlen(snippets[sn]);

            blt_patch_info patches[16];
            size_t M = fixed_patches(N, 4, patches);
            TEST_ASSERT(M >= 2);

            blt_block_batch batch;
            blt_block_batch_build(&batch, scratch, text, N, patches, M,
                                  /*B=*/4, /*seed=*/1000 + step);

            size_t bshape[1] = {N};
            blt_tensor bytes_in = blt_tensor_create(scratch, bshape, 1, BLT_DTYPE_UINT8);
            memcpy(bytes_in.data, text, N);

            // ---- forward: encoder -> global -> diffusion decoder ----
            size_t p_shape[2] = {M, E};
            blt_tensor P = blt_tensor_create(scratch, p_shape, 2, BLT_DTYPE_FP32);
            size_t h_shape[2] = {N, E};
            blt_tensor h = blt_tensor_create(scratch, h_shape, 2, BLT_DTYPE_FP32);
            blt_local_encoder_forward(model->encoder, &bytes_in, patches, M, NULL, 0, &P, &h, scratch);

            blt_tensor O = blt_tensor_create(scratch, p_shape, 2, BLT_DTYPE_FP32);
            blt_global_transformer_forward(model->global, &P, NULL, 0, &O, scratch);

            size_t S = N + batch.n_block_rows;
            size_t lg_shape[2] = {S, 256};
            blt_tensor logits = blt_tensor_create(scratch, lg_shape, 2, BLT_DTYPE_FP32);
            size_t sc_shape[1] = {1};
            blt_tensor loss = blt_tensor_create(scratch, sc_shape, 1, BLT_DTYPE_FP32);
            blt_local_decoder_forward_diffusion(model->decoder, &h, &O, patches, M, text, NULL, &batch, BLT_D0_LEARNED,
                                                &logits, &loss, scratch);

            const float lv = ((const float *)loss.data)[0];
            avg += lv;
            if (step == 0 && sn == 0) first_total = lv;

            // ---- backward: diffusion decoder -> global -> encoder ----
            zero_tensor(&grad->encoder_grad->embedding_grad);
            for (size_t i = 0; i < model->encoder->ngram_weights.num_tables; i++)
                zero_tensor(&grad->encoder_grad->ngram_grads.tables[i]);
            zero_tensor(&grad->decoder_grad->d0_embed_grad);

            blt_tensor grad_h = blt_tensor_create(scratch, h_shape, 2, BLT_DTYPE_FP32);
            blt_tensor grad_O = blt_tensor_create(scratch, p_shape, 2, BLT_DTYPE_FP32);
            blt_local_decoder_backward_diffusion(model->decoder, &h, &O, patches, M, text, NULL, &batch, BLT_D0_LEARNED,
                                                 &grad_h, &grad_O, grad->decoder_grad, scratch);

            blt_tensor grad_P = blt_tensor_create(scratch, p_shape, 2, BLT_DTYPE_FP32);
            blt_global_transformer_backward(model->global, &P, NULL, 0, &grad_O, &grad_P, grad->global_grad, scratch);

            blt_local_encoder_backward(model->encoder, &bytes_in, patches, M, NULL, 0, &grad_P, &grad_h,
                                       grad->encoder_grad, scratch);

            clip_all(model, grad, 5.0f);
            sgd_all(model, grad, LR);
        }
        last_total = avg / num_snippets;
    }

    printf("    overfit gate: first=%.3f last=%.3f\n", (double)first_total, (double)last_total);
    TEST_ASSERT(first_total > 0.0f);
    TEST_ASSERT(last_total < 0.35f * first_total); // learning happened
    TEST_ASSERT(last_total < 2.0f);                // and got low

    blt_arena_destroy(scratch);
    blt_arena_destroy(model_arena);
    return 1;
}
