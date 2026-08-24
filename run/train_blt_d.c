// BLT-D training launcher
//
// Trains a full BLT-D pipeline (local encoder -> global transformer ->
// diffusion decoder) on a raw byte corpus with fixed-stride patches and
// per-window freshly sampled corruption (t ~ U(0,1), seeded RNG).
//
// This is the entry point for the real BLT-D vs BLT comparison runs;
// unit_block_diffusion.c's overfit gate only proves the machinery learns.
//
// Usage:
//   bin/train_blt_d --corpus FILE [options]
// Options:
//   --steps N          optimizer steps (default 2000)
//   --lr F             learning rate (default 0.05)
//   --block-size B     diffusion block size B (default 4)
//   --window W         clean-sequence length N per example (default 48)
//   --embed E          model width (default 64)
//   --hidden H         FFN width (default 128)
//   --layers L         layers per submodule (default 2)
//   --d0-mode MODE     zeros | learned (default learned)
//   --seed S           RNG seed (default 7)
//   --report-every K   print every K steps (default 25)

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "blt/core/allocator.h"
#include "blt/core/backend.h"
#include "blt/models/model.h"
#include "blt/models/checkpoint.h"
#include "blt/models/local_encoder.h"
#include "blt/models/global_transformer.h"
#include "blt/models/local_common.h"
#include "blt/models/block_diffusion.h"
#include "blt/ops/optim.h"


typedef struct {
    const char* corpus_path;
    size_t steps;
    float lr;
    size_t block_size;
    size_t window;
    size_t embed;
    size_t hidden;
    size_t layers;
    int d0_learned;
    uint64_t seed;
    size_t report_every;
    int diffusion;              // 0 = plain causal BLT baseline, 1 = BLT-D
    const char* eval_path;      // held-out corpus for causal-BPB eval
    size_t eval_windows;
    const char* save_path;      // write weights here after training/eval setup
    const char* load_path;      // load weights before training (steps=0 -> eval only)
    size_t eval_skip;           // bytes to skip before the first eval window
                                // (head-of-corpus boilerplate near-duplicates
                                // training content; skip deep for honest
                                // generalization numbers)
    float t_min;                // diffusion timestep floor (stabilizes 1/t)
    int lr_decay;               // x0.3 at 60% and 85% of steps
    size_t mask_warmup;         // ramp L_mask scale 0->1 over this many steps
    float mask_scale;           // ceiling for the L_mask weight (1.0 = paper Eq. 7)
} args_t;


static void usage(void) {
    fprintf(stderr,
        "usage: bin/train_blt_d --corpus FILE [--steps N] [--lr F] "
        "[--block-size B] [--window W] [--embed E] [--hidden H] [--layers L] "
        "[--d0-mode zeros|learned] [--seed S] [--report-every K]\n");
}


//----------------------------------------------------------------------
// Gradient clipping + SGD over every parameter in the model

static float grad_sq(blt_tensor* t) {
    const float* d = (const float*)t->data;
    float s = 0.0f;
    for (size_t i = 0; i < t->numel; i++) s += d[i] * d[i];
    return s;
}

static void layer_grads(blt_local_layer_grad* l, blt_tensor* ts[12]) {
    ts[0] = &l->norm1_weight; ts[1] = &l->attn_qkv_w; ts[2] = &l->attn_proj_w;
    ts[3] = &l->norm2_weight; ts[4] = &l->ffn_up_w; ts[5] = &l->ffn_gate_w;
    ts[6] = &l->ffn_down_w; ts[7] = &l->cross_norm_weight; ts[8] = &l->cross_weight_q;
    ts[9] = &l->cross_weight_k; ts[10] = &l->cross_weight_v; ts[11] = &l->cross_weight_proj;
}

static void clip_all(blt_model* m, blt_model_grad* g, float max_norm) {
    float sq = grad_sq(&g->encoder_grad->embedding_grad);
    for (size_t i = 0; i < m->encoder->ngram_weights.num_tables; i++)
        sq += grad_sq(&g->encoder_grad->ngram_grads.tables[i]);
    blt_tensor* ts[12];
    for (size_t i = 0; i < m->encoder->config.num_layers; i++) {
        layer_grads(&g->encoder_grad->layer_grads[i], ts);
        for (size_t j = 0; j < 12; j++) sq += grad_sq(ts[j]);
    }
    for (size_t i = 0; i < m->global->stack.num_layers; i++) {
        blt_transformer_layer_grad* l = &g->global_grad->stack_grad->layer_grads[i];
        blt_tensor* tt[7] = {&l->norm1_weight, &l->attn_qkv_w, &l->attn_proj_w,
            &l->norm2_weight, &l->ffn_up_w, &l->ffn_gate_w, &l->ffn_down_w};
        for (size_t j = 0; j < 7; j++) sq += grad_sq(tt[j]);
    }
    for (size_t i = 0; i < m->decoder->config.num_layers; i++) {
        layer_grads(&g->decoder_grad->layer_grads[i], ts);
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
        layer_grads(&g->encoder_grad->layer_grads[i], ts);
        for (size_t j = 0; j < 12; j++) blt_scale(ts[j], scale);
    }
    for (size_t i = 0; i < m->global->stack.num_layers; i++) {
        blt_transformer_layer_grad* l = &g->global_grad->stack_grad->layer_grads[i];
        blt_tensor* tt[7] = {&l->norm1_weight, &l->attn_qkv_w, &l->attn_proj_w,
            &l->norm2_weight, &l->ffn_up_w, &l->ffn_gate_w, &l->ffn_down_w};
        for (size_t j = 0; j < 7; j++) blt_scale(tt[j], scale);
    }
    for (size_t i = 0; i < m->decoder->config.num_layers; i++) {
        layer_grads(&g->decoder_grad->layer_grads[i], ts);
        for (size_t j = 0; j < 12; j++) blt_scale(ts[j], scale);
    }
    blt_scale(&g->decoder_grad->lm_head_grad, scale);
    blt_scale(&g->decoder_grad->d0_embed_grad, scale);
}

static void sgd_all(blt_model* m, blt_model_grad* g, float lr) {
    blt_local_encoder* enc = m->encoder;
    blt_local_encoder_grad* eg = g->encoder_grad;
    blt_sgd_step(&enc->byte_embedding_weight, &eg->embedding_grad, lr);
    for (size_t i = 0; i < enc->ngram_weights.num_tables; i++)
        blt_sgd_step(&enc->ngram_weights.tables[i], &eg->ngram_grads.tables[i], lr);

    blt_tensor* ws[12];
    blt_tensor* gs[12];
    for (size_t i = 0; i < enc->config.num_layers; i++) {
        blt_local_layer_storage* w = &enc->layers[i];
        ws[0] = &w->norm1_weight; ws[1] = &w->attn_qkv_w; ws[2] = &w->attn_proj_w;
        ws[3] = &w->norm2_weight; ws[4] = &w->ffn_up_w; ws[5] = &w->ffn_gate_w;
        ws[6] = &w->ffn_down_w; ws[7] = &w->cross_norm_weight; ws[8] = &w->cross_weight_q;
        ws[9] = &w->cross_weight_k; ws[10] = &w->cross_weight_v; ws[11] = &w->cross_weight_proj;
        layer_grads(&eg->layer_grads[i], gs);
        for (size_t j = 0; j < 12; j++) blt_sgd_step(ws[j], gs[j], lr);
    }
    for (size_t i = 0; i < m->global->stack.num_layers; i++) {
        blt_transformer_layer_storage* w = &m->global->stack.layer_storage[i];
        blt_transformer_layer_grad* gg = &g->global_grad->stack_grad->layer_grads[i];
        blt_tensor* tt[7] = {&w->norm1_weight, &w->attn_qkv_w, &w->attn_proj_w,
            &w->norm2_weight, &w->ffn_up_w, &w->ffn_gate_w, &w->ffn_down_w};
        blt_tensor* tg[7] = {&gg->norm1_weight, &gg->attn_qkv_w, &gg->attn_proj_w,
            &gg->norm2_weight, &gg->ffn_up_w, &gg->ffn_gate_w, &gg->ffn_down_w};
        for (size_t j = 0; j < 7; j++) blt_sgd_step(tt[j], tg[j], lr);
    }
    for (size_t i = 0; i < m->decoder->config.num_layers; i++) {
        blt_local_layer_storage* w = &m->decoder->layers[i];
        ws[0] = &w->norm1_weight; ws[1] = &w->attn_qkv_w; ws[2] = &w->attn_proj_w;
        ws[3] = &w->norm2_weight; ws[4] = &w->ffn_up_w; ws[5] = &w->ffn_gate_w;
        ws[6] = &w->ffn_down_w; ws[7] = &w->cross_norm_weight; ws[8] = &w->cross_weight_q;
        ws[9] = &w->cross_weight_k; ws[10] = &w->cross_weight_v; ws[11] = &w->cross_weight_proj;
        layer_grads(&g->decoder_grad->layer_grads[i], gs);
        for (size_t j = 0; j < 12; j++) blt_sgd_step(ws[j], gs[j], lr);
    }
    blt_sgd_step(&m->decoder->lm_head_weight, &g->decoder_grad->lm_head_grad, lr);
    blt_sgd_step(&m->decoder->d0_embed_weight, &g->decoder_grad->d0_embed_grad, lr);
}


// Mean next-byte CE (nats/byte) over the clean rows of one window, computed
// from the decoder logits. For BOTH training modes: with the Figure-5 plain-
// causal TRAIN mask, clean-row logits never depend on block rows, so this is
// the true causal BPB of the model in either case.
static double window_causal_ce(blt_arena* arena, const blt_model* model,
                               const uint8_t* text, size_t N,
                               const blt_block_batch* batch_or_null,
                               int diffusion, blt_d0_mode d0m,
                               const blt_patch_info* patches, size_t M,
                               size_t* masked_hits, size_t* masked_total) {
    size_t bshape[1] = {N};
    blt_tensor bytes_in = blt_tensor_create(arena, bshape, 1, BLT_DTYPE_UINT8);
    memcpy(bytes_in.data, text, N);

    size_t p_shape[2] = {M, model->config.encoder_config.embed_dim};
    blt_tensor P = blt_tensor_create(arena, p_shape, 2, BLT_DTYPE_FP32);
    size_t h_shape[2] = {N, model->config.encoder_config.embed_dim};
    blt_tensor h = blt_tensor_create(arena, h_shape, 2, BLT_DTYPE_FP32);
    blt_local_encoder_forward(model->encoder, &bytes_in, patches, M, NULL, 0, &P, &h, arena);

    blt_tensor O = blt_tensor_create(arena, p_shape, 2, BLT_DTYPE_FP32);
    blt_global_transformer_forward(model->global, &P, NULL, 0, &O, arena);

    const size_t V = 256;
    double ce_sum = 0.0;
    size_t count = 0;

    if (diffusion && batch_or_null != NULL) {
        size_t S = N + batch_or_null->n_block_rows;
        size_t lg[2] = {S, V};
        blt_tensor logits = blt_tensor_create(arena, lg, 2, BLT_DTYPE_FP32);
        size_t sc[1] = {1};
        blt_tensor loss = blt_tensor_create(arena, sc, 1, BLT_DTYPE_FP32);
        blt_local_decoder_forward_diffusion(model->decoder, &h, &O, patches, M,
            text, batch_or_null, d0m, &logits, &loss, arena);
        const float* L = (const float*)logits.data;

        // argmax over a row
        #define ROW_ARGMAX(row_) ({ \
            size_t best_ = 0; float bv_ = (row_)[0]; \
            for (size_t v_ = 1; v_ < V; v_++) if ((row_)[v_] > bv_) { bv_ = (row_)[v_]; best_ = v_; } \
            best_; })

        // clean rows [0..N-2] predict bytes [1..N-1]
        for (size_t i = 0; i + 1 < N; i++) {
            const float* row = L + i * V;
            float mx = row[0];
            for (size_t v = 1; v < V; v++) if (row[v] > mx) mx = row[v];
            double sum = 0.0, pt = 0.0;
            for (size_t v = 0; v < V; v++) {
                double e = exp((double)row[v] - (double)mx);
                sum += e;
                if (v == (size_t)text[i + 1]) pt = e;
            }
            ce_sum += -log(fmax(pt / sum, 1e-12));
            count++;
        }

        // masked-block cell accuracy (Phase-E drafting capability)
        if (masked_hits != NULL && masked_total != NULL) {
            for (size_t r = 0; r < batch_or_null->n_block_rows; r++) {
                if (!batch_or_null->cell_masked[r]) continue;
                const float* row = L + (N + r) * V;
                (*masked_total)++;
                if (ROW_ARGMAX(row) == (size_t)batch_or_null->targets[r]) (*masked_hits)++;
            }
        }
        #undef ROW_ARGMAX
    } else {
        size_t lg[2] = {N, V};
        blt_tensor logits = blt_tensor_create(arena, lg, 2, BLT_DTYPE_FP32);
        size_t sc[1] = {1};
        blt_tensor loss = blt_tensor_create(arena, sc, 1, BLT_DTYPE_FP32);
        blt_local_decoder_forward(model->decoder, &h, &O, patches, M,
            &bytes_in, NULL, 0, &logits, &loss, arena);
        const float* L = (const float*)logits.data;
        for (size_t i = 0; i + 1 < N; i++) {
            const float* row = L + i * V;
            float mx = row[0];
            for (size_t v = 1; v < V; v++) if (row[v] > mx) mx = row[v];
            double sum = 0.0, pt = 0.0;
            for (size_t v = 0; v < V; v++) {
                double e = exp((double)row[v] - (double)mx);
                sum += e;
                if (v == (size_t)text[i + 1]) pt = e;
            }
            ce_sum += -log(fmax(pt / sum, 1e-12));
            count++;
        }
    }
    BLT_REQUIRE(count > 0, "eval window produced no predictions");
    return ce_sum / (double)count;
}

static void fill_const1(blt_tensor* t) {
    float* d = (float*)t->data;
    for (size_t i = 0; i < t->numel; i++) d[i] = 1.0f;
}

static size_t fixed_stride(size_t seq_len, size_t patch_len, blt_patch_info* out) {
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

int main(int argc, char** argv) {
    args_t a = { .corpus_path = NULL, .steps = 2000, .lr = 0.05f,
                 .block_size = 4, .window = 48, .embed = 64, .hidden = 128,
                 .layers = 2, .d0_learned = 1, .seed = 7, .report_every = 25,
                 .diffusion = 1, .eval_path = NULL, .eval_windows = 200,
                 .save_path = NULL, .load_path = NULL,
                 .eval_skip = 0,
                 .t_min = 0.05f, .lr_decay = 0, .mask_warmup = 0,
                 .mask_scale = 1.0f };

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--corpus") && i + 1 < argc) a.corpus_path = argv[++i];
        else if (!strcmp(argv[i], "--steps") && i + 1 < argc) a.steps = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--lr") && i + 1 < argc) a.lr = atof(argv[++i]);
        else if (!strcmp(argv[i], "--block-size") && i + 1 < argc) a.block_size = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--window") && i + 1 < argc) a.window = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--embed") && i + 1 < argc) a.embed = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--hidden") && i + 1 < argc) a.hidden = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--layers") && i + 1 < argc) a.layers = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--d0-mode") && i + 1 < argc)
            a.d0_learned = !strcmp(argv[++i], "learned");
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) a.seed = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--report-every") && i + 1 < argc)
            a.report_every = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--diffusion") && i + 1 < argc)
            a.diffusion = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--save-weights") && i + 1 < argc) a.save_path = argv[++i];
        else if (!strcmp(argv[i], "--load-weights") && i + 1 < argc) a.load_path = argv[++i];
        else if (!strcmp(argv[i], "--eval-corpus") && i + 1 < argc) a.eval_path = argv[++i];
        else if (!strcmp(argv[i], "--eval-windows") && i + 1 < argc)
            a.eval_windows = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--eval-skip") && i + 1 < argc)
            a.eval_skip = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--t-min") && i + 1 < argc) a.t_min = atof(argv[++i]);
        else if (!strcmp(argv[i], "--lr-decay") && i + 1 < argc) a.lr_decay = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--mask-warmup") && i + 1 < argc)
            a.mask_warmup = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--mask-scale") && i + 1 < argc)
            a.mask_scale = atof(argv[++i]);
        else { usage(); return 1; }
    }
    if (a.corpus_path == NULL) { usage(); return 1; }

    // Load corpus
    FILE* fp = fopen(a.corpus_path, "rb");
    if (!fp) { fprintf(stderr, "cannot open corpus '%s'\n", a.corpus_path); return 1; }
    fseek(fp, 0, SEEK_END);
    long fsize = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    uint8_t* corpus = (uint8_t*)malloc((size_t)fsize);
    if (!corpus || fread(corpus, 1, (size_t)fsize, fp) != (size_t)fsize) {
        fprintf(stderr, "failed to read corpus\n");
        return 1;
    }
    fclose(fp);

    const size_t MS = 512;   // max_seq_len headroom for clean + blocks
    if (a.window < 8 || a.window > 256) {
        fprintf(stderr, "--window must be in [8, 256]\n");
        return 1;
    }

    blt_arena* model_arena = blt_arena_create(64 * 1024 * 1024, BLT_BACKEND_CPU);
    blt_arena* scratch = blt_arena_create(256 * 1024 * 1024, BLT_BACKEND_CPU);

    // Model config (paper-ish defaults; ngram tables as in the repo baseline)
    blt_model_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.encoder_config.embed_dim = a.embed; cfg.encoder_config.patch_dim = 0;
    cfg.encoder_config.num_layers = a.layers; cfg.encoder_config.hidden_dim = a.hidden;
    cfg.encoder_config.num_heads = 4; cfg.encoder_config.cross_attn_heads = 4;
    cfg.encoder_config.local_window = 0; cfg.encoder_config.cross_attn_all_layers = true;
    cfg.encoder_config.pool_type = BLT_POOL_MEAN; cfg.encoder_config.rope_theta = 500000.0f;
    cfg.encoder_config.max_seq_len = MS;
    cfg.encoder_config.ngram_config.ngram_sizes[0] = 3;
    cfg.encoder_config.ngram_config.ngram_sizes[1] = 4;
    cfg.encoder_config.ngram_config.num_ngram_sizes = 2;
    cfg.encoder_config.ngram_config.per_ngram_vocab = 50;
    cfg.encoder_config.ngram_config.hash_prime = 1000000007ULL;
    cfg.encoder_config.ngram_config.normalize = true;
    cfg.encoder_config.ngram_config.embed_dim = a.embed;
    cfg.encoder_config.pool_type = BLT_POOL_MEAN;
    cfg.global_config.embed_dim = a.embed; cfg.global_config.num_layers = a.layers;
    cfg.global_config.hidden_dim = a.hidden; cfg.global_config.num_heads = 4;
    cfg.global_config.rope_theta = 500000.0f; cfg.global_config.max_seq_len = MS;
    cfg.decoder_config.embed_dim = a.embed; cfg.decoder_config.patch_dim = 0;
    cfg.decoder_config.num_layers = a.layers; cfg.decoder_config.hidden_dim = a.hidden;
    cfg.decoder_config.num_heads = 4; cfg.decoder_config.cross_attn_heads = 4;
    cfg.decoder_config.local_window = 0; cfg.decoder_config.cross_attn_all_layers = true;
    cfg.decoder_config.rope_theta = 500000.0f; cfg.decoder_config.max_seq_len = MS;
    cfg.decoder_config.vocab_size = 256;

    blt_model* model = blt_model_create(model_arena, &cfg);
    if (a.load_path) {
        blt_model_load(model, a.load_path);
        printf("[CKPT] loaded weights from %s\n", a.load_path);
    }
    blt_model_grad* grad = blt_model_grad_create(model_arena, model);

    // Deterministic random init (libc-independent LCG). Skipped entirely
    // when --load-weights supplied the full parameter set.
    uint64_t rng_state = a.seed ? a.seed : 1;
    const int need_init = (a.load_path == NULL);
    if (need_init) {
    #define RAND01() ((float)((rng_state += 0x9E3779B97F4A7C15ULL) >> 40) / (float)(1u << 24))
    #define FILL_RAND(t, s) do { \
        float* d_ = (float*)(t).data; \
        for (size_t i_ = 0; i_ < (t).numel; i_++) d_[i_] = (RAND01() * 2.0f - 1.0f) * (s); \
    } while (0)

    FILL_RAND(model->encoder->byte_embedding_weight, 0.1f);
    for (size_t l = 0; l < cfg.encoder_config.num_layers; l++) {
        blt_local_layer_storage* ly = &model->encoder->layers[l];
        blt_tensor* ws[12] = {&ly->norm1_weight, &ly->attn_qkv_w, &ly->attn_proj_w,
            &ly->norm2_weight, &ly->ffn_up_w, &ly->ffn_gate_w, &ly->ffn_down_w,
            &ly->cross_norm_weight, &ly->cross_weight_q, &ly->cross_weight_k,
            &ly->cross_weight_v, &ly->cross_weight_proj};
        for (size_t j = 0; j < 12; j++) {
            const int is_norm = (j == 0 || j == 3 || j == 7);
            if (is_norm) { float* d = (float*)ws[j]->data;
                for (size_t i2 = 0; i2 < ws[j]->numel; i2++) d[i2] = 1.0f; }
            else FILL_RAND(*ws[j], 0.1f);
        }
    }
    for (size_t l = 0; l < cfg.global_config.num_layers; l++) {
        blt_transformer_layer_storage* ly = &model->global->stack.layer_storage[l];
        fill_const1(&ly->norm1_weight); fill_const1(&ly->norm2_weight);
        FILL_RAND(ly->attn_qkv_w, 0.1f); FILL_RAND(ly->attn_proj_w, 0.1f);
        FILL_RAND(ly->ffn_up_w, 0.1f); FILL_RAND(ly->ffn_gate_w, 0.1f);
        FILL_RAND(ly->ffn_down_w, 0.1f);
    }
    for (size_t l = 0; l < cfg.decoder_config.num_layers; l++) {
        blt_local_layer_storage* ly = &model->decoder->layers[l];
        blt_tensor* ws[12] = {&ly->norm1_weight, &ly->attn_qkv_w, &ly->attn_proj_w,
            &ly->norm2_weight, &ly->ffn_up_w, &ly->ffn_gate_w, &ly->ffn_down_w,
            &ly->cross_norm_weight, &ly->cross_weight_q, &ly->cross_weight_k,
            &ly->cross_weight_v, &ly->cross_weight_proj};
        for (size_t j = 0; j < 12; j++) {
            const int is_norm = (j == 0 || j == 3 || j == 7);
            if (is_norm) { float* d = (float*)ws[j]->data;
                for (size_t i2 = 0; i2 < ws[j]->numel; i2++) d[i2] = 1.0f; }
            else FILL_RAND(*ws[j], 0.1f);
        }
    }
    FILL_RAND(model->decoder->lm_head_weight, 0.1f);
    FILL_RAND(model->decoder->d0_embed_weight, 0.05f);
    }

    printf("train_blt_d: mode=%s corpus=%s (%ld bytes) steps=%zu lr=%.4g B=%zu W=%zu "
           "embed=%zu hidden=%zu layers=%zu d0=%s\n",
        a.diffusion ? "BLT-D" : "PLAIN-BLT",
        a.corpus_path, fsize, a.steps, (double)a.lr, a.block_size, a.window,
        a.embed, a.hidden, a.layers, a.d0_learned ? "learned" : "zeros");

    // -----------------------------------------------------------------
    // Training loop: non-overlapping windows over the corpus, fixed-stride
    // patches, freshly sampled corruption per visit.
    //
    size_t num_windows = (size_t)fsize / a.window;
    BLT_REQUIRE(num_windows >= 1, "corpus smaller than one training window");

    double running = 0.0f;
    size_t running_n = 0;

    for (size_t step = 0; step < a.steps; step++) {
        blt_arena_reset(scratch);

        const size_t w = step % num_windows;
        const uint8_t* text = corpus + w * a.window;
        const size_t N = a.window;

        blt_patch_info patches[128];
        size_t M = fixed_stride(N, 4, patches);
        if (M < 2) continue;

        blt_block_batch batch;
        if (a.diffusion) {
            blt_block_batch_build(&batch, scratch, text, N, patches, M,
                a.block_size, a.seed + step);
            // Floor the timestep: without it, rare tiny-t draws give 1/t
            // weights up to ~1e6 that dominate gradients and starve
            // L_clean (observed as wild loss swings). Standard masked-
            // diffusion stabilization; caps the weight at 1/t_min.
            if (batch.t < a.t_min) batch.t = a.t_min;
            // Optional warmup: ramp the mask-loss weight 0->1 (paper §6
            // explicitly sanctions reweighting toward next-byte prediction).
            if (a.mask_warmup > 0) {
                batch.loss_scale = (float)step / (float)a.mask_warmup;
            }
            if (batch.loss_scale > a.mask_scale) batch.loss_scale = a.mask_scale;
        }

        size_t bshape[1] = {N};
        blt_tensor bytes_in = blt_tensor_create(scratch, bshape, 1, BLT_DTYPE_UINT8);
        memcpy(bytes_in.data, text, N);

        size_t h_shape[2] = {N, a.embed};
        size_t sc_shape[1] = {1};

        zero_tensor(&grad->encoder_grad->embedding_grad);
        for (size_t i = 0; i < model->encoder->ngram_weights.num_tables; i++)
            zero_tensor(&grad->encoder_grad->ngram_grads.tables[i]);
        zero_tensor(&grad->decoder_grad->d0_embed_grad);

        if (a.diffusion) {
            size_t p_shape2[2] = {M, a.embed};
            blt_tensor P = blt_tensor_create(scratch, p_shape2, 2, BLT_DTYPE_FP32);
            blt_tensor h = blt_tensor_create(scratch, h_shape, 2, BLT_DTYPE_FP32);
            blt_local_encoder_forward(model->encoder, &bytes_in, patches, M,
                NULL, 0, &P, &h, scratch);

            blt_tensor O = blt_tensor_create(scratch, p_shape2, 2, BLT_DTYPE_FP32);
            blt_global_transformer_forward(model->global, &P, NULL, 0, &O, scratch);

            size_t S = N + batch.n_block_rows;
            size_t lg_shape[2] = {S, 256};
            blt_tensor logits = blt_tensor_create(scratch, lg_shape, 2, BLT_DTYPE_FP32);
            blt_tensor loss = blt_tensor_create(scratch, sc_shape, 1, BLT_DTYPE_FP32);
            blt_local_decoder_forward_diffusion(model->decoder, &h, &O, patches, M,
                text, &batch,
                a.d0_learned ? BLT_D0_LEARNED : BLT_D0_ZEROS, &logits, &loss, scratch);

            const float lv = ((const float*)loss.data)[0];
            running += lv;
            running_n++;

            blt_tensor grad_h = blt_tensor_create(scratch, h_shape, 2, BLT_DTYPE_FP32);
            blt_tensor grad_O = blt_tensor_create(scratch, p_shape2, 2, BLT_DTYPE_FP32);
            blt_local_decoder_backward_diffusion(model->decoder, &h, &O, patches, M,
                text, &batch,
                a.d0_learned ? BLT_D0_LEARNED : BLT_D0_ZEROS,
                &grad_h, &grad_O, grad->decoder_grad, scratch);

            blt_tensor grad_P = blt_tensor_create(scratch, p_shape2, 2, BLT_DTYPE_FP32);
            blt_global_transformer_backward(model->global, &P, NULL, 0,
                &grad_O, &grad_P, grad->global_grad, scratch);

            blt_local_encoder_backward(model->encoder, &bytes_in, patches, M,
                NULL, 0, &grad_P, &grad_h, grad->encoder_grad, scratch);
        } else {
            // plain causal BLT baseline: identical pipeline, standard
            // shifted-CE objective through the composed model API
            size_t lg_shape[2] = {N, 256};
            blt_tensor logits = blt_tensor_create(scratch, lg_shape, 2, BLT_DTYPE_FP32);
            blt_tensor loss = blt_tensor_create(scratch, sc_shape, 1, BLT_DTYPE_FP32);
            blt_model_forward(model, &bytes_in, patches, M, NULL, 0,
                &logits, &loss, scratch);

            const float lv = ((const float*)loss.data)[0];
            running += lv;
            running_n++;

            blt_model_backward(model, &bytes_in, patches, M, NULL, 0, grad, scratch);
        }

        float lr = a.lr;
        if (a.lr_decay) {
            if (step >= (a.steps * 85) / 100) lr *= 0.09f;
            else if (step >= (a.steps * 60) / 100) lr *= 0.3f;
        }

        clip_all(model, grad, 5.0f);
        sgd_all(model, grad, lr);

        if ((step + 1) % a.report_every == 0 || step + 1 == a.steps) {
            printf("step %6zu/%zu  window %zu/%zu  avg_loss %.4f\n",
                step + 1, a.steps, w + 1, num_windows, running / running_n);
            fflush(stdout);
            running = 0.0;
            running_n = 0;
        }
    }

    if (a.save_path) {
        blt_model_save(model, a.save_path);
        printf("[CKPT] saved weights to %s\n", a.save_path);
    }

    // -----------------------------------------------------------------
    // Held-out causal BPB evaluation. Same windowing/patches as training;
    // clean-row CE only -- valid for BOTH arms because the Figure-5 TRAIN
    // mask is plain causal (clean rows cannot see block rows).
    //
    double bpb = -1.0;
    if (a.eval_path != NULL) {
        FILE* ef = fopen(a.eval_path, "rb");
        if (!ef) { fprintf(stderr, "cannot open eval corpus '%s'\n", a.eval_path); return 1; }
        fseek(ef, 0, SEEK_END);
        long esz = ftell(ef);
        fseek(ef, 0, SEEK_SET);
        uint8_t* eval_bytes = (uint8_t*)malloc((size_t)esz);
        if (!eval_bytes || fread(eval_bytes, 1, (size_t)esz, ef) != (size_t)esz) {
            fprintf(stderr, "failed to read eval corpus\n");
            return 1;
        }
        fclose(ef);

        if ((size_t)esz <= a.eval_skip + a.window) {
            fprintf(stderr, "eval corpus too small for --eval-skip\n");
            return 1;
        }
        size_t avail = (size_t)esz - a.eval_skip;
        size_t eval_count = a.eval_windows;
        size_t max_windows = avail / a.window;
        if (eval_count > max_windows) eval_count = max_windows;
        uint8_t* base = eval_bytes + a.eval_skip;

        double ce_sum = 0.0;
        size_t hits = 0, total = 0;
        for (size_t wi = 0; wi < eval_count; wi++) {
            blt_arena_reset(scratch);
            const uint8_t* text = base + wi * a.window;

            blt_patch_info ep[128];
            size_t eM = fixed_stride(a.window, 4, ep);

            blt_block_batch ebatch;
            if (a.diffusion) {
                // deterministic eval corruption (never used by the CE above)
                blt_block_batch_build(&ebatch, scratch, text, a.window, ep, eM,
                    a.block_size, a.seed + 999999);
            }

            ce_sum += window_causal_ce(scratch, model, text, a.window,
                a.diffusion ? &ebatch : NULL, a.diffusion,
                a.d0_learned ? BLT_D0_LEARNED : BLT_D0_ZEROS, ep, eM,
                &hits, &total);
        }
        bpb = (ce_sum / (double)eval_count) / log(2.0);
        printf("RESULT mode=%s steps=%zu embed=%zu window=%zu B=%zu "
               "eval_windows=%zu causal_bpb=%.4f masked_acc=%.3f (%zu/%zu)\n",
            a.diffusion ? "bltd" : "plain", a.steps, a.embed, a.window,
            a.block_size, eval_count, bpb,
            total ? (double)hits / (double)total : 0.0, hits, total);
        free(eval_bytes);
    }

    free(corpus);
    return 0;
}
