// BLT-D training launcher
//
// Trains a full BLT-D pipeline (local encoder -> global transformer ->
// diffusion decoder) on a raw byte corpus with fixed-stride patches and
// per-window freshly sampled corruption.
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
#include "core/platform.h"

#include "core/allocator.h"
#include "core/backend.h"
#include "models/model.h"
#include "models/checkpoint.h"
#include "models/entropy_lm.h"
#include "models/patcher.h"
#include "models/model_builder.h"
#include "models/entropy.h"
#include "ops/softmax.h"
#include "models/local_encoder.h"
#include "models/global_transformer.h"
#include "models/local_common.h"
#include "models/block_diffusion.h"
#include "ops/optim.h"
#include "ops/vecmath.h"
#include "core/cuda_shim.h"
#include "models/param_visitor.h"
#include "train_args.h"
#include "train_optim.h"
#include "train_diag.h"
#include "train_eval.h"

//----------------------------------------------------------------------
// Gradient clipping + SGD/AdamW over every parameter in the model

// splitmix64 stream for the trainer-side timestep draw (host-side so
// curriculum schedules can reshape t before it reaches masking).
static uint64_t t_rng_next(uint64_t *state) {
    *state += 0x9E3779B97F4A7C15ULL;
    uint64_t z = *state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

int main(int argc, char **argv) {
    args_t a = parse_args(argc, argv);

    // Load corpus
    FILE *fp = fopen(a.corpus_path, "rb");
    if (!fp) {
        fprintf(stderr, "cannot open corpus '%s'\n", a.corpus_path);
        return 1;
    }
    fseek(fp, 0, SEEK_END);
    long fsize = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    uint8_t *corpus = (uint8_t *)malloc((size_t)fsize);
    if (!corpus || fread(corpus, 1, (size_t)fsize, fp) != (size_t)fsize) {
        fprintf(stderr, "failed to read corpus\n");
        return 1;
    }
    fclose(fp);

    size_t MS = 1024; // max_seq_len headroom for clean + blocks
    if (a.window < 8 || a.window > 512) {
        fprintf(stderr, "--window must be in [8, 512]\n");
        return 1;
    }
#if !defined(BLT_WITH_CUDA)
    if (a.use_cuda) {
        fprintf(stderr, "--backend cuda requires a CUDA=1 build\n");
        return 1;
    }
#endif
    // The standalone entropy-LM trainer stays CPU-only: it is a tiny
    // side model whose save path and patcher interplay are host-bound,
    // and it never benefits from the device at this size.
    if (a.train_entlm && a.use_cuda) {
        printf("[TRAIN] note: --train-entropy-lm runs on CPU regardless of "
               "--backend\n");
        a.use_cuda = 0;
    }
    const blt_backend dev = a.use_cuda ? BLT_BACKEND_CUDA : BLT_BACKEND_CPU;
    g_blt_deterministic = a.deterministic;
#ifdef BLT_WITH_CUDA
    if (a.use_cuda && a.cuda_scratch_mb > 0) blt_cuda_set_scratch_size(a.cuda_scratch_mb * 1024ULL * 1024);
#endif

    blt_arena *model_arena = blt_arena_create(a.use_cuda ? 1024ULL * 1024 * 1024 : 64 * 1024 * 1024, dev);
    blt_arena *scratch = blt_arena_create(512 * 1024 * 1024, dev);
    blt_arena *adamw_state_arena = NULL;
    if (a.optimizer == 1) adamw_state_arena = blt_arena_create(32ULL * 1024 * 1024, dev);
    // Host arena for the segmentation LM + patcher (host-only by design),
    // valid in both backends. Segmentation buffers get their own resettable
    // arena so per-step calls never overwrite the LM weights.
    blt_arena *host_lm_arena = blt_arena_create(32 * 1024 * 1024, BLT_BACKEND_CPU);
    blt_arena *host_seg_arena = blt_arena_create(16 * 1024 * 1024, BLT_BACKEND_CPU);

    // Model config (paper-ish defaults; ngram tables as in the repo baseline).
    // Per-submodule layer counts default to --layers; the decoder can be
    // scaled independently for depth sweeps (paper: decoder scaling matters
    // most for BLT-D/DV).
    if (a.enc_layers == 0) a.enc_layers = a.layers;
    if (a.glob_layers == 0) a.glob_layers = a.layers;
    if (a.dec_layers == 0) a.dec_layers = a.layers;

    blt_model_config cfg;
    blt_model_config_defaults(&cfg, a.embed, a.hidden, a.enc_layers, a.glob_layers, a.dec_layers, MS, a.cross_last);

    blt_model *model = blt_model_create(model_arena, &cfg);
    if (a.load_path) {
        blt_model_load(model, a.load_path);
        printf("[CKPT] loaded weights from %s\n", a.load_path);
    }
    blt_model_grad *grad = blt_model_grad_create(model_arena, model);
    blt_entropy_lm *train_lm = a.entropy_patches ? blt_make_entropy_lm(host_lm_arena, MS, NULL, 11) : NULL;
    if (train_lm && a.entropy_lm) {
        blt_entropy_lm_load(train_lm, a.entropy_lm);
        printf("[TRAIN] entropy-patch segmentation (trained LM: %s)\n", a.entropy_lm);
    } else if (train_lm) {
        printf("[TRAIN] entropy-patch segmentation (random LM)\n");
    }

    // Standalone entropy-LM training mode: plain byte CE on corpus
    // windows, SGD+clip, then save. Skips the main model entirely.
    if (a.train_entlm) {
        const size_t num_windows_ = (size_t)fsize / a.window;
        blt_entropy_lm *lm0 = blt_make_entropy_lm(model_arena, MS, NULL, 11);
        if (a.entropy_lm) blt_entropy_lm_load(lm0, a.entropy_lm);
        blt_entropy_lm_grad *lm_grad = blt_entropy_lm_grad_create(model_arena, lm0);
        printf("[ENTLM] training entropy LM -> %s\n", a.train_entlm);
        blt_tensor *lm_ts[9];
        blt_tensor *lm_gs[9];
        {
            blt_transformer_layer_storage *l = &lm0->stack.layer_storage[0];
            blt_transformer_layer_grad *g = &lm_grad->stack_grad->layer_grads[0];
            blt_tensor *wtmp[] = {&l->norm1_weight, &l->attn_qkv_w, &l->attn_proj_w, &l->norm2_weight,
                                  &l->ffn_up_w,     &l->ffn_gate_w, &l->ffn_down_w};
            blt_tensor *gtmp[] = {&g->norm1_weight, &g->attn_qkv_w, &g->attn_proj_w, &g->norm2_weight,
                                  &g->ffn_up_w,     &g->ffn_gate_w, &g->ffn_down_w};
            lm_ts[2] = wtmp[0];
            lm_gs[2] = gtmp[0];
            lm_ts[3] = wtmp[1];
            lm_gs[3] = gtmp[1];
            lm_ts[4] = wtmp[2];
            lm_gs[4] = gtmp[2];
            lm_ts[5] = wtmp[3];
            lm_gs[5] = gtmp[3];
            lm_ts[6] = wtmp[4];
            lm_gs[6] = gtmp[4];
            lm_ts[7] = wtmp[5];
            lm_gs[7] = gtmp[5];
            lm_ts[8] = wtmp[6];
            lm_gs[8] = gtmp[6];
        }
        lm_ts[0] = &lm0->embedding_weight;
        lm_gs[0] = &lm_grad->embedding_grad;
        lm_ts[1] = &lm0->lm_head_weight;
        lm_gs[1] = &lm_grad->lm_head_grad;

        for (size_t step = 0; step < a.steps; step++) {
            blt_arena_reset(scratch);
            const size_t w = step % num_windows_;
            const uint8_t *text = corpus + w * a.window;
            const size_t N = a.window;
            size_t sh1[1] = {N};
            blt_tensor bytes_in = blt_tensor_create(scratch, sh1, 1, BLT_DTYPE_UINT8);
            memcpy(bytes_in.data, text, N);
            size_t sh2[2] = {N, 256};
            blt_tensor logits = blt_tensor_create(scratch, sh2, 2, BLT_DTYPE_FP32);
            size_t shs[1] = {1};
            blt_tensor loss = blt_tensor_create(scratch, shs, 1, BLT_DTYPE_FP32);
            blt_entropy_lm_forward(lm0, &bytes_in, &logits, &loss, scratch);
            for (size_t ti = 0; ti < 9; ti++) zero_tensor(lm_gs[ti]);
            blt_entropy_lm_backward(lm0, &bytes_in, lm_grad, scratch);
            // global grad clip
            float sq = 0.0f;
            for (size_t ti = 0; ti < 9; ti++) {
                const float *d = (const float *)lm_gs[ti]->data;
                for (size_t j = 0; j < lm_gs[ti]->numel; j++) sq += d[j] * d[j];
            }
            const float nrm = sqrtf(sq);
            if (nrm > 5.0f && nrm > 0.0f) {
                const float scl = 5.0f / nrm;
                for (size_t ti = 0; ti < 9; ti++) blt_scale(lm_gs[ti], scl);
            }
            for (size_t ti = 0; ti < 9; ti++) blt_sgd_step(lm_ts[ti], lm_gs[ti], a.lr);
            if (a.report_every && step % a.report_every == 0)
                printf("[ENTLM] step %zu/%zu loss %.4f\n", step, a.steps, ((const float *)loss.data)[0]);
        }
        blt_entropy_lm_save(lm0, a.train_entlm);
        printf("[ENTLM] saved %s\n", a.train_entlm);
        return 0;
    }

    // Deterministic random init (libc-independent LCG). Skipped entirely
    // when --load-weights supplied the full parameter set. Values are drawn
    // in a fixed order; under a device backend they are generated host-side
    // and uploaded per tensor.
    uint64_t rng_state = a.seed ? a.seed : 1;
    const int need_init = (a.load_path == NULL);
    if (need_init) {
#define RAND01() ((float)((rng_state += 0x9E3779B97F4A7C15ULL) >> 40) / (float)(1u << 24))
#define FILL_RAND(t, s)                                                                                                \
    do {                                                                                                               \
        blt_tensor *t_ = &(t);                                                                                         \
        const size_t n_ = t_->numel;                                                                                   \
        float *d_ = (float *)(t_->backend == BLT_BACKEND_CPU ? t_->data : malloc(n_ * sizeof(float)));                 \
        BLT_REQUIRE(d_ != NULL, "init: staging alloc failed");                                                         \
        for (size_t i_ = 0; i_ < n_; i_++) d_[i_] = (RAND01() * 2.0f - 1.0f) * (s);                                    \
        if (t_->backend != BLT_BACKEND_CPU) {                                                                          \
            blt_tensor_upload(t_, d_, n_ * sizeof(float));                                                             \
            free(d_);                                                                                                  \
        }                                                                                                              \
    } while (0)
#define FILL_ONES(t)                                                                                                   \
    do {                                                                                                               \
        blt_tensor *t_ = &(t);                                                                                         \
        const size_t n_ = t_->numel;                                                                                   \
        float *d_ = (float *)(t_->backend == BLT_BACKEND_CPU ? t_->data : malloc(n_ * sizeof(float)));                 \
        BLT_REQUIRE(d_ != NULL, "init: staging alloc failed");                                                         \
        for (size_t i2_ = 0; i2_ < n_; i2_++) d_[i2_] = 1.0f;                                                          \
        if (t_->backend != BLT_BACKEND_CPU) {                                                                          \
            blt_tensor_upload(t_, d_, n_ * sizeof(float));                                                             \
            free(d_);                                                                                                  \
        }                                                                                                              \
    } while (0)

        FILL_RAND(model->encoder->byte_embedding_weight, 0.1f);
        for (size_t l = 0; l < cfg.encoder_config.num_layers; l++) {
            blt_local_layer_storage *ly = &model->encoder->layers[l];
            blt_tensor *ws[12] = {&ly->norm1_weight,   &ly->attn_qkv_w,        &ly->attn_proj_w,
                                  &ly->norm2_weight,   &ly->ffn_up_w,          &ly->ffn_gate_w,
                                  &ly->ffn_down_w,     &ly->cross_norm_weight, &ly->cross_weight_q,
                                  &ly->cross_weight_k, &ly->cross_weight_v,    &ly->cross_weight_proj};
            for (size_t j = 0; j < 12; j++) {
                const int is_norm = (j == 0 || j == 3 || j == 7);
                if (is_norm) FILL_ONES(*ws[j]);
                else FILL_RAND(*ws[j], 0.1f);
            }
        }
        for (size_t l = 0; l < cfg.global_config.num_layers; l++) {
            blt_transformer_layer_storage *ly = &model->global->stack.layer_storage[l];
            FILL_ONES(ly->norm1_weight);
            FILL_ONES(ly->norm2_weight);
            FILL_RAND(ly->attn_qkv_w, 0.1f);
            FILL_RAND(ly->attn_proj_w, 0.1f);
            FILL_RAND(ly->ffn_up_w, 0.1f);
            FILL_RAND(ly->ffn_gate_w, 0.1f);
            FILL_RAND(ly->ffn_down_w, 0.1f);
        }
        for (size_t l = 0; l < cfg.decoder_config.num_layers; l++) {
            blt_local_layer_storage *ly = &model->decoder->layers[l];
            blt_tensor *ws[12] = {&ly->norm1_weight,   &ly->attn_qkv_w,        &ly->attn_proj_w,
                                  &ly->norm2_weight,   &ly->ffn_up_w,          &ly->ffn_gate_w,
                                  &ly->ffn_down_w,     &ly->cross_norm_weight, &ly->cross_weight_q,
                                  &ly->cross_weight_k, &ly->cross_weight_v,    &ly->cross_weight_proj};
            for (size_t j = 0; j < 12; j++) {
                const int is_norm = (j == 0 || j == 3 || j == 7);
                if (is_norm) FILL_ONES(*ws[j]);
                else FILL_RAND(*ws[j], 0.1f);
            }
        }
        FILL_RAND(model->decoder->lm_head_weight, 0.1f);
        FILL_RAND(model->decoder->d0_embed_weight, 0.05f);
#undef FILL_ONES
#undef FILL_RAND
#undef RAND01
    }

    printf("train_blt_d: mode=%s corpus=%s (%ld bytes) steps=%zu lr=%.4g B=%zu W=%zu "
           "embed=%zu hidden=%zu enc/glob/dec layers=%zu/%zu/%zu xattn=%s backend=%s d0=%s"
           " optimizer=%s\n",
           a.diffusion ? "BLT-D" : "PLAIN-BLT", a.corpus_path, fsize, a.steps, (double)a.lr, a.block_size, a.window,
           a.embed, a.hidden, a.enc_layers, a.glob_layers, a.dec_layers, a.cross_last ? "last" : "all",
           a.use_cuda ? "cuda" : "cpu", a.d0_learned ? "learned" : "zeros", a.optimizer ? "adamw" : "sgd");

    // Training loop: non-overlapping windows over the corpus, fixed-stride
    // patches, freshly sampled corruption per visit.

    size_t num_windows = (size_t)fsize / a.window;
    BLT_REQUIRE(num_windows >= 1, "corpus smaller than one training window");

    size_t total_epochs = (a.steps + num_windows - 1) / num_windows;
    printf("train_blt_d: %zu windows (~%zu epochs)\n", num_windows, total_epochs);

    adamw_state *aw = NULL;
    if (a.optimizer == 1) aw = adamw_state_create(adamw_state_arena, model);

    double running = 0.0f;
    size_t running_n = 0;

    FILE *gnorm_fp = NULL;
    if (a.grad_norm_log) gnorm_fp = fopen(a.grad_norm_log, "w");
    FILE *unorm_fp = NULL;
    if (a.update_norm_log) unorm_fp = fopen(a.update_norm_log, "w");
    FILE *cnorm_fp = NULL;
    if (a.component_norm_log) cnorm_fp = fopen(a.component_norm_log, "w");
    FILE *adump_fp = NULL;
    if (a.activation_dump_log) adump_fp = fopen(a.activation_dump_log, "w");
    FILE *batch_fp = NULL;
    if (a.batch_log) batch_fp = fopen(a.batch_log, "w");
    FILE *loss_fp = NULL;
    if (a.loss_log) loss_fp = fopen(a.loss_log, "w");

    double t_fwd = 0.0, t_bwd = 0.0, t_opt = 0.0, t_step = 0.0;
    double ts0, tsA, tsB, tsC, tsD;
    const int timing = (getenv("BLT_TRAIN_TIMING") != NULL);

    double train_start = blt_time_sec();

    for (size_t step = 0; step < a.steps; step++) {
        if (timing) ts0 = blt_time_sec();
        blt_arena_reset(scratch);
#ifdef BLT_WITH_CUDA
        if (a.use_cuda) blt_cuda_scratch_reset();
#endif

        const size_t w = step % num_windows;
        const uint8_t *text = corpus + w * a.window;
        const size_t N = a.window;

        blt_patch_info patches[128];
        const size_t M = train_lm ? (blt_arena_reset(host_seg_arena),
                                     entropy_segment(host_seg_arena, train_lm, text, N, patches, 128))
                                  : fixed_stride(N, 4, patches);
        BLT_REQUIRE(M >= 2, "training window produced < 2 patches");
        if (M < 2) continue;

        blt_block_batch batch;
        if (a.diffusion) {
            uint64_t trng = a.seed * 0x9E3779B97F4A7C15ULL + step + 1;
            float t_draw = (float)((double)(t_rng_next(&trng) >> 11) / 9007199254740992.0);
            if (t_draw <= 1e-6f) t_draw = 1e-6f;
            // High-t curriculum: hold the draw above a floor that decays
            // linearly from --t-hi-start to t_min over the first
            // --t-warmup-hi fraction of steps, so early training emphasizes
            // heavily-masked blocks (the regime block drafting relies on)
            // before annealing into the full U(0,1) distribution.
            if (a.t_warmup_frac > 0.0f && a.steps > 0 && a.t_hi_start > a.t_min) {
                const size_t warm = (size_t)((double)a.steps * (double)a.t_warmup_frac);
                if (warm > 0 && step < warm) {
                    const float thr = a.t_hi_start + (a.t_min - a.t_hi_start) * ((float)step / (float)warm);
                    if (t_draw < thr) t_draw = thr;
                }
            }
            blt_block_batch_build_t(&batch, scratch, text, N, patches, M, a.block_size, a.seed + step, t_draw);
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
            float cap = a.mask_scale;
            if (a.mask_late_step > 0 && step >= a.mask_late_step && a.steps > a.mask_late_step) {
                const float frac = (float)(step - a.mask_late_step) / (float)(a.steps - a.mask_late_step);
                cap = a.mask_scale + (a.mask_late_scale - a.mask_scale) * frac;
            }
            if (batch.loss_scale > cap) batch.loss_scale = cap;
            // Log batch properties for spike investigation
            if (batch_fp) log_batch_properties(batch_fp, step, w * a.window, text, N, patches, M, &batch);
        }

        size_t bshape[1] = {N};
        blt_tensor bytes_in = blt_tensor_create(scratch, bshape, 1, BLT_DTYPE_UINT8);
        blt_tensor_upload(&bytes_in, text, N);

        size_t h_shape[2] = {N, a.embed};
        size_t sc_shape[1] = {1};

        zero_tensor(&grad->encoder_grad->embedding_grad);
        for (size_t i = 0; i < model->encoder->ngram_weights.num_tables; i++)
            zero_tensor(&grad->encoder_grad->ngram_grads.tables[i]);
        zero_tensor(&grad->decoder_grad->d0_embed_grad);
        blt_model_visit_params(model, grad, zero_norm_fn, NULL, 1);

        if (a.diffusion) {
            size_t p_shape2[2] = {M, a.embed};
            blt_tensor P = blt_tensor_create(scratch, p_shape2, 2, BLT_DTYPE_FP32);
            blt_tensor h = blt_tensor_create(scratch, h_shape, 2, BLT_DTYPE_FP32);
            if (timing) tsA = blt_time_sec();
            blt_local_encoder_forward(model->encoder, &bytes_in, patches, M, NULL, 0, &P, &h, scratch);

            blt_tensor O = blt_tensor_create(scratch, p_shape2, 2, BLT_DTYPE_FP32);
            blt_global_transformer_forward(model->global, &P, NULL, 0, &O, scratch);

            size_t S = N + batch.n_block_rows;
            size_t lg_shape[2] = {S, 256};
            blt_tensor logits = blt_tensor_create(scratch, lg_shape, 2, BLT_DTYPE_FP32);
            blt_tensor loss = blt_tensor_create(scratch, sc_shape, 1, BLT_DTYPE_FP32);
            blt_local_decoder_forward_diffusion(model->decoder, &h, &O, patches, M, text, &batch,
                                                a.d0_learned ? BLT_D0_LEARNED : BLT_D0_ZEROS, &logits, &loss, scratch);

            if (adump_fp && ((step + 1) % a.report_every == 0 || step + 1 == a.steps))
                log_forward_activation_dump(adump_fp, step + 1, &P, &h, &O, &logits);

            float lv;
            blt_tensor_download(&loss, &lv, sizeof(float));
            if (loss_fp && ((step + 1) % a.report_every == 0 || step + 1 == a.steps))
                fprintf(loss_fp, "%zu %.6f\n", step + 1, (double)lv);
            if (timing) {
                tsB = blt_time_sec();
                t_fwd += tsB - tsA;
            }
            running += lv;
            running_n++;

            blt_tensor grad_h = blt_tensor_create(scratch, h_shape, 2, BLT_DTYPE_FP32);
            blt_tensor grad_O = blt_tensor_create(scratch, p_shape2, 2, BLT_DTYPE_FP32);
            blt_local_decoder_backward_diffusion(model->decoder, &h, &O, patches, M, text, &batch,
                                                 a.d0_learned ? BLT_D0_LEARNED : BLT_D0_ZEROS, &grad_h, &grad_O,
                                                 grad->decoder_grad, scratch);

            blt_tensor grad_P = blt_tensor_create(scratch, p_shape2, 2, BLT_DTYPE_FP32);
            blt_global_transformer_backward(model->global, &P, NULL, 0, &grad_O, &grad_P, grad->global_grad, scratch);

            blt_local_encoder_backward(model->encoder, &bytes_in, patches, M, NULL, 0, &grad_P, &grad_h,
                                       grad->encoder_grad, scratch);
            if (adump_fp && ((step + 1) % a.report_every == 0 || step + 1 == a.steps))
                log_gradient_activation_dump(adump_fp, step + 1, model, grad);
            if (timing) {
                tsC = blt_time_sec();
                t_bwd += tsC - tsB;
            }
        } else {
            // plain causal BLT baseline: identical pipeline, standard
            // shifted-CE objective through the composed model API
            size_t lg_shape[2] = {N, 256};
            blt_tensor logits = blt_tensor_create(scratch, lg_shape, 2, BLT_DTYPE_FP32);
            blt_tensor loss = blt_tensor_create(scratch, sc_shape, 1, BLT_DTYPE_FP32);
            blt_model_forward(model, &bytes_in, patches, M, NULL, 0, &logits, &loss, scratch);

            if (adump_fp && ((step + 1) % a.report_every == 0 || step + 1 == a.steps))
                log_forward_activation_dump(adump_fp, step + 1, NULL, NULL, NULL, &logits);

            float lv;
            blt_tensor_download(&loss, &lv, sizeof(float));
            if (loss_fp && ((step + 1) % a.report_every == 0 || step + 1 == a.steps))
                fprintf(loss_fp, "%zu %.6f\n", step + 1, (double)lv);
            running += lv;
            running_n++;

            blt_model_backward(model, &bytes_in, patches, M, NULL, 0, grad, scratch);

            if (adump_fp && ((step + 1) % a.report_every == 0 || step + 1 == a.steps))
                log_gradient_activation_dump(adump_fp, step + 1, model, grad);
        }

        float lr = a.lr;
        if (a.lr_decay_count > 0) {
            // custom decay: multiply by factor for each decay step passed
            for (int d = 0; d < a.lr_decay_count; d++) {
                if (step >= a.lr_decay_steps[d]) lr *= a.lr_decay_factor;
            }
        } else if (a.lr_decay) {
            if (step >= (a.steps * 85) / 100) lr *= 0.09f;
            else if (step >= (a.steps * 60) / 100) lr *= 0.3f;
        }

        if (timing) tsC = blt_time_sec();
        if (cnorm_fp && ((step + 1) % a.report_every == 0 || step + 1 == a.steps))
            log_component_norms(cnorm_fp, step + 1, model, grad);
        float pre_clip_norm = clip_all(model, grad, a.max_norm);
        if (gnorm_fp && ((step + 1) % a.report_every == 0 || step + 1 == a.steps))
            fprintf(gnorm_fp, "%zu %.6f\n", step + 1, pre_clip_norm);
        if (unorm_fp && pre_clip_norm > 100.0f) {
            float post_clip = pre_clip_norm <= a.max_norm ? pre_clip_norm : a.max_norm;
            float u_norm;
            if (a.optimizer == 1) {
                blt_adamw_config ucfg = {.lr = lr,
                                         .beta1 = a.beta1,
                                         .beta2 = a.beta2,
                                         .eps = a.eps,
                                         .weight_decay = a.weight_decay,
                                         .step = 0};
                u_norm = adamw_update_norm(model, grad, aw, &ucfg);
            } else {
                u_norm = lr * post_clip;
            }
            if ((step + 1) % a.report_every == 0 || step + 1 == a.steps)
                fprintf(unorm_fp, "%zu %.6f %.6f\n", step + 1, post_clip, u_norm);
        }
        if (a.optimizer == 1) {
            blt_adamw_config cfg = {
                .lr = lr, .beta1 = a.beta1, .beta2 = a.beta2, .eps = a.eps, .weight_decay = a.weight_decay, .step = 0};
            adamw_all(model, grad, aw, &cfg);
        } else {
            sgd_all(model, grad, lr);
        }
        if (timing) {
            tsD = blt_time_sec();
            t_opt += tsD - tsC;
            t_step += tsD - ts0;
            if ((step % 50) == 49) {
                printf("[TIMING] step %zu fwd %.1fms bwd %.1fms opt %.1fms step %.1fms\n", step + 1,
                       1000.0 * t_fwd / 50.0, 1000.0 * t_bwd / 50.0, 1000.0 * t_opt / 50.0, 1000.0 * t_step / 50.0);
                fflush(stdout);
                t_fwd = t_bwd = t_opt = t_step = 0.0;
            }
        }

        if ((step + 1) % a.report_every == 0 || step + 1 == a.steps) {
            size_t epoch = step / num_windows;
            double elapsed = blt_time_sec() - train_start;
            double rate = (double)(step + 1) / elapsed;
            double remaining = (double)(a.steps - step - 1) / rate;
            size_t rem_h = (size_t)remaining / 3600;
            size_t rem_m = ((size_t)remaining % 3600) / 60;
            printf("step %6zu/%zu  epoch %zu/%zu  avg_loss %.4f  ETA %zuh%02zum\n", step + 1, a.steps, epoch,
                   total_epochs, running / running_n, rem_h, rem_m);
            fflush(stdout);
            running = 0.0;
            running_n = 0;
        }

        if (a.save_path && a.save_every > 0 && (step + 1) % a.save_every == 0) {
            blt_model_save(model, a.save_path);
            printf("[CKPT] periodic save step %zu -> %s\n", step + 1, a.save_path);
            fflush(stdout);
        }

        // Periodic eval: run causal BPB on the held-out corpus every N steps
        if (a.eval_every > 0 && (step + 1) % a.eval_every == 0 && a.eval_path != NULL) {
            FILE *ef = fopen(a.eval_path, "rb");
            if (ef) {
                fseek(ef, 0, SEEK_END);
                long esz = ftell(ef);
                fseek(ef, 0, SEEK_SET);
                uint8_t *eval_bytes = (uint8_t *)malloc((size_t)esz);
                if (eval_bytes && fread(eval_bytes, 1, (size_t)esz, ef) == (size_t)esz) {
                    size_t avail = (size_t)esz - a.eval_skip;
                    size_t eval_count = a.eval_windows;
                    size_t max_windows = avail / a.window;
                    if (eval_count > max_windows) eval_count = max_windows;
                    uint8_t *base = eval_bytes + a.eval_skip;
                    double ce_sum = 0.0;
                    size_t hits = 0, total = 0;
                    for (size_t wi = 0; wi < eval_count; wi++) {
                        blt_arena_reset(scratch);
                        const uint8_t *text = base + wi * a.window;
                        blt_patch_info ep[128];
                        size_t eM = 0;
                        if (train_lm) {
                            blt_arena_reset(host_seg_arena);
                            eM = entropy_segment(host_seg_arena, train_lm, text, a.window, ep, 128);
                        } else {
                            eM = fixed_stride(a.window, 4, ep);
                        }
                        blt_block_batch ebatch;
                        if (a.diffusion) {
                            blt_block_batch_build(&ebatch, scratch, text, a.window, ep, eM, a.block_size,
                                                  a.seed + 999999);
                        }
                        ce_sum +=
                            window_causal_ce(scratch, model, text, a.window, a.diffusion ? &ebatch : NULL, a.diffusion,
                                             a.d0_learned ? BLT_D0_LEARNED : BLT_D0_ZEROS, ep, eM, &hits, &total);
                    }
                    double bpb = (ce_sum / (double)eval_count) / log(2.0);
                    printf("EVAL step=%zu causal_bpb=%.4f masked_acc=%.3f\n", step + 1, bpb,
                           total ? (double)hits / (double)total : 0.0);
                    fflush(stdout);
                }
                free(eval_bytes);
                fclose(ef);
            }
        }
    }

    if (a.save_path) {
        blt_model_save(model, a.save_path);
        printf("[CKPT] saved weights to %s\n", a.save_path);
    }

    // Held-out causal BPB evaluation
    // Same windowing/patches as training

    double bpb = -1.0;
    if (a.eval_path != NULL) {
        FILE *ef = fopen(a.eval_path, "rb");
        if (!ef) {
            fprintf(stderr, "cannot open eval corpus '%s'\n", a.eval_path);
            return 1;
        }
        fseek(ef, 0, SEEK_END);
        long esz = ftell(ef);
        fseek(ef, 0, SEEK_SET);
        uint8_t *eval_bytes = (uint8_t *)malloc((size_t)esz);
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
        uint8_t *base = eval_bytes + a.eval_skip;

        double ce_sum = 0.0;
        size_t hits = 0, total = 0;
        for (size_t wi = 0; wi < eval_count; wi++) {
            blt_arena_reset(scratch);
            const uint8_t *text = base + wi * a.window;

            blt_patch_info ep[128];
            size_t eM = 0;
            if (train_lm) {
                blt_arena_reset(host_seg_arena);
                eM = entropy_segment(host_seg_arena, train_lm, text, a.window, ep, 128);
            } else {
                eM = fixed_stride(a.window, 4, ep);
            }

            blt_block_batch ebatch;
            if (a.diffusion) {
                // deterministic eval corruption (never used by the CE above)
                blt_block_batch_build(&ebatch, scratch, text, a.window, ep, eM, a.block_size, a.seed + 999999);
            }

            ce_sum += window_causal_ce(scratch, model, text, a.window, a.diffusion ? &ebatch : NULL, a.diffusion,
                                       a.d0_learned ? BLT_D0_LEARNED : BLT_D0_ZEROS, ep, eM, &hits, &total);
        }
        bpb = (ce_sum / (double)eval_count) / log(2.0);
        printf("RESULT mode=%s steps=%zu embed=%zu window=%zu B=%zu "
               "enc/glob/dec=%zu/%zu/%zu xattn=%s backend=%s "
               "eval_windows=%zu causal_bpb=%.4f masked_acc=%.3f (%zu/%zu)\n",
               a.diffusion ? "bltd" : "plain", a.steps, a.embed, a.window, a.block_size, a.enc_layers, a.glob_layers,
               a.dec_layers, a.cross_last ? "last" : "all", a.use_cuda ? "cuda" : "cpu", eval_count, bpb,
               total ? (double)hits / (double)total : 0.0, hits, total);
        free(eval_bytes);
    }

    blt_arena_destroy(host_lm_arena);
    blt_arena_destroy(host_seg_arena);
    blt_arena_destroy(model_arena);
    blt_arena_destroy(scratch);
    if (adamw_state_arena) blt_arena_destroy(adamw_state_arena);
    if (aw) free(aw);
    if (gnorm_fp) fclose(gnorm_fp);
    if (unorm_fp) fclose(unorm_fp);
    if (cnorm_fp) fclose(cnorm_fp);
    if (adump_fp) fclose(adump_fp);
    if (batch_fp) fclose(batch_fp);
    if (loss_fp) fclose(loss_fp);
    free(corpus);
    return 0;
}
