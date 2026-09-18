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

typedef struct {
    const char *corpus_path;
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
    int diffusion;         // 0 = plain causal BLT baseline, 1 = BLT-D
    const char *eval_path; // held-out corpus for causal-BPB eval
    size_t eval_windows;
    const char *save_path; // write weights here after training/eval setup
    size_t save_every;     // periodic checkpoint interval (0 = only at end)
    const char *load_path; // load weights before training (steps=0 -> eval only)
    size_t eval_skip;      // bytes to skip before the first eval window

    float t_min;               // diffusion timestep floor (stabilizes 1/t)
    int lr_decay;              // x0.3 at 60% and 85% of steps (default schedule)
    size_t lr_decay_steps[16]; // custom decay step positions (0 = unused)
    int lr_decay_count;        // number of custom decay points
    float lr_decay_factor;     // factor per custom decay point (default 0.3)
    size_t mask_warmup;        // ramp L_mask scale 0->1 over this many steps
    float mask_scale;          // ceiling for the L_mask weight (1.0 = paper Eq. 7)
    const char *train_entlm;   // train the entropy LM (plain CE) and save it to this path instead of the main model
                               //
    const char *entropy_lm;    // load pretrained entropy-LM weights for
                               // --entropy-patches segmentation
    size_t mask_late_step;     // raise the cap from mask_scale to
    float mask_late_scale;     // mask_late_scale between this step and the end

    int entropy_patches; // segment training/eval windows with the
                         // entropy LM + patcher (matches inference)
    int use_cuda;        // model + training passes on the CUDA backend

    size_t enc_layers; // per-submodule layer overrides (0 = --layers)
    size_t glob_layers;
    size_t dec_layers;
    int cross_last;      // cross-attn only after the final local
                         // layer (encoder + decoder); default all
    float t_warmup_frac; // high-t curriculum: fraction of steps over
    float t_hi_start;    // which the t floor decays from t_hi_start
                         // down to t_min (0 = disabled)
    int optimizer;       // 0 = sgd, 1 = adamw
    float beta1;
    float beta2;
    float eps;
    float weight_decay;
    float max_norm;                  // gradient clip threshold
    const char *grad_norm_log;       // write pre-clip grad norm every step
    const char *update_norm_log;     // write (step, post_clip_norm, update_norm) at spike steps
    const char *component_norm_log;  // write per-component gradient norms every step
    const char *activation_dump_log; // log activation stats every report-every steps
    const char *batch_log;           // log batch properties every step
    size_t eval_every;               // run causal BPB eval every N steps (0 = disabled)
    const char *loss_log;            // write per-step loss to FILE
    int deterministic;               // single-GPU bit-reproducible training
    size_t cuda_scratch_mb;          // CUDA scratch arena size in MB (default 512)
    size_t model_mb;                 // CUDA model arena size in MB (default 1024)
} args_t;

static void usage(void) {
    fprintf(stderr, "usage: bin/train_blt_d --corpus FILE [options]\n"
                    "\n"
                    "required:\n"
                    "  --corpus FILE           raw byte corpus\n"
                    "\n"
                    "training:\n"
                    "  --steps N               optimizer steps (default 2000)\n"
                    "  --lr F                  learning rate (default 0.05)\n"
                    "  --optimizer sgd|adamw   optimizer (default sgd)\n"
                    "  --beta1 F               AdamW first moment decay (default 0.9)\n"
                    "  --beta2 F               AdamW second moment decay (default 0.999)\n"
                    "  --eps F                 AdamW epsilon (default 1e-8)\n"
                    "  --weight-decay F        AdamW weight decay (default 0.01)\n"
                    "  --max-norm F            gradient clip threshold (default 5.0)\n"
                    "  --seed S                RNG seed (default 7)\n"
                    "  --backend cpu|cuda      backend (default cpu)\n"
                    "  --cuda-scratch-mb N     CUDA scratch arena size in MB (default 512)\n"
                    "  --model-mb N           CUDA model arena size in MB (default 1024)\n"
                    "  --deterministic         bit-reproducible training (slower)\n"
                    "\n"
                    "model:\n"
                    "  --embed E               model width (default 64)\n"
                    "  --hidden H              FFN width (default 128)\n"
                    "  --layers L              layers per submodule (default 2)\n"
                    "  --enc-layers N          encoder layers (overrides --layers)\n"
                    "  --glob-layers N         global transformer layers (overrides --layers)\n"
                    "  --dec-layers N          decoder layers (overrides --layers)\n"
                    "  --cross-attn all|last   cross-attention placement (default all)\n"
                    "  --d0-mode zeros|learned decoder d0 init (default learned)\n"
                    "\n"
                    "diffusion:\n"
                    "  --diffusion 0|1         enable BLT-D masked diffusion (default 1)\n"
                    "  --t-min F               diffusion timestep floor (default 0.1)\n"
                    "  --t-warmup-hi F         high-t curriculum fraction (default 0, disabled)\n"
                    "  --t-hi-start F          high-t curriculum start floor (default 0.8)\n"
                    "  --block-size B          diffusion block size (default 4)\n"
                    "  --window W              clean-sequence length (default 48)\n"
                    "\n"
                    "mask schedule:\n"
                    "  --mask-warmup N         ramp mask scale 0->1 over N steps (default 0)\n"
                    "  --mask-scale F          mask loss ceiling (default 1.0)\n"
                    "  --mask-late-step N      step to begin raising mask cap (default 0, disabled)\n"
                    "  --mask-late-scale F     final mask cap after ramp (default 1.0)\n"
                    "\n"
                    "learning rate schedule:\n"
                    "  --lr-decay 0|1          x0.3 at 60%% and 85%% of steps (default 0)\n"
                    "  --lr-decay-steps LIST   custom decay step positions (CSV)\n"
                    "  --lr-decay-factor F     factor per custom decay point (default 0.3)\n"
                    "\n"
                    "entropy patching:\n"
                    "  --entropy-patches       enable entropy-LM patcher segmentation\n"
                    "  --train-entropy-lm FILE train entropy LM and save to FILE\n"
                    "  --entropy-lm FILE       load pretrained entropy-LM weights\n"
                    "\n"
                    "evaluation:\n"
                    "  --eval-corpus FILE      held-out corpus for causal BPB eval\n"
                    "  --eval-windows N        eval window count (default 200)\n"
                    "  --eval-skip N           bytes to skip before first eval window (default 0)\n"
                    "  --eval-every N          run eval every N steps (default 0, disabled)\n"
                    "\n"
                    "I/O:\n"
                    "  --save-weights PATH     write weights after training\n"
                    "  --save-every N          save checkpoint every N steps (0 = only at end)\n"
                    "  --load-weights PATH     load weights before training\n"
                    "  --report-every K        print every K steps (default 25)\n"
                    "\n"
                    "logging (diagnostic, all optional):\n"
                    "  --grad-norm-log FILE        pre-clip gradient norms per step\n"
                    "  --update-norm-log FILE      post-clip update norms at spike steps\n"
                    "  --component-norm-log FILE   per-component gradient norms per step\n"
                    "  --activation-dump-log FILE  activation stats every report-every steps\n"
                    "  --batch-log FILE            batch properties per step\n"
                    "  --loss-log FILE             per-step loss\n");
    exit(1);
}

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

static float grad_sq(blt_tensor *t) {
    return blt_vec_dot(t->backend, (const float *)t->data, (const float *)t->data, t->numel);
}

static void layer_grads(blt_local_layer_grad *l, blt_tensor *ts[12]) {
    ts[0] = &l->norm1_weight;
    ts[1] = &l->attn_qkv_w;
    ts[2] = &l->attn_proj_w;
    ts[3] = &l->norm2_weight;
    ts[4] = &l->ffn_up_w;
    ts[5] = &l->ffn_gate_w;
    ts[6] = &l->ffn_down_w;
    ts[7] = &l->cross_norm_weight;
    ts[8] = &l->cross_weight_q;
    ts[9] = &l->cross_weight_k;
    ts[10] = &l->cross_weight_v;
    ts[11] = &l->cross_weight_proj;
}

static float clip_all(blt_model *m, blt_model_grad *g, float max_norm) {
    float sq = grad_sq(&g->encoder_grad->embedding_grad);
    for (size_t i = 0; i < m->encoder->ngram_weights.num_tables; i++)
        sq += grad_sq(&g->encoder_grad->ngram_grads.tables[i]);
    blt_tensor *ts[12];
    for (size_t i = 0; i < m->encoder->config.num_layers; i++) {
        layer_grads(&g->encoder_grad->layer_grads[i], ts);
        for (size_t j = 0; j < 12; j++) sq += grad_sq(ts[j]);
    }
    for (size_t i = 0; i < m->global->stack.num_layers; i++) {
        blt_transformer_layer_grad *l = &g->global_grad->stack_grad->layer_grads[i];
        blt_tensor *tt[7] = {&l->norm1_weight, &l->attn_qkv_w, &l->attn_proj_w, &l->norm2_weight,
                             &l->ffn_up_w,     &l->ffn_gate_w, &l->ffn_down_w};
        for (size_t j = 0; j < 7; j++) sq += grad_sq(tt[j]);
    }
    for (size_t i = 0; i < m->decoder->config.num_layers; i++) {
        layer_grads(&g->decoder_grad->layer_grads[i], ts);
        for (size_t j = 0; j < 12; j++) sq += grad_sq(ts[j]);
    }
    sq += grad_sq(&g->decoder_grad->lm_head_grad);
    sq += grad_sq(&g->decoder_grad->d0_embed_grad);

    const float norm = sqrtf(sq);
    if (norm <= max_norm || norm == 0.0f) return norm;
    const float scale = max_norm / norm;

    blt_scale(&g->encoder_grad->embedding_grad, scale);
    for (size_t i = 0; i < m->encoder->ngram_weights.num_tables; i++)
        blt_scale(&g->encoder_grad->ngram_grads.tables[i], scale);
    for (size_t i = 0; i < m->encoder->config.num_layers; i++) {
        layer_grads(&g->encoder_grad->layer_grads[i], ts);
        for (size_t j = 0; j < 12; j++) blt_scale(ts[j], scale);
    }
    for (size_t i = 0; i < m->global->stack.num_layers; i++) {
        blt_transformer_layer_grad *l = &g->global_grad->stack_grad->layer_grads[i];
        blt_tensor *tt[7] = {&l->norm1_weight, &l->attn_qkv_w, &l->attn_proj_w, &l->norm2_weight,
                             &l->ffn_up_w,     &l->ffn_gate_w, &l->ffn_down_w};
        for (size_t j = 0; j < 7; j++) blt_scale(tt[j], scale);
    }
    for (size_t i = 0; i < m->decoder->config.num_layers; i++) {
        layer_grads(&g->decoder_grad->layer_grads[i], ts);
        for (size_t j = 0; j < 12; j++) blt_scale(ts[j], scale);
    }
    blt_scale(&g->decoder_grad->lm_head_grad, scale);
    blt_scale(&g->decoder_grad->d0_embed_grad, scale);
    return norm;
}

// Log per-component L2 gradient norms to the component-norm-log file.
// Encoder split into embedding / ngram / per-layer to pinpoint
// which sub-tensor is responsible for encoder-dominated gradient spikes.
// Decoder layers split into self_attn / cross_attn / ffn for the same reason.
static void log_component_norms(FILE *fp, size_t step, blt_model *m, blt_model_grad *g) {
    float enc_embed_sq = grad_sq(&g->encoder_grad->embedding_grad);

    float enc_ngram_sq = 0.0f;
    for (size_t i = 0; i < m->encoder->ngram_weights.num_tables; i++)
        enc_ngram_sq += grad_sq(&g->encoder_grad->ngram_grads.tables[i]);

    float enc_layers_sq = 0.0f;
    blt_tensor *ts[12];
    for (size_t i = 0; i < m->encoder->config.num_layers; i++) {
        layer_grads(&g->encoder_grad->layer_grads[i], ts);
        for (size_t j = 0; j < 12; j++) enc_layers_sq += grad_sq(ts[j]);
    }

    float glob_sq = 0.0f;
    for (size_t i = 0; i < m->global->stack.num_layers; i++) {
        blt_transformer_layer_grad *l = &g->global_grad->stack_grad->layer_grads[i];
        blt_tensor *tt[7] = {&l->norm1_weight, &l->attn_qkv_w, &l->attn_proj_w, &l->norm2_weight,
                             &l->ffn_up_w,     &l->ffn_gate_w, &l->ffn_down_w};
        for (size_t j = 0; j < 7; j++) glob_sq += grad_sq(tt[j]);
    }
    const float glob_norm = sqrtf(glob_sq);

    float dec_self_sq = 0.0f, dec_cross_sq = 0.0f, dec_ffn_sq = 0.0f;
    for (size_t i = 0; i < m->decoder->config.num_layers; i++) {
        layer_grads(&g->decoder_grad->layer_grads[i], ts);
        dec_self_sq += grad_sq(ts[0]) + grad_sq(ts[1]) + grad_sq(ts[2]);
        dec_cross_sq += grad_sq(ts[7]) + grad_sq(ts[8]) + grad_sq(ts[9]) + grad_sq(ts[10]) + grad_sq(ts[11]);
        dec_ffn_sq += grad_sq(ts[3]) + grad_sq(ts[4]) + grad_sq(ts[5]) + grad_sq(ts[6]);
    }
    const float dec_self_norm = sqrtf(dec_self_sq);
    const float dec_cross_norm = sqrtf(dec_cross_sq);
    const float dec_ffn_norm = sqrtf(dec_ffn_sq);

    float head_sq = grad_sq(&g->decoder_grad->lm_head_grad) + grad_sq(&g->decoder_grad->d0_embed_grad);
    const float head_norm = sqrtf(head_sq);

    const float total_norm = sqrtf(enc_embed_sq + enc_ngram_sq + enc_layers_sq + glob_sq + dec_self_sq + dec_cross_sq +
                                   dec_ffn_sq + head_sq);

    fprintf(fp, "%zu %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n", step, sqrtf(enc_embed_sq), sqrtf(enc_ngram_sq),
            sqrtf(enc_layers_sq), glob_norm, dec_self_norm, dec_cross_norm, dec_ffn_norm, head_norm, total_norm);
}

static void sgd_all(blt_model *m, blt_model_grad *g, float lr) {
    blt_local_encoder *enc = m->encoder;
    blt_local_encoder_grad *eg = g->encoder_grad;
    blt_sgd_step(&enc->byte_embedding_weight, &eg->embedding_grad, lr);
    for (size_t i = 0; i < enc->ngram_weights.num_tables; i++)
        blt_sgd_step(&enc->ngram_weights.tables[i], &eg->ngram_grads.tables[i], lr);

    blt_tensor *ws[12];
    blt_tensor *gs[12];
    for (size_t i = 0; i < enc->config.num_layers; i++) {
        blt_local_layer_storage *w = &enc->layers[i];
        ws[0] = &w->norm1_weight;
        ws[1] = &w->attn_qkv_w;
        ws[2] = &w->attn_proj_w;
        ws[3] = &w->norm2_weight;
        ws[4] = &w->ffn_up_w;
        ws[5] = &w->ffn_gate_w;
        ws[6] = &w->ffn_down_w;
        ws[7] = &w->cross_norm_weight;
        ws[8] = &w->cross_weight_q;
        ws[9] = &w->cross_weight_k;
        ws[10] = &w->cross_weight_v;
        ws[11] = &w->cross_weight_proj;
        layer_grads(&eg->layer_grads[i], gs);
        for (size_t j = 0; j < 12; j++) blt_sgd_step(ws[j], gs[j], lr);
    }
    for (size_t i = 0; i < m->global->stack.num_layers; i++) {
        blt_transformer_layer_storage *w = &m->global->stack.layer_storage[i];
        blt_transformer_layer_grad *gg = &g->global_grad->stack_grad->layer_grads[i];
        blt_tensor *tt[7] = {&w->norm1_weight, &w->attn_qkv_w, &w->attn_proj_w, &w->norm2_weight,
                             &w->ffn_up_w,     &w->ffn_gate_w, &w->ffn_down_w};
        blt_tensor *tg[7] = {&gg->norm1_weight, &gg->attn_qkv_w, &gg->attn_proj_w, &gg->norm2_weight,
                             &gg->ffn_up_w,     &gg->ffn_gate_w, &gg->ffn_down_w};
        for (size_t j = 0; j < 7; j++) blt_sgd_step(tt[j], tg[j], lr);
    }
    for (size_t i = 0; i < m->decoder->config.num_layers; i++) {
        blt_local_layer_storage *w = &m->decoder->layers[i];
        ws[0] = &w->norm1_weight;
        ws[1] = &w->attn_qkv_w;
        ws[2] = &w->attn_proj_w;
        ws[3] = &w->norm2_weight;
        ws[4] = &w->ffn_up_w;
        ws[5] = &w->ffn_gate_w;
        ws[6] = &w->ffn_down_w;
        ws[7] = &w->cross_norm_weight;
        ws[8] = &w->cross_weight_q;
        ws[9] = &w->cross_weight_k;
        ws[10] = &w->cross_weight_v;
        ws[11] = &w->cross_weight_proj;
        layer_grads(&g->decoder_grad->layer_grads[i], gs);
        for (size_t j = 0; j < 12; j++) blt_sgd_step(ws[j], gs[j], lr);
    }
    blt_sgd_step(&m->decoder->lm_head_weight, &g->decoder_grad->lm_head_grad, lr);
    blt_sgd_step(&m->decoder->d0_embed_weight, &g->decoder_grad->d0_embed_grad, lr);
}

// AdamW state: one exp_avg + one exp_avg_sq per parameter tensor.
// Allocated once before training; zero-initialized by arena.
typedef struct {
    blt_tensor em;  // exp_avg
    blt_tensor esq; // exp_avg_sq
} adamw_pair;

typedef struct {
    adamw_pair emb;             // byte_embedding
    adamw_pair *ngram;          // [num_tables]
    adamw_pair enc_layer[128];  // 12 per encoder layer, indexed enc_layer[layer*12+j]
    adamw_pair glob_layer[128]; // 7 per global layer
    adamw_pair dec_layer[128];  // 12 per decoder layer
    adamw_pair lm_head;
    adamw_pair d0_embed;
    size_t n_enc_layers;
    size_t n_glob_layers;
    size_t n_dec_layers;
    size_t n_ngram;
    size_t step; // 1-based step counter for bias correction
} adamw_state;

static adamw_pair mk_pair(blt_arena *arena, const blt_tensor *ref) {
    adamw_pair p;
    p.em = blt_tensor_create(arena, ref->shape, ref->ndim, BLT_DTYPE_FP32);
    p.esq = blt_tensor_create(arena, ref->shape, ref->ndim, BLT_DTYPE_FP32);
    return p;
}

static adamw_state *adamw_state_create(blt_arena *arena, blt_model *m) {
    adamw_state *s = (adamw_state *)malloc(sizeof(adamw_state));
    memset(s, 0, sizeof(*s));
    s->n_enc_layers = m->encoder->config.num_layers;
    s->n_glob_layers = m->global->stack.num_layers;
    s->n_dec_layers = m->decoder->config.num_layers;
    s->n_ngram = m->encoder->ngram_weights.num_tables;
    s->step = 0;

    s->emb = mk_pair(arena, &m->encoder->byte_embedding_weight);
    s->ngram = (adamw_pair *)malloc(sizeof(adamw_pair) * s->n_ngram);
    memset(s->ngram, 0, sizeof(adamw_pair) * s->n_ngram);
    for (size_t i = 0; i < s->n_ngram; i++) s->ngram[i] = mk_pair(arena, &m->encoder->ngram_weights.tables[i]);

    for (size_t i = 0; i < s->n_enc_layers; i++) {
        blt_local_layer_storage *w = &m->encoder->layers[i];
        s->enc_layer[i * 12 + 0] = mk_pair(arena, &w->norm1_weight);
        s->enc_layer[i * 12 + 1] = mk_pair(arena, &w->attn_qkv_w);
        s->enc_layer[i * 12 + 2] = mk_pair(arena, &w->attn_proj_w);
        s->enc_layer[i * 12 + 3] = mk_pair(arena, &w->norm2_weight);
        s->enc_layer[i * 12 + 4] = mk_pair(arena, &w->ffn_up_w);
        s->enc_layer[i * 12 + 5] = mk_pair(arena, &w->ffn_gate_w);
        s->enc_layer[i * 12 + 6] = mk_pair(arena, &w->ffn_down_w);
        s->enc_layer[i * 12 + 7] = mk_pair(arena, &w->cross_norm_weight);
        s->enc_layer[i * 12 + 8] = mk_pair(arena, &w->cross_weight_q);
        s->enc_layer[i * 12 + 9] = mk_pair(arena, &w->cross_weight_k);
        s->enc_layer[i * 12 + 10] = mk_pair(arena, &w->cross_weight_v);
        s->enc_layer[i * 12 + 11] = mk_pair(arena, &w->cross_weight_proj);
    }
    for (size_t i = 0; i < s->n_glob_layers; i++) {
        blt_transformer_layer_storage *w = &m->global->stack.layer_storage[i];
        s->glob_layer[i * 7 + 0] = mk_pair(arena, &w->norm1_weight);
        s->glob_layer[i * 7 + 1] = mk_pair(arena, &w->attn_qkv_w);
        s->glob_layer[i * 7 + 2] = mk_pair(arena, &w->attn_proj_w);
        s->glob_layer[i * 7 + 3] = mk_pair(arena, &w->norm2_weight);
        s->glob_layer[i * 7 + 4] = mk_pair(arena, &w->ffn_up_w);
        s->glob_layer[i * 7 + 5] = mk_pair(arena, &w->ffn_gate_w);
        s->glob_layer[i * 7 + 6] = mk_pair(arena, &w->ffn_down_w);
    }
    for (size_t i = 0; i < s->n_dec_layers; i++) {
        blt_local_layer_storage *w = &m->decoder->layers[i];
        s->dec_layer[i * 12 + 0] = mk_pair(arena, &w->norm1_weight);
        s->dec_layer[i * 12 + 1] = mk_pair(arena, &w->attn_qkv_w);
        s->dec_layer[i * 12 + 2] = mk_pair(arena, &w->attn_proj_w);
        s->dec_layer[i * 12 + 3] = mk_pair(arena, &w->norm2_weight);
        s->dec_layer[i * 12 + 4] = mk_pair(arena, &w->ffn_up_w);
        s->dec_layer[i * 12 + 5] = mk_pair(arena, &w->ffn_gate_w);
        s->dec_layer[i * 12 + 6] = mk_pair(arena, &w->ffn_down_w);
        s->dec_layer[i * 12 + 7] = mk_pair(arena, &w->cross_norm_weight);
        s->dec_layer[i * 12 + 8] = mk_pair(arena, &w->cross_weight_q);
        s->dec_layer[i * 12 + 9] = mk_pair(arena, &w->cross_weight_k);
        s->dec_layer[i * 12 + 10] = mk_pair(arena, &w->cross_weight_v);
        s->dec_layer[i * 12 + 11] = mk_pair(arena, &w->cross_weight_proj);
    }
    s->lm_head = mk_pair(arena, &m->decoder->lm_head_weight);
    s->d0_embed = mk_pair(arena, &m->decoder->d0_embed_weight);
    return s;
}

static void adamw_step_pair(blt_tensor *param, blt_tensor *grad, adamw_pair *p, const blt_adamw_config *cfg) {
    blt_adamw_step(param, grad, &p->em, &p->esq, cfg);
}

static void adamw_all(blt_model *m, blt_model_grad *g, adamw_state *s, const blt_adamw_config *cfg) {
    s->step++;
    blt_adamw_config c = *cfg;
    c.step = s->step;

    blt_local_encoder *enc = m->encoder;
    blt_local_encoder_grad *eg = g->encoder_grad;
    adamw_step_pair(&enc->byte_embedding_weight, &eg->embedding_grad, &s->emb, &c);
    for (size_t i = 0; i < enc->ngram_weights.num_tables; i++)
        adamw_step_pair(&enc->ngram_weights.tables[i], &eg->ngram_grads.tables[i], &s->ngram[i], &c);

    blt_tensor *ws[12];
    blt_tensor *gs[12];
    for (size_t i = 0; i < enc->config.num_layers; i++) {
        blt_local_layer_storage *w = &enc->layers[i];
        ws[0] = &w->norm1_weight;
        ws[1] = &w->attn_qkv_w;
        ws[2] = &w->attn_proj_w;
        ws[3] = &w->norm2_weight;
        ws[4] = &w->ffn_up_w;
        ws[5] = &w->ffn_gate_w;
        ws[6] = &w->ffn_down_w;
        ws[7] = &w->cross_norm_weight;
        ws[8] = &w->cross_weight_q;
        ws[9] = &w->cross_weight_k;
        ws[10] = &w->cross_weight_v;
        ws[11] = &w->cross_weight_proj;
        layer_grads(&eg->layer_grads[i], gs);
        for (size_t j = 0; j < 12; j++) adamw_step_pair(ws[j], gs[j], &s->enc_layer[i * 12 + j], &c);
    }
    for (size_t i = 0; i < m->global->stack.num_layers; i++) {
        blt_transformer_layer_storage *w = &m->global->stack.layer_storage[i];
        blt_transformer_layer_grad *gg = &g->global_grad->stack_grad->layer_grads[i];
        blt_tensor *tt[7] = {&w->norm1_weight, &w->attn_qkv_w, &w->attn_proj_w, &w->norm2_weight,
                             &w->ffn_up_w,     &w->ffn_gate_w, &w->ffn_down_w};
        blt_tensor *tg[7] = {&gg->norm1_weight, &gg->attn_qkv_w, &gg->attn_proj_w, &gg->norm2_weight,
                             &gg->ffn_up_w,     &gg->ffn_gate_w, &gg->ffn_down_w};
        for (size_t j = 0; j < 7; j++) adamw_step_pair(tt[j], tg[j], &s->glob_layer[i * 7 + j], &c);
    }
    for (size_t i = 0; i < m->decoder->config.num_layers; i++) {
        blt_local_layer_storage *w = &m->decoder->layers[i];
        ws[0] = &w->norm1_weight;
        ws[1] = &w->attn_qkv_w;
        ws[2] = &w->attn_proj_w;
        ws[3] = &w->norm2_weight;
        ws[4] = &w->ffn_up_w;
        ws[5] = &w->ffn_gate_w;
        ws[6] = &w->ffn_down_w;
        ws[7] = &w->cross_norm_weight;
        ws[8] = &w->cross_weight_q;
        ws[9] = &w->cross_weight_k;
        ws[10] = &w->cross_weight_v;
        ws[11] = &w->cross_weight_proj;
        layer_grads(&g->decoder_grad->layer_grads[i], gs);
        for (size_t j = 0; j < 12; j++) adamw_step_pair(ws[j], gs[j], &s->dec_layer[i * 12 + j], &c);
    }
    adamw_step_pair(&m->decoder->lm_head_weight, &g->decoder_grad->lm_head_grad, &s->lm_head, &c);
    adamw_step_pair(&m->decoder->d0_embed_weight, &g->decoder_grad->d0_embed_grad, &s->d0_embed, &c);
}

// Compute ||update||^2 contribution from one parameter tensor (AdamW).
// update_i = m_hat_i/(sqrt(v_hat_i)+eps) + wd*p_i.
// Does NOT modify any tensor — reads only.
static float adamw_unorm_one(blt_tensor *param, blt_tensor *grad, blt_tensor *em, blt_tensor *esq, float beta1,
                             float beta2, float eps, float wd, size_t step) {
    float bc1 = 1.0f - powf(beta1, (float)step);
    float bc2 = 1.0f - powf(beta2, (float)step);
    size_t n = param->numel;
    float *pp = (float *)malloc(n * sizeof(float));
    float *gg = (float *)malloc(n * sizeof(float));
    float *mm = (float *)malloc(n * sizeof(float));
    float *vv = (float *)malloc(n * sizeof(float));
    blt_tensor_copy_to_host(param, pp, n * sizeof(float));
    blt_tensor_copy_to_host(grad, gg, n * sizeof(float));
    blt_tensor_copy_to_host(em, mm, n * sizeof(float));
    blt_tensor_copy_to_host(esq, vv, n * sizeof(float));
    float sq = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float m_new = beta1 * mm[i] + (1.0f - beta1) * gg[i];
        float v_new = beta2 * vv[i] + (1.0f - beta2) * gg[i] * gg[i];
        float m_hat = m_new / bc1;
        float v_hat = v_new / bc2;
        float u = m_hat / (sqrtf(v_hat) + eps) + wd * pp[i];
        sq += u * u;
    }
    free(pp);
    free(gg);
    free(mm);
    free(vv);
    return sq;
}

// Total ||update|| across all model parameters (AdamW). Returns ||Δp||.
static float adamw_update_norm(blt_model *m, blt_model_grad *g, adamw_state *s, const blt_adamw_config *cfg) {
    size_t step = s->step + 1;
    float b1 = cfg->beta1, b2 = cfg->beta2, ep = cfg->eps, wd = cfg->weight_decay;
    float sq = 0.0f;

    sq += adamw_unorm_one(&m->encoder->byte_embedding_weight, &g->encoder_grad->embedding_grad, &s->emb.em, &s->emb.esq,
                          b1, b2, ep, wd, step);

    for (size_t i = 0; i < m->encoder->ngram_weights.num_tables; i++)
        sq += adamw_unorm_one(&m->encoder->ngram_weights.tables[i], &g->encoder_grad->ngram_grads.tables[i],
                              &s->ngram[i].em, &s->ngram[i].esq, b1, b2, ep, wd, step);

    blt_tensor *ws[12];
    blt_tensor *gs[12];
    for (size_t i = 0; i < m->encoder->config.num_layers; i++) {
        blt_local_layer_storage *w = &m->encoder->layers[i];
        ws[0] = &w->norm1_weight;
        ws[1] = &w->attn_qkv_w;
        ws[2] = &w->attn_proj_w;
        ws[3] = &w->norm2_weight;
        ws[4] = &w->ffn_up_w;
        ws[5] = &w->ffn_gate_w;
        ws[6] = &w->ffn_down_w;
        ws[7] = &w->cross_norm_weight;
        ws[8] = &w->cross_weight_q;
        ws[9] = &w->cross_weight_k;
        ws[10] = &w->cross_weight_v;
        ws[11] = &w->cross_weight_proj;
        layer_grads(&g->encoder_grad->layer_grads[i], gs);
        for (size_t j = 0; j < 12; j++)
            sq += adamw_unorm_one(ws[j], gs[j], &s->enc_layer[i * 12 + j].em, &s->enc_layer[i * 12 + j].esq, b1, b2, ep,
                                  wd, step);
    }
    for (size_t i = 0; i < m->global->stack.num_layers; i++) {
        blt_transformer_layer_storage *w = &m->global->stack.layer_storage[i];
        blt_transformer_layer_grad *gg = &g->global_grad->stack_grad->layer_grads[i];
        blt_tensor *tt[7] = {&w->norm1_weight, &w->attn_qkv_w, &w->attn_proj_w, &w->norm2_weight,
                             &w->ffn_up_w,     &w->ffn_gate_w, &w->ffn_down_w};
        blt_tensor *tg[7] = {&gg->norm1_weight, &gg->attn_qkv_w, &gg->attn_proj_w, &gg->norm2_weight,
                             &gg->ffn_up_w,     &gg->ffn_gate_w, &gg->ffn_down_w};
        for (size_t j = 0; j < 7; j++)
            sq += adamw_unorm_one(tt[j], tg[j], &s->glob_layer[i * 7 + j].em, &s->glob_layer[i * 7 + j].esq, b1, b2, ep,
                                  wd, step);
    }
    for (size_t i = 0; i < m->decoder->config.num_layers; i++) {
        blt_local_layer_storage *w = &m->decoder->layers[i];
        ws[0] = &w->norm1_weight;
        ws[1] = &w->attn_qkv_w;
        ws[2] = &w->attn_proj_w;
        ws[3] = &w->norm2_weight;
        ws[4] = &w->ffn_up_w;
        ws[5] = &w->ffn_gate_w;
        ws[6] = &w->ffn_down_w;
        ws[7] = &w->cross_norm_weight;
        ws[8] = &w->cross_weight_q;
        ws[9] = &w->cross_weight_k;
        ws[10] = &w->cross_weight_v;
        ws[11] = &w->cross_weight_proj;
        layer_grads(&g->decoder_grad->layer_grads[i], gs);
        for (size_t j = 0; j < 12; j++)
            sq += adamw_unorm_one(ws[j], gs[j], &s->dec_layer[i * 12 + j].em, &s->dec_layer[i * 12 + j].esq, b1, b2, ep,
                                  wd, step);
    }
    sq += adamw_unorm_one(&m->decoder->lm_head_weight, &g->decoder_grad->lm_head_grad, &s->lm_head.em, &s->lm_head.esq,
                          b1, b2, ep, wd, step);
    sq += adamw_unorm_one(&m->decoder->d0_embed_weight, &g->decoder_grad->d0_embed_grad, &s->d0_embed.em,
                          &s->d0_embed.esq, b1, b2, ep, wd, step);

    return sqrtf(sq) * cfg->lr;
}

// Mean next-byte CE (nats/byte) over the clean rows of one window, computed
// from the decoder logits. For BOTH training modes: with the Figure-5 plain-
// causal TRAIN mask, clean-row logits never depend on block rows, so this is
// the true causal BPB of the model in either case.
static double window_causal_ce(blt_arena *arena, const blt_model *model, const uint8_t *text, size_t N,
                               const blt_block_batch *batch_or_null, int diffusion, blt_d0_mode d0m,
                               const blt_patch_info *patches, size_t M, size_t *masked_hits, size_t *masked_total) {
    size_t bshape[1] = {N};
    blt_tensor bytes_in = blt_tensor_create(arena, bshape, 1, BLT_DTYPE_UINT8);
    blt_tensor_upload(&bytes_in, text, N);

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
        blt_local_decoder_forward_diffusion(model->decoder, &h, &O, patches, M, text, batch_or_null, d0m, &logits,
                                            &loss, arena);
        float *L = (float *)malloc(logits.numel * sizeof(float));
        BLT_REQUIRE(L != NULL, "window_causal_ce: logits staging alloc failed");
        blt_tensor_download(&logits, L, logits.numel * sizeof(float));

// argmax over a row
#define ROW_ARGMAX(row_)                                                                                               \
    ({                                                                                                                 \
        size_t best_ = 0;                                                                                              \
        float bv_ = (row_)[0];                                                                                         \
        for (size_t v_ = 1; v_ < V; v_++)                                                                              \
            if ((row_)[v_] > bv_) {                                                                                    \
                bv_ = (row_)[v_];                                                                                      \
                best_ = v_;                                                                                            \
            }                                                                                                          \
        best_;                                                                                                         \
    })

        // clean rows [0..N-2] predict bytes [1..N-1]
        for (size_t i = 0; i + 1 < N; i++) {
            const float *row = L + i * V;
            float mx = row[0];
            for (size_t v = 1; v < V; v++)
                if (row[v] > mx) mx = row[v];
            double sum = 0.0, pt = 0.0;
            for (size_t v = 0; v < V; v++) {
                double e = exp((double)row[v] - (double)mx);
                sum += e;
                if (v == (size_t)text[i + 1]) pt = e;
            }
            ce_sum += -log(fmax(pt / sum, 1e-12));
            count++;
        }

        // masked-block cell accuracy (the capability block drafting relies on)
        if (masked_hits != NULL && masked_total != NULL) {
            for (size_t r = 0; r < batch_or_null->n_block_rows; r++) {
                if (!batch_or_null->cell_masked[r]) continue;
                const float *row = L + (N + r) * V;
                (*masked_total)++;
                if (ROW_ARGMAX(row) == (size_t)batch_or_null->targets[r]) (*masked_hits)++;
            }
        }
#undef ROW_ARGMAX
        free(L);
    } else {
        size_t lg[2] = {N, V};
        blt_tensor logits = blt_tensor_create(arena, lg, 2, BLT_DTYPE_FP32);
        size_t sc[1] = {1};
        blt_tensor loss = blt_tensor_create(arena, sc, 1, BLT_DTYPE_FP32);
        blt_local_decoder_forward(model->decoder, &h, &O, patches, M, &bytes_in, NULL, 0, &logits, &loss, arena);
        float *L = (float *)malloc(logits.numel * sizeof(float));
        BLT_REQUIRE(L != NULL, "window_causal_ce: logits staging alloc failed");
        blt_tensor_download(&logits, L, logits.numel * sizeof(float));
        for (size_t i = 0; i + 1 < N; i++) {
            const float *row = L + i * V;
            float mx = row[0];
            for (size_t v = 1; v < V; v++)
                if (row[v] > mx) mx = row[v];
            double sum = 0.0, pt = 0.0;
            for (size_t v = 0; v < V; v++) {
                double e = exp((double)row[v] - (double)mx);
                sum += e;
                if (v == (size_t)text[i + 1]) pt = e;
            }
            ce_sum += -log(fmax(pt / sum, 1e-12));
            count++;
        }
        free(L);
    }
    BLT_REQUIRE(count > 0, "eval window produced no predictions");
    return ce_sum / (double)count;
}

static size_t fixed_stride(size_t seq_len, size_t patch_len, blt_patch_info *out) {
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

// Entropy LM used for --entropy-patches. Delegates to the shared
// blt_make_entropy_lm with the same config as bench/infer_bench.c.

// Entropy-LM patching (same numerics as the inference controllers).
// The patcher is host-only, so under a device backend the byte ids are
// uploaded and the entropies staged back through host memory.
static size_t entropy_segment(blt_arena *arena, blt_entropy_lm *lm, const uint8_t *bytes, size_t len,
                              blt_patch_info *out, size_t max_patches) {
    size_t shape1[1] = {len};
    blt_tensor bytes_in = blt_tensor_create(arena, shape1, 1, BLT_DTYPE_UINT8);
    blt_tensor_upload(&bytes_in, bytes, len);

    size_t logits_shape[2] = {len, 256};
    blt_tensor logits = blt_tensor_create(arena, logits_shape, 2, BLT_DTYPE_FP32);
    size_t scalar_shape[1] = {1};
    blt_tensor discard = blt_tensor_create(arena, scalar_shape, 1, BLT_DTYPE_FP32);
    blt_entropy_lm_forward(lm, &bytes_in, &logits, &discard, arena);

    size_t probs_shape[2] = {len, 256};
    blt_tensor probs = blt_tensor_create(arena, probs_shape, 2, BLT_DTYPE_FP32);
    blt_softmax(&logits, &probs);

    size_t vals_shape[1] = {len};
    blt_tensor vals = blt_tensor_create(arena, vals_shape, 1, BLT_DTYPE_FP32);
    blt_entropy_config ecfg = {.vocab_size = 256, .use_log2 = false};
    blt_compute_entropy(&probs, &vals, &ecfg);

    blt_patcher_config pcfg;
    blt_make_patcher_cfg(&pcfg, 0, 2.5f, 1.0f, 16);

    if (vals.backend == BLT_BACKEND_CPU) {
        return blt_segment_patches(&vals, bytes, out, max_patches, &pcfg);
    }
    float *vals_host = (float *)malloc(len * sizeof(float));
    BLT_REQUIRE(vals_host != NULL, "entropy_segment: staging alloc failed");
    blt_tensor_download(&vals, vals_host, len * sizeof(float));
    blt_tensor vals_view;
    view_1d(&vals_view, vals_host, len, BLT_DTYPE_FP32, BLT_BACKEND_CPU);
    const size_t n = blt_segment_patches(&vals_view, bytes, out, max_patches, &pcfg);
    free(vals_host);
    return n;
}

//----------------------------------------------------------------------
// Activation dump: scan tensors for numerical anomalies (NaN/Inf/saturation)
// to correlate with gradient-norm spikes.

static void tensor_stats(const blt_tensor *t, float *max_abs, size_t *nan_count, size_t *inf_count) {
    *max_abs = 0.0f;
    *nan_count = 0;
    *inf_count = 0;
    if (t->numel == 0) return;

    float *host_buf = NULL;
    const float *data;
    if (t->backend == BLT_BACKEND_CPU) {
        data = (const float *)t->data;
    } else {
        host_buf = (float *)malloc(t->numel * sizeof(float));
        if (!host_buf) {
            *nan_count = t->numel;
            return;
        }
        blt_tensor_copy_to_host(t, host_buf, t->numel * sizeof(float));
        data = host_buf;
    }

    for (size_t i = 0; i < t->numel; i++) {
        float v = data[i];
        float a = fabsf(v);
        if (a > *max_abs) *max_abs = a;
        if (isnan(v)) (*nan_count)++;
        else if (isinf(v)) (*inf_count)++;
    }
    free(host_buf);
}

static void scan_log_tensor(FILE *fp, size_t step, const char *name, const blt_tensor *t) {
    float max_abs;
    size_t nan_count, inf_count;
    tensor_stats(t, &max_abs, &nan_count, &inf_count);
    fprintf(fp, "%zu %s %.6f %zu %zu\n", step, name, max_abs, nan_count, inf_count);
}

// Log forward activation stats (before backward). NULL tensors are skipped.
static void log_forward_activation_dump(FILE *fp, size_t step, const blt_tensor *P, const blt_tensor *h,
                                        const blt_tensor *O, const blt_tensor *logits) {
    if (P) scan_log_tensor(fp, step, "fwd/P", P);
    if (h) scan_log_tensor(fp, step, "fwd/h", h);
    if (O) scan_log_tensor(fp, step, "fwd/O", O);
    if (logits) scan_log_tensor(fp, step, "fwd/logits", logits);
}

// Log gradient activation stats (after backward). Scans decoder head,
// decoder layers, global layers, and encoder gradients.
static void log_gradient_activation_dump(FILE *fp, size_t step, blt_model *m, blt_model_grad *g) {
    // Decoder head gradients
    scan_log_tensor(fp, step, "grad/dec_lm_head", &g->decoder_grad->lm_head_grad);
    scan_log_tensor(fp, step, "grad/dec_d0_embed", &g->decoder_grad->d0_embed_grad);

    // Decoder layer gradients: max across all layers and weight tensors
    {
        float layer_max = 0.0f;
        size_t layer_nan = 0, layer_inf = 0;
        for (size_t i = 0; i < m->decoder->config.num_layers; i++) {
            blt_tensor *ts[12];
            layer_grads(&g->decoder_grad->layer_grads[i], ts);
            for (size_t j = 0; j < 12; j++) {
                float mx;
                size_t nn, ni;
                tensor_stats(ts[j], &mx, &nn, &ni);
                if (mx > layer_max) layer_max = mx;
                layer_nan += nn;
                layer_inf += ni;
            }
        }
        fprintf(fp, "%zu grad/dec_layers %.6f %zu %zu\n", step, layer_max, layer_nan, layer_inf);
    }

    // Global transformer gradients
    {
        float layer_max = 0.0f;
        size_t layer_nan = 0, layer_inf = 0;
        for (size_t i = 0; i < m->global->stack.num_layers; i++) {
            blt_transformer_layer_grad *l = &g->global_grad->stack_grad->layer_grads[i];
            blt_tensor *tt[7] = {&l->norm1_weight, &l->attn_qkv_w, &l->attn_proj_w, &l->norm2_weight,
                                 &l->ffn_up_w,     &l->ffn_gate_w, &l->ffn_down_w};
            for (size_t j = 0; j < 7; j++) {
                float mx;
                size_t nn, ni;
                tensor_stats(tt[j], &mx, &nn, &ni);
                if (mx > layer_max) layer_max = mx;
                layer_nan += nn;
                layer_inf += ni;
            }
        }
        fprintf(fp, "%zu grad/glob_layers %.6f %zu %zu\n", step, layer_max, layer_nan, layer_inf);
    }

    // Encoder gradients
    {
        float enc_max = 0.0f;
        size_t enc_nan = 0, enc_inf = 0;
        float mx;
        size_t nn, ni;

        tensor_stats(&g->encoder_grad->embedding_grad, &mx, &nn, &ni);
        if (mx > enc_max) enc_max = mx;
        enc_nan += nn;
        enc_inf += ni;

        for (size_t i = 0; i < m->encoder->ngram_weights.num_tables; i++) {
            tensor_stats(&g->encoder_grad->ngram_grads.tables[i], &mx, &nn, &ni);
            if (mx > enc_max) enc_max = mx;
            enc_nan += nn;
            enc_inf += ni;
        }

        for (size_t i = 0; i < m->encoder->config.num_layers; i++) {
            blt_tensor *ts[12];
            layer_grads(&g->encoder_grad->layer_grads[i], ts);
            for (size_t j = 0; j < 12; j++) {
                tensor_stats(ts[j], &mx, &nn, &ni);
                if (mx > enc_max) enc_max = mx;
                enc_nan += nn;
                enc_inf += ni;
            }
        }
        fprintf(fp, "%zu grad/encoder %.6f %zu %zu\n", step, enc_max, enc_nan, enc_inf);
    }
}

//----------------------------------------------------------------------
// Batch property logging: record per-step input characteristics so we
// can check whether gradient-norm spike steps share unusual inputs.

static uint64_t fnv1a_hash(const uint8_t *data, size_t len) {
    uint64_t hash = 14695981039346656037ULL;
    for (size_t i = 0; i < len; i++) {
        hash ^= data[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

// Log per-step batch properties: window identity, patch stats, diffusion
// corruption stats, and the effective loss weight. Space-separated, one
// line per step.
static void log_batch_properties(FILE *fp, size_t step, size_t window_offset, const uint8_t *text, size_t window_size,
                                 const blt_patch_info *patches, size_t num_patches, const blt_block_batch *batch) {

    const uint64_t whash = fnv1a_hash(text, window_size);

    // average patch length
    float avg_patch_len = 0.0f;
    if (num_patches > 0) {
        size_t total = 0;
        for (size_t i = 0; i < num_patches; i++) total += patches[i].length;
        avg_patch_len = (float)total / (float)num_patches;
    }

    // count valid and masked cells in the diffusion batch
    size_t n_valid = 0, n_masked = 0;
    if (batch != NULL) {
        for (size_t r = 0; r < batch->n_block_rows; r++) {
            if (batch->cell_valid[r]) n_valid++;
            if (batch->cell_masked[r]) n_masked++;
        }
    }
    const float masked_frac = n_valid > 0 ? (float)n_masked / (float)n_valid : 0.0f;

    const float t = batch != NULL ? batch->t : 0.0f;
    const float loss_scale = batch != NULL ? batch->loss_scale : 0.0f;
    const float eff_weight = (t > 0.0f) ? loss_scale / t : 0.0f;

    fprintf(fp, "%zu %zu %016llx %zu %.3f %zu %zu %.4f %.6f %.6f %.6f\n", step, window_offset,
            (unsigned long long)whash, num_patches, avg_patch_len, n_valid, n_masked, masked_frac, t, loss_scale,
            eff_weight);
}

int main(int argc, char **argv) {
    args_t a = {.corpus_path = NULL,
                .steps = 2000,
                .lr = 0.05f,
                .block_size = 4,
                .window = 48,
                .embed = 64,
                .hidden = 128,
                .layers = 2,
                .d0_learned = 1,
                .seed = 7,
                .report_every = 25,
                .diffusion = 1,
                .eval_path = NULL,
                .eval_windows = 200,
                .save_path = NULL,
                .save_every = 0,
                .load_path = NULL,
                .eval_skip = 0,
                .t_min = 0.1f,
                .lr_decay = 0,
                .lr_decay_count = 0,
                .lr_decay_factor = 0.3f,
                .mask_warmup = 0,
                .mask_scale = 1.0f,
                .mask_late_step = 0,
                .mask_late_scale = 1.0f,
                .entropy_patches = 0,
                .train_entlm = NULL,
                .entropy_lm = NULL,
                .use_cuda = 0,
                .enc_layers = 0,
                .glob_layers = 0,
                .dec_layers = 0,
                .cross_last = 0,
                .t_warmup_frac = 0.0f,
                .t_hi_start = 0.8f,
                .optimizer = 0,
                .beta1 = 0.9f,
                .beta2 = 0.999f,
                .eps = 1e-8f,
                .weight_decay = 0.01f,
                .max_norm = 5.0f,
                .grad_norm_log = NULL,
                .update_norm_log = NULL,
                .component_norm_log = NULL,
                .activation_dump_log = NULL,
                .batch_log = NULL,
                .eval_every = 0,
                .loss_log = NULL,
                .deterministic = 0,
                .cuda_scratch_mb = 0,
                .model_mb = 0};

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--corpus") && i + 1 < argc) a.corpus_path = argv[++i];
        else if (!strcmp(argv[i], "--steps") && i + 1 < argc) a.steps = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--lr") && i + 1 < argc) a.lr = atof(argv[++i]);
        else if (!strcmp(argv[i], "--block-size") && i + 1 < argc) a.block_size = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--window") && i + 1 < argc) a.window = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--embed") && i + 1 < argc) a.embed = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--hidden") && i + 1 < argc) a.hidden = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--layers") && i + 1 < argc) a.layers = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--d0-mode") && i + 1 < argc) a.d0_learned = !strcmp(argv[++i], "learned");
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) a.seed = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--report-every") && i + 1 < argc) a.report_every = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--diffusion") && i + 1 < argc) a.diffusion = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--save-weights") && i + 1 < argc) a.save_path = argv[++i];
        else if (!strcmp(argv[i], "--save-every") && i + 1 < argc) a.save_every = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--load-weights") && i + 1 < argc) a.load_path = argv[++i];
        else if (!strcmp(argv[i], "--eval-corpus") && i + 1 < argc) a.eval_path = argv[++i];
        else if (!strcmp(argv[i], "--eval-windows") && i + 1 < argc) a.eval_windows = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--eval-skip") && i + 1 < argc) a.eval_skip = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--eval-every") && i + 1 < argc) a.eval_every = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--t-min") && i + 1 < argc) a.t_min = atof(argv[++i]);
        else if (!strcmp(argv[i], "--lr-decay") && i + 1 < argc) a.lr_decay = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--lr-decay-steps") && i + 1 < argc) {
            // comma-separated list of step numbers, e.g. "3000" or "2000,4000,6000"
            char *tok = strtok(argv[++i], ",");
            while (tok && a.lr_decay_count < 16) {
                a.lr_decay_steps[a.lr_decay_count++] = strtoull(tok, NULL, 10);
                tok = strtok(NULL, ",");
            }
        } else if (!strcmp(argv[i], "--lr-decay-factor") && i + 1 < argc) a.lr_decay_factor = atof(argv[++i]);
        else if (!strcmp(argv[i], "--mask-warmup") && i + 1 < argc) a.mask_warmup = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--mask-scale") && i + 1 < argc) a.mask_scale = atof(argv[++i]);
        else if (!strcmp(argv[i], "--mask-late-step") && i + 1 < argc) a.mask_late_step = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--mask-late-scale") && i + 1 < argc) a.mask_late_scale = atof(argv[++i]);
        else if (!strcmp(argv[i], "--entropy-patches")) a.entropy_patches = 1;
        else if (!strcmp(argv[i], "--train-entropy-lm") && i + 1 < argc) a.train_entlm = argv[++i];
        else if (!strcmp(argv[i], "--entropy-lm") && i + 1 < argc) a.entropy_lm = argv[++i];
        else if (!strcmp(argv[i], "--backend") && i + 1 < argc) {
            ++i;
            if (!strcmp(argv[i], "cuda")) a.use_cuda = 1;
            else if (!strcmp(argv[i], "cpu")) a.use_cuda = 0;
            else {
                fprintf(stderr, "unknown --backend '%s'\n", argv[i]);
                return 1;
            }
        } else if (!strcmp(argv[i], "--enc-layers") && i + 1 < argc) a.enc_layers = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--glob-layers") && i + 1 < argc) a.glob_layers = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--dec-layers") && i + 1 < argc) a.dec_layers = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--cross-attn") && i + 1 < argc) {
            ++i;
            if (!strcmp(argv[i], "all")) a.cross_last = 0;
            else if (!strcmp(argv[i], "last")) a.cross_last = 1;
            else {
                fprintf(stderr, "unknown --cross-attn '%s'\n", argv[i]);
                return 1;
            }
        } else if (!strcmp(argv[i], "--t-warmup-hi") && i + 1 < argc) a.t_warmup_frac = atof(argv[++i]);
        else if (!strcmp(argv[i], "--t-hi-start") && i + 1 < argc) a.t_hi_start = atof(argv[++i]);
        else if (!strcmp(argv[i], "--optimizer") && i + 1 < argc) {
            ++i;
            if (!strcmp(argv[i], "sgd")) a.optimizer = 0;
            else if (!strcmp(argv[i], "adamw")) a.optimizer = 1;
            else {
                fprintf(stderr, "unknown --optimizer '%s'\n", argv[i]);
                return 1;
            }
        } else if (!strcmp(argv[i], "--beta1") && i + 1 < argc) a.beta1 = atof(argv[++i]);
        else if (!strcmp(argv[i], "--beta2") && i + 1 < argc) a.beta2 = atof(argv[++i]);
        else if (!strcmp(argv[i], "--eps") && i + 1 < argc) a.eps = atof(argv[++i]);
        else if (!strcmp(argv[i], "--weight-decay") && i + 1 < argc) a.weight_decay = atof(argv[++i]);
        else if (!strcmp(argv[i], "--grad-norm-log") && i + 1 < argc) a.grad_norm_log = argv[++i];
        else if (!strcmp(argv[i], "--max-norm") && i + 1 < argc) a.max_norm = atof(argv[++i]);
        else if (!strcmp(argv[i], "--update-norm-log") && i + 1 < argc) a.update_norm_log = argv[++i];
        else if (!strcmp(argv[i], "--component-norm-log") && i + 1 < argc) a.component_norm_log = argv[++i];
        else if (!strcmp(argv[i], "--activation-dump-log") && i + 1 < argc) a.activation_dump_log = argv[++i];
        else if (!strcmp(argv[i], "--batch-log") && i + 1 < argc) a.batch_log = argv[++i];
        else if (!strcmp(argv[i], "--loss-log") && i + 1 < argc) a.loss_log = argv[++i];
        else if (!strcmp(argv[i], "--cuda-scratch-mb") && i + 1 < argc)
            a.cuda_scratch_mb = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--model-mb") && i + 1 < argc) a.model_mb = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--deterministic")) a.deterministic = 1;
        else {
            usage();
            return 1;
        }
    }
    if (a.corpus_path == NULL) {
        usage();
        return 1;
    }

    // --deterministic guardrails
    if (a.deterministic) {
#if defined(BLT_WITH_CUDA)
        fprintf(stderr, "deterministic mode: single-threaded scatter_add + pedantic cuBLAS, expect ~3-10x slowdown, "
                        "not for production training\n");
#endif
        // CPU backend is always deterministic; flag is a no-op there.
    }

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

        for (size_t i = 0; i < model->encoder->config.num_layers; i++) {
            zero_tensor(&grad->encoder_grad->layer_grads[i].norm1_weight);
            zero_tensor(&grad->encoder_grad->layer_grads[i].norm2_weight);
            zero_tensor(&grad->encoder_grad->layer_grads[i].cross_norm_weight);
        }
        for (size_t i = 0; i < model->global->stack.num_layers; i++) {
            zero_tensor(&grad->global_grad->stack_grad->layer_grads[i].norm1_weight);
            zero_tensor(&grad->global_grad->stack_grad->layer_grads[i].norm2_weight);
        }
        for (size_t i = 0; i < model->decoder->config.num_layers; i++) {
            zero_tensor(&grad->decoder_grad->layer_grads[i].norm1_weight);
            zero_tensor(&grad->decoder_grad->layer_grads[i].norm2_weight);
            zero_tensor(&grad->decoder_grad->layer_grads[i].cross_norm_weight);
        }

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
            printf("step %6zu/%zu  epoch %zu/%zu  avg_loss %.4f  ETA %zuh%02um\n", step + 1, a.steps, epoch,
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
    if (aw) {
        free(aw->ngram);
        free(aw);
    }
    if (gnorm_fp) fclose(gnorm_fp);
    if (unorm_fp) fclose(unorm_fp);
    if (cnorm_fp) fclose(cnorm_fp);
    if (adump_fp) fclose(adump_fp);
    if (batch_fp) fclose(batch_fp);
    if (loss_fp) fclose(loss_fp);
    free(corpus);
    return 0;
}
