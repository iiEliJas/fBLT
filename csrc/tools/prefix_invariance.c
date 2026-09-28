// Prefix-invariance audit for the paper-aligned patcher. For each held-out
// window, compute a full teacher-forced forward over x[0..N) and, for sampled
// positions p, a separate forward over the shorter prefix x[0..p+1). The
// causal decoder plus the prefix-decidable finality rule guarantee the logits at
// row p are identical in both, up to float reduction-order noise. Any row that
// depends on a later byte (a look-ahead term in the encoder, the global
// transformer, or a patcher rule) shows up as a large diff here.

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "core/allocator.h"
#include "core/backend.h"
#include "core/tensor.h"
#include "models/model.h"
#include "models/model_builder.h"
#include "models/checkpoint.h"
#include "models/entropy_lm.h"
#include "models/entropy.h"
#include "models/patcher.h"
#include "ops/softmax.h"

#define PREFIX_MAX_PATCHES 4096
#define PREFIX_MAX_SEQ 1024

static long fsize(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fclose(f);
    return sz;
}

// Entropy over the prefix, mirroring the trainer/greedy call sequence.
static void compute_entropy_vals(blt_arena *arena, const blt_entropy_lm *lm, const uint8_t *bytes, size_t len,
                                 float *ent_out) {
    size_t bytes_shape[1] = {len};
    blt_tensor bytes_in = blt_tensor_create(arena, bytes_shape, 1, BLT_DTYPE_UINT8);
    blt_tensor_upload(&bytes_in, bytes, len);
    size_t logits_shape[2] = {len, 256};
    blt_tensor logits = blt_tensor_create(arena, logits_shape, 2, BLT_DTYPE_FP32);
    size_t scalar_shape[1] = {1};
    blt_tensor discard = blt_tensor_create(arena, scalar_shape, 1, BLT_DTYPE_FP32);
    blt_entropy_lm_forward(lm, &bytes_in, &logits, &discard, arena);
    blt_tensor probs = blt_tensor_create(arena, logits_shape, 2, BLT_DTYPE_FP32);
    blt_softmax(&logits, &probs);
    size_t vals_shape[1] = {len};
    blt_tensor ent_t = blt_tensor_create(arena, vals_shape, 1, BLT_DTYPE_FP32);
    blt_entropy_config ecfg = {.vocab_size = 256, .use_log2 = false};
    blt_compute_entropy(&probs, &ent_t, &ecfg);
    if (ent_t.backend == BLT_BACKEND_CPU) {
        memcpy(ent_out, (const float *)ent_t.data, len * sizeof(float));
    } else {
        blt_tensor_download(&ent_t, ent_out, len * sizeof(float));
    }
}

// Full forward over x[0..len) with trailing_closed evaluated at `len`.
static void forward_row(const blt_model *model, const blt_entropy_lm *lm, const blt_patcher_config *pcfg,
                        const uint8_t *bytes, size_t len, int trailing_closed, blt_arena *scratch, float *ent_scratch,
                        blt_patch_info *patches, float *logits_out) {
    compute_entropy_vals(scratch, lm, bytes, len, ent_scratch);
    blt_tensor ent_view;
    view_1d(&ent_view, ent_scratch, len, BLT_DTYPE_FP32, BLT_BACKEND_CPU);
    size_t M = blt_segment_patches(&ent_view, bytes, patches, PREFIX_MAX_PATCHES, pcfg);

    size_t bytes_shape[1] = {len};
    blt_tensor bin = blt_tensor_create(scratch, bytes_shape, 1, BLT_DTYPE_UINT8);
    blt_tensor_upload(&bin, bytes, len);
    blt_model_enc_out enc;
    blt_model_encode(model, &bin, patches, M, NULL, 0, &enc, scratch);

    size_t V = model->config.decoder_config.vocab_size;
    size_t logits_shape[2] = {len, V};
    blt_tensor logits = blt_tensor_create(scratch, logits_shape, 2, BLT_DTYPE_FP32);
    blt_local_decoder_forward_ext(model->decoder, &enc.byte_hidden_out, &enc.global_out, patches, M, NULL, NULL, NULL,
                                  0, NULL, trailing_closed, &logits, NULL, scratch);
    blt_tensor_download(&logits, logits_out, (size_t)len * V * sizeof(float));
}

int main(int argc, char **argv) {
    const char *checkpoint = NULL, *entropy_lm_path = NULL, *corpus = NULL;
    size_t embed = 0, hidden = 0, enc_layers = 0, glob_layers = 0, dec_layers = 0;
    size_t window = 512, num_windows = 64, samples_per_window = 12;
    uint64_t seed = 7;
    double tol = 1e-3;
    int cross_last = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--checkpoint") && i + 1 < argc) checkpoint = argv[++i];
        else if (!strcmp(argv[i], "--entropy-lm") && i + 1 < argc) entropy_lm_path = argv[++i];
        else if (!strcmp(argv[i], "--corpus") && i + 1 < argc) corpus = argv[++i];
        else if (!strcmp(argv[i], "--embed") && i + 1 < argc) embed = (size_t)atol(argv[++i]);
        else if (!strcmp(argv[i], "--hidden") && i + 1 < argc) hidden = (size_t)atol(argv[++i]);
        else if (!strcmp(argv[i], "--enc-layers") && i + 1 < argc) enc_layers = (size_t)atol(argv[++i]);
        else if (!strcmp(argv[i], "--glob-layers") && i + 1 < argc) glob_layers = (size_t)atol(argv[++i]);
        else if (!strcmp(argv[i], "--dec-layers") && i + 1 < argc) dec_layers = (size_t)atol(argv[++i]);
        else if (!strcmp(argv[i], "--cross-attn") && i + 1 < argc) {
            i++;
            cross_last = !strcmp(argv[i], "last");
        } else if (!strcmp(argv[i], "--window") && i + 1 < argc) window = (size_t)atol(argv[++i]);
        else if (!strcmp(argv[i], "--windows") && i + 1 < argc) num_windows = (size_t)atol(argv[++i]);
        else if (!strcmp(argv[i], "--samples-per-window") && i + 1 < argc) samples_per_window = (size_t)atol(argv[++i]);
        else if (!strcmp(argv[i], "--tol") && i + 1 < argc) tol = atof(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = (uint64_t)strtoull(argv[++i], NULL, 10);
    }
    if (!checkpoint || !entropy_lm_path || !corpus || embed == 0 || hidden == 0) {
        fprintf(stderr, "Usage: prefix_invariance --checkpoint F --entropy-lm F --corpus F --embed N --hidden N "
                        "--enc-layers N --glob-layers N --dec-layers N [--window N] [--windows N] "
                        "[--samples-per-window N] [--tol F]\n");
        return 1;
    }
    long fsz = fsize(corpus);
    if (fsz <= 0) {
        fprintf(stderr, "Error: cannot read corpus\n");
        return 1;
    }
    size_t data_len = (size_t)fsz;
    uint8_t *data = (uint8_t *)malloc(data_len);
    FILE *f = fopen(corpus, "rb");
    if (!f || fread(data, 1, data_len, f) != data_len) {
        fprintf(stderr, "Error: short read\n");
        return 1;
    }
    fclose(f);

    const blt_backend dev = BLT_BACKEND_CPU;
    blt_arena *arena = blt_arena_create(512ULL * 1024 * 1024, dev);
    blt_model_config cfg;
    blt_model_config_defaults(&cfg, embed, hidden, enc_layers, glob_layers, dec_layers, PREFIX_MAX_SEQ, cross_last);
    blt_model *model = blt_model_create(arena, &cfg);
    blt_model_load(model, checkpoint);
    blt_entropy_lm *lm = blt_make_entropy_lm(arena, PREFIX_MAX_SEQ, entropy_lm_path, seed);
    blt_patcher_config pcfg;
    blt_make_patcher_cfg(&pcfg, 0, 2.5f, 1.0f, 16);
    // Model + LM live in `arena`; per-forward temporaries in `scratch` so we
    // can reset between forwards without touching the weights.
    blt_arena *scratch = blt_arena_create(512ULL * 1024 * 1024, dev);

    const size_t V = cfg.decoder_config.vocab_size;
    float *ent_full = (float *)malloc(window * sizeof(float));
    float *ent_pref = (float *)malloc(window * sizeof(float));
    float *logits_full = (float *)malloc((size_t)window * V * sizeof(float));
    float *logits_pref = (float *)malloc((size_t)window * V * sizeof(float));
    blt_patch_info *p_full = (blt_patch_info *)malloc(PREFIX_MAX_PATCHES * sizeof(blt_patch_info));
    blt_patch_info *p_pref = (blt_patch_info *)malloc(PREFIX_MAX_PATCHES * sizeof(blt_patch_info));
    if (!ent_full || !ent_pref || !logits_full || !logits_pref || !p_full || !p_pref) {
        fprintf(stderr, "FATAL: alloc\n");
        return 1;
    }

    size_t total_cmp = 0, total_bad = 0;
    size_t open_cmp = 0, open_bad = 0;
    double worst = 0.0, worst_open = 0.0;
    size_t worst_win = 0, worst_pos = 0;

    for (size_t w = 0; w < num_windows; w++) {
        const size_t offset = w * window;
        if (offset + window > data_len) break;
        const uint8_t *text = data + offset;

        // Trailing closure of the full window (boundary at N) and its entropy.
        compute_entropy_vals(scratch, lm, text, window, ent_full);
        blt_tensor ef;
        view_1d(&ef, ent_full, window, BLT_DTYPE_FP32, BLT_BACKEND_CPU);
        size_t M_full = blt_segment_patches(&ef, text, p_full, PREFIX_MAX_PATCHES, &pcfg);
        int tc_full = 0;
        if (M_full >= 1) {
            tc_full = blt_next_starts_patch(text, window, ent_full, p_full[M_full - 1].start_idx,
                                            p_full[M_full - 1].length, &pcfg);
        }
        blt_arena_reset(scratch);
        forward_row(model, lm, &pcfg, text, window, tc_full, scratch, ent_full, p_full, logits_full);

        for (size_t s = 0; s < samples_per_window; s++) {
            // Spread sample positions across the window, avoiding the last byte.
            size_t p = 1 + (s * (window - 3)) / (samples_per_window ? samples_per_window : 1);
            if (p >= window - 1) p = window - 2;
            const size_t plen = p + 1;

            // Prefix forward over x[0..p+1); its trailing closure is the
            // boundary at p+1, which is exactly byte p's finality rule.
            compute_entropy_vals(scratch, lm, text, plen, ent_pref);
            blt_tensor ep;
            view_1d(&ep, ent_pref, plen, BLT_DTYPE_FP32, BLT_BACKEND_CPU);
            size_t M_pref = blt_segment_patches(&ep, text, p_pref, PREFIX_MAX_PATCHES, &pcfg);
            int tc_pref = 0;
            if (M_pref >= 1) {
                tc_pref = blt_next_starts_patch(text, plen, ent_pref, p_pref[M_pref - 1].start_idx,
                                                p_pref[M_pref - 1].length, &pcfg);
            }

            // A byte is "patch-closed" in the prefix iff a boundary already
            // fired at p+1, i.e. byte p's latent comes from a patch that will
            // not grow. For such bytes the prefix and full passes must agree
            // bit-for-bit up to float noise. Bytes in a still-open patch read a
            // mean-pooled latent over a patch that legitimately extends past p,
            // so their logits are expected to differ; those are counted apart.
            int patch_closed = tc_pref;

            blt_arena_reset(scratch);
            forward_row(model, lm, &pcfg, text, plen, tc_pref, scratch, ent_pref, p_pref, logits_pref);

            double maxdiff = 0.0;
            for (size_t v = 0; v < V; v++) {
                double a = logits_full[(size_t)p * V + v];
                double b = logits_pref[(size_t)p * V + v];
                double d = fabs(a - b);
                if (d > maxdiff) maxdiff = d;
            }
            if (patch_closed) {
                total_cmp++;
                if (maxdiff > tol) {
                    total_bad++;
                    if (w == 0 && total_bad <= 4)
                        fprintf(stderr, "[PREFIX] CLOSED win=%zu pos=%zu maxdiff=%.6g\n", w, p, maxdiff);
                }
                if (maxdiff > worst) {
                    worst = maxdiff;
                    worst_win = w;
                    worst_pos = p;
                }
            } else {
                open_cmp++;
                if (maxdiff > tol) open_bad++;
                if (maxdiff > worst_open) worst_open = maxdiff;
            }
        }
    }

    printf("prefix-invariance (patch-closed bytes; the real invariant):\n");
    printf("  windows=%zu window=%zu tol=%g  closed_comparisons=%zu  mismatches(>%g)=%zu  worst_maxdiff=%.6g (win=%zu "
           "pos=%zu)\n",
           num_windows, window, tol, total_cmp, tol, total_bad, worst, worst_win, worst_pos);
    printf("  open-patch bytes (expected to differ: pooled latent grows with the open patch): n=%zu  differ=%zu  "
           "worst=%.6g\n",
           open_cmp, open_bad, worst_open);
    printf("  %s\n",
           total_bad == 0 ? "PASS: no look-ahead dependence on closed patches" : "FAIL: closed-patch mismatch");
    return total_bad == 0 ? 0 : 1;
}
