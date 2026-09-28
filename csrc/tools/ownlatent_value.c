// Own-latent value: paired A/B of the decoder's "final byte reads its own
// patch latent" rule in blt_patch_decoder_latent_at. Arm (a) decodes with the
// pristine global_out (real rule); arm (b) decodes with global_out row j
// overwritten by row j-1, so patch j's final byte row cross-attends to the
// previous patch's latent. Encoder output, RoPE, D_0, self-attention, weights
// and the patch array are identical across arms.

#include <float.h>
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
#include "models/local_decoder.h"
#include "models/patcher.h"
#include "ops/softmax.h"
#include "ops/patch_pool.h"

#define OWNLAT_MAX_PATCHES 4096
#define OWNLAT_NUM_BUCKETS 6
#define OWNLAT_MAX_SEQ 1024

static const char *const bucket_names[OWNLAT_NUM_BUCKETS] = {"1", "2-3", "4-6", "7-9", "10-16", "17+"};

typedef struct {
    size_t n;
    size_t hits_a;
    size_t hits_b;
    size_t n10;
    size_t n01;
    double margin_a_sum;
    double margin_b_sum;
} owncat_stats;

typedef struct {
    size_t j;
    size_t p;
    size_t len;
    int hit_a;
    float margin_a;
} own_sample;

static long fsize(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fclose(f);
    return sz;
}

// The patcher runs on the host, so device entropy tensors get staged through
// host memory. Returns a malloc'd buffer the caller must free (NULL on CPU).
static const blt_tensor *stage_entropy_host(const blt_tensor *vals, blt_tensor *host_view_out, float **buf_out) {
    if (vals->backend == BLT_BACKEND_CPU) {
        *buf_out = NULL;
        return vals;
    }
    float *buf = (float *)malloc(vals->numel * sizeof(float));
    BLT_REQUIRE(buf != NULL, "stage_entropy_host: allocation failed");
    blt_tensor_download(vals, buf, vals->numel * sizeof(float));
    view_1d(host_view_out, buf, vals->numel, vals->dtype, BLT_BACKEND_CPU);
    *buf_out = buf;
    return host_view_out;
}

// Entropy LM logits over the bytes, softmax, per-position entropy (nats);
// mirrors compute_entropy_vals in csrc/tools/selfspec_audit.c.
static void compute_entropy_vals(blt_arena *arena, const blt_entropy_lm *entropy_model, const uint8_t *bytes,
                                 size_t len, blt_tensor *entropy_vals_out) {
    size_t bytes_shape[1] = {len};
    blt_tensor bytes_in = blt_tensor_create(arena, bytes_shape, 1, BLT_DTYPE_UINT8);
    blt_tensor_upload(&bytes_in, bytes, len);

    size_t logits_shape[2] = {len, 256};
    blt_tensor logits = blt_tensor_create(arena, logits_shape, 2, BLT_DTYPE_FP32);
    size_t scalar_shape[1] = {1};
    blt_tensor discard_loss = blt_tensor_create(arena, scalar_shape, 1, BLT_DTYPE_FP32);

    blt_entropy_lm_forward(entropy_model, &bytes_in, &logits, &discard_loss, arena);

    blt_tensor probs = blt_tensor_create(arena, logits_shape, 2, BLT_DTYPE_FP32);
    blt_softmax(&logits, &probs);

    size_t vals_shape[1] = {len};
    *entropy_vals_out = blt_tensor_create(arena, vals_shape, 1, BLT_DTYPE_FP32);
    blt_entropy_config entropy_cfg = {.vocab_size = 256, .use_log2 = false};
    blt_compute_entropy(&probs, entropy_vals_out, &entropy_cfg);
}

// Segments one window with the entropy patcher; returns the patch count.
static size_t segment_window(blt_arena *arena, const blt_entropy_lm *lm, const uint8_t *bytes, size_t len,
                             blt_patch_info *patches_out, size_t max_patches, const blt_patcher_config *pcfg) {
    blt_tensor entropy_vals;
    compute_entropy_vals(arena, lm, bytes, len, &entropy_vals);

    blt_tensor entropy_host_view;
    float *entropy_host_buf = NULL;
    const blt_tensor *entropy_for_patcher = stage_entropy_host(&entropy_vals, &entropy_host_view, &entropy_host_buf);
    const size_t n = blt_segment_patches(entropy_for_patcher, bytes, patches_out, max_patches, pcfg);
    free(entropy_host_buf);
    return n;
}

// First-max-wins argmax plus top1-top2 margin for one logits row.
static int row_top2(const float *row, size_t vocab, float *margin_out) {
    size_t best = 0;
    for (size_t v = 1; v < vocab; v++) {
        if (row[v] > row[best]) best = v;
    }
    float second = -FLT_MAX;
    for (size_t v = 0; v < vocab; v++) {
        if (v == best) continue;
        if (row[v] > second) second = row[v];
    }
    *margin_out = row[best] - second;
    return (int)best;
}

// Bucket by the length of the patch whose final byte the row is.
static int len_bucket(size_t len) {
    if (len == 1) return 0;
    if (len <= 3) return 1;
    if (len <= 6) return 2;
    if (len <= 9) return 3;
    if (len <= 16) return 4;
    return 5;
}

static void record_pair(owncat_stats *bucket, owncat_stats *overall, int hit_a, int hit_b, float margin_a,
                        float margin_b) {
    bucket->n++;
    overall->n++;
    bucket->hits_a += (size_t)hit_a;
    overall->hits_a += (size_t)hit_a;
    bucket->hits_b += (size_t)hit_b;
    overall->hits_b += (size_t)hit_b;
    if (hit_a && !hit_b) {
        bucket->n10++;
        overall->n10++;
    }
    if (!hit_a && hit_b) {
        bucket->n01++;
        overall->n01++;
    }
    bucket->margin_a_sum += (double)margin_a;
    overall->margin_a_sum += (double)margin_a;
    bucket->margin_b_sum += (double)margin_b;
    overall->margin_b_sum += (double)margin_b;
}

// McNemar paired CI: delta = (n10 - n01)/n, se = sqrt(n10 + n01)/n.
static void print_row(const char *name, const owncat_stats *s) {
    const double n = (double)s->n;
    const double acc_a = s->n ? (double)s->hits_a / n : 0.0;
    const double acc_b = s->n ? (double)s->hits_b / n : 0.0;
    const double delta = s->n ? ((double)s->n10 - (double)s->n01) / n : 0.0;
    const double se = s->n ? sqrt((double)(s->n10 + s->n01)) / n : 0.0;
    const double hw = 1.96 * se;
    const double margin_a = s->n ? s->margin_a_sum / n : 0.0;
    const double margin_b = s->n ? s->margin_b_sum / n : 0.0;
    printf("%-8s %6zu %8.4f %8.4f %+9.4f %9.4f %6zu %6zu %10.4f %10.4f\n", name, s->n, acc_a, acc_b, delta, hw, s->n10,
           s->n01, margin_a, margin_b);
}

int main(int argc, char **argv) {
    const char *checkpoint = NULL;
    const char *entropy_lm_path = NULL;
    const char *corpus = NULL;
    size_t embed = 0;
    size_t hidden = 0;
    size_t enc_layers = 0;
    size_t glob_layers = 0;
    size_t dec_layers = 0;
    int cross_last = 0;
    size_t window = 512;
    size_t num_windows = 8;
    size_t target_samples = 500;
    size_t max_windows = 40;
    uint64_t seed = 7;
    int use_cuda = 0;

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
            if (!strcmp(argv[i], "last")) cross_last = 1;
            else if (strcmp(argv[i], "all") != 0) {
                fprintf(stderr, "Error: --cross-attn must be all|last\n");
                return 1;
            }
        } else if (!strcmp(argv[i], "--window") && i + 1 < argc) window = (size_t)atol(argv[++i]);
        else if (!strcmp(argv[i], "--windows") && i + 1 < argc) num_windows = (size_t)atol(argv[++i]);
        else if (!strcmp(argv[i], "--target-samples") && i + 1 < argc) target_samples = (size_t)atol(argv[++i]);
        else if (!strcmp(argv[i], "--max-windows") && i + 1 < argc) max_windows = (size_t)atol(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = (uint64_t)strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--backend") && i + 1 < argc) {
            i++;
            if (!strcmp(argv[i], "cuda")) use_cuda = 1;
            else if (strcmp(argv[i], "cpu") != 0) {
                fprintf(stderr, "Error: --backend must be cpu|cuda\n");
                return 1;
            }
        }
    }

    if (!checkpoint || !entropy_lm_path || !corpus || embed == 0 || hidden == 0 || enc_layers == 0 ||
        glob_layers == 0 || dec_layers == 0) {
        fprintf(stderr, "Usage: ownlatent_value --checkpoint FILE --entropy-lm FILE --corpus FILE "
                        "--embed N --hidden N --enc-layers N --glob-layers N --dec-layers N "
                        "[--cross-attn all|last] [--window N] [--windows N] [--target-samples N] "
                        "[--max-windows N] [--seed N] [--backend cpu]\n");
        return 1;
    }
    if (use_cuda) {
        fprintf(stderr, "FATAL: ownlatent_value does not implement --backend cuda; use --backend cpu\n");
        return 1;
    }
    if (window < 2 || window > OWNLAT_MAX_SEQ) {
        fprintf(stderr, "Error: --window must be in [2, %d]\n", OWNLAT_MAX_SEQ);
        return 1;
    }
    if (num_windows == 0 || max_windows == 0 || target_samples == 0) {
        fprintf(stderr, "Error: --windows, --max-windows and --target-samples must be >= 1\n");
        return 1;
    }

    long file_size = fsize(corpus);
    if (file_size <= 0) {
        fprintf(stderr, "Error: cannot read corpus '%s'\n", corpus);
        return 1;
    }
    const size_t data_len = (size_t)file_size;
    uint8_t *data = (uint8_t *)malloc(data_len);
    BLT_REQUIRE(data != NULL, "ownlatent_value: corpus staging alloc failed");
    FILE *f = fopen(corpus, "rb");
    BLT_REQUIRE(f != NULL, "ownlatent_value: cannot open corpus '%s'", corpus);
    if (fread(data, 1, data_len, f) != data_len) {
        fclose(f);
        fprintf(stderr, "Error: short read from '%s'\n", corpus);
        free(data);
        return 1;
    }
    fclose(f);

    const blt_backend dev = BLT_BACKEND_CPU;
    blt_arena *model_arena = blt_arena_create(160ULL * 1024 * 1024, dev);
    blt_arena *scratch = blt_arena_create(256ULL * 1024 * 1024, dev);
    blt_arena *lm_arena = blt_arena_create(32ULL * 1024 * 1024, dev);
    blt_arena *seg_arena = blt_arena_create(16ULL * 1024 * 1024, dev);
    // Encoder output must outlive every per-run scratch reset within a window.
    blt_arena *enc_arena = blt_arena_create(256ULL * 1024 * 1024, dev);

    blt_model_config cfg;
    blt_model_config_defaults(&cfg, embed, hidden, enc_layers, glob_layers, dec_layers, OWNLAT_MAX_SEQ, cross_last);
    blt_model *model = blt_model_create(model_arena, &cfg);
    blt_model_load(model, checkpoint);
    blt_entropy_lm *lm = blt_make_entropy_lm(lm_arena, OWNLAT_MAX_SEQ, entropy_lm_path, seed);

    blt_patcher_config pcfg;
    blt_make_patcher_cfg(&pcfg, 0, 2.5f, 1.0f, 16);

    const size_t E = cfg.decoder_config.embed_dim;
    const size_t V = cfg.decoder_config.vocab_size;
    const size_t patch_dim = cfg.decoder_config.patch_dim ? cfg.decoder_config.patch_dim : E;
    BLT_REQUIRE(patch_dim == E && E > 0,
                "ownlatent_value: this tool needs k = patch_dim/embed_dim == 1 (patch_dim=%zu embed=%zu)", patch_dim,
                E);

    fprintf(stderr, "[OWNLAT] checkpoint: %s\n", checkpoint);
    fprintf(stderr, "[OWNLAT] entropy LM: %s (seed=%llu)\n", entropy_lm_path, (unsigned long long)seed);
    fprintf(stderr, "[OWNLAT] corpus: %s (%zu bytes)\n", corpus, data_len);
    fprintf(stderr, "[OWNLAT] config: embed=%zu hidden=%zu enc=%zu glob=%zu dec=%zu cross=%s backend=cpu\n", embed,
            hidden, enc_layers, glob_layers, dec_layers, cross_last ? "last" : "all");
    fprintf(stderr, "[OWNLAT] window=%zu windows=%zu max-windows=%zu target-samples=%zu\n", window, num_windows,
            max_windows, target_samples);

    // Host staging: pristine global_out, treatment copy, control logits, samples.
    float *o_pristine = (float *)malloc((size_t)window * E * sizeof(float));
    float *o_modified = (float *)malloc((size_t)window * E * sizeof(float));
    float *logits_host = (float *)malloc((size_t)window * V * sizeof(float));
    own_sample *samples = (own_sample *)malloc((size_t)window * sizeof(own_sample));
    BLT_REQUIRE(o_pristine && o_modified && logits_host && samples, "ownlatent_value: host staging alloc failed");

    owncat_stats buckets[OWNLAT_NUM_BUCKETS];
    owncat_stats overall;
    memset(buckets, 0, sizeof(buckets));
    memset(&overall, 0, sizeof(overall));
    size_t excluded_j0 = 0;
    size_t excluded_boundary = 0;
    size_t processed = 0;
    size_t skipped = 0;

    // Control-arm sanity diagnostics (stderr): top-1 over all rows, over the
    // patch-final rows the table reads, and over the remaining rows.
    size_t diag_all_hits = 0;
    size_t diag_all_n = 0;
    size_t diag_fin_hits = 0;
    size_t diag_fin_n = 0;
    size_t diag_space_fin = 0;
    size_t diag_space_all = 0;

    const size_t limit = num_windows < max_windows ? num_windows : max_windows;
    for (size_t w = 0; w < limit && overall.n < target_samples; w++) {
        const size_t offset = w * window;
        if (offset + window > data_len) break;
        const uint8_t *text = data + offset;

        blt_arena_reset(seg_arena);
        blt_patch_info patches[OWNLAT_MAX_PATCHES];
        const size_t M = segment_window(seg_arena, lm, text, window, patches, OWNLAT_MAX_PATCHES, &pcfg);
        if (M < 3) {
            skipped++;
            continue;
        }
        BLT_REQUIRE(M <= window, "ownlatent_value: window %zu yielded %zu patches > %zu bytes", window, M, window);

        // Stage 1: encoder + global transformer once; latents live in enc_arena.
        blt_arena_reset(enc_arena);
        size_t bytes_shape[1] = {window};
        blt_tensor bytes_in = blt_tensor_create(enc_arena, bytes_shape, 1, BLT_DTYPE_UINT8);
        blt_tensor_upload(&bytes_in, text, window);
        blt_model_enc_out enc;
        blt_model_encode(model, &bytes_in, patches, M, NULL, 0, &enc, enc_arena);
        BLT_REQUIRE(enc.global_out.shape[0] == M && enc.global_out.shape[1] == E,
                    "ownlatent_value: global_out must be [%zu, %zu]", M, E);
        blt_tensor_download(&enc.global_out, o_pristine, M * E * sizeof(float));

        // Arm (a): one decoder pass with the real own-latent rule.
        blt_arena_reset(scratch);
        size_t logits_shape[2] = {window, V};
        blt_tensor logits = blt_tensor_create(scratch, logits_shape, 2, BLT_DTYPE_FP32);
        blt_local_decoder_forward_ext(model->decoder, &enc.byte_hidden_out, &enc.global_out, patches, M, NULL, NULL,
                                      NULL, 0, NULL, &logits, NULL, scratch);
        blt_tensor_download(&logits, logits_host, window * V * sizeof(float));

        size_t n_samp = 0;
        size_t win_all_hits = 0;
        size_t win_all_n = 0;
        size_t win_fin_hits = 0;
        size_t win_fin_n = 0;
        for (size_t i = 0; i + 1 < window; i++) {
            float m = 0.0f;
            const int pred = row_top2(logits_host + i * V, V, &m);
            if (pred == (int)text[i + 1]) win_all_hits++;
            if (text[i + 1] == 0x20) diag_space_all++;
            win_all_n++;
            diag_all_n++;
        }
        for (size_t j = 0; j < M; j++) {
            const size_t p = patches[j].start_idx + patches[j].length;
            if (p < window) {
                float m = 0.0f;
                const int pred = row_top2(logits_host + (p - 1) * V, V, &m);
                win_fin_n++;
                if (pred == (int)text[p]) win_fin_hits++;
                if (text[p] == 0x20) diag_space_fin++;
            }
            if (j == 0) {
                excluded_j0++;
                continue;
            }
            if (p >= window) {
                excluded_boundary++;
                continue;
            }
            own_sample *s = &samples[n_samp++];
            s->j = j;
            s->p = p;
            s->len = patches[j].length;
            float margin = 0.0f;
            const int pred = row_top2(logits_host + (p - 1) * V, V, &margin);
            s->hit_a = (pred == (int)text[p]);
            s->margin_a = margin;
            if (w == 0 && n_samp <= 6) {
                fprintf(stderr, "[OWNLAT] sample p=%zu j=%zu len=%zu actual=0x%02X pred=0x%02X %s\n", p, j, s->len,
                        (unsigned)text[p], (unsigned)pred, s->hit_a ? "HIT" : "MISS");
            }
        }
        diag_all_hits += win_all_hits;
        diag_fin_hits += win_fin_hits;
        diag_fin_n += win_fin_n;
        const size_t win_nonfin_n = win_all_n - win_fin_n;
        const size_t win_nonfin_hits = win_all_hits - win_fin_hits;
        fprintf(stderr,
                "[OWNLAT] window %zu control top1: all=%.4f (%zu/%zu) final=%.4f (%zu/%zu) nonfinal=%.4f (%zu/%zu)\n",
                w, win_all_n ? (double)win_all_hits / (double)win_all_n : 0.0, win_all_hits, win_all_n,
                win_fin_n ? (double)win_fin_hits / (double)win_fin_n : 0.0, win_fin_hits, win_fin_n,
                win_nonfin_n ? (double)win_nonfin_hits / (double)win_nonfin_n : 0.0, win_nonfin_hits, win_nonfin_n);

        // Arm (b): one decoder pass per position, patch_in row j <- row j-1.
        // Only row e_j - 1 (read here) and patch j+1's interior rows (never
        // read) attend to row j, and causal self-attention keeps patch j+1
        // out of row e_j - 1, so this run isolates that one row's latent.
        for (size_t i = 0; i < n_samp && overall.n < target_samples; i++) {
            const own_sample *s = &samples[i];
            memcpy(o_modified, o_pristine, M * E * sizeof(float));
            memcpy(o_modified + s->j * E, o_modified + (s->j - 1) * E, E * sizeof(float));

            blt_arena_reset(scratch);
            size_t o_shape[2] = {M, E};
            blt_tensor o_mod = blt_tensor_create(scratch, o_shape, 2, BLT_DTYPE_FP32);
            blt_tensor_upload(&o_mod, o_modified, M * E * sizeof(float));
            blt_tensor logits_b = blt_tensor_create(scratch, logits_shape, 2, BLT_DTYPE_FP32);
            blt_local_decoder_forward_ext(model->decoder, &enc.byte_hidden_out, &o_mod, patches, M, NULL, NULL, NULL, 0,
                                          NULL, &logits_b, NULL, scratch);
            blt_tensor_download(&logits_b, logits_host, window * V * sizeof(float));

            float margin_b = 0.0f;
            const int pred_b = row_top2(logits_host + (s->p - 1) * V, V, &margin_b);
            const int hit_b = (pred_b == (int)text[s->p]);
            record_pair(&buckets[len_bucket(s->len)], &overall, s->hit_a, hit_b, s->margin_a, margin_b);
        }

        processed++;
        fprintf(stderr, "[OWNLAT] window %zu/%zu M=%zu eligible=%zu paired=%zu\n", w + 1, limit, M, n_samp, overall.n);
    }

    if (overall.n < target_samples) {
        fprintf(stderr, "[OWNLAT] stopped with %zu < %zu target samples after %zu window(s)\n", overall.n,
                target_samples, processed);
    }
    fprintf(stderr, "[OWNLAT] windows processed=%zu skipped=%zu boundary-excluded=%zu\n", processed, skipped,
            excluded_boundary);
    if (diag_all_n > 0) {
        fprintf(stderr, "[OWNLAT] target composition: space at patch-final %zu/%zu, space over all rows %zu/%zu\n",
                diag_space_fin, diag_fin_n, diag_space_all, diag_all_n);
        fprintf(stderr, "[OWNLAT] control top1 overall: all=%.4f (%zu/%zu) final=%.4f (%zu/%zu) nonfinal=%.4f\n",
                (double)diag_all_hits / (double)diag_all_n, diag_all_hits, diag_all_n,
                diag_fin_n ? (double)diag_fin_hits / (double)diag_fin_n : 0.0, diag_fin_hits, diag_fin_n,
                (diag_all_n > diag_fin_n) ? (double)(diag_all_hits - diag_fin_hits) / (double)(diag_all_n - diag_fin_n)
                                          : 0.0);
    }

    printf("%-8s %6s %8s %8s %9s %9s %6s %6s %10s %10s\n", "bucket", "n", "acc_a", "acc_b", "delta", "ci95_hw", "n10",
           "n01", "margin_a", "margin_b");
    for (int b = 0; b < OWNLAT_NUM_BUCKETS; b++) print_row(bucket_names[b], &buckets[b]);
    print_row("OVERALL", &overall);
    printf("\nn10 + n01 = %zu discordant pairs; with n10 + n01 this small the normal approximation CI is "
           "optimistic.\n",
           overall.n10 + overall.n01);
    printf("j == 0 positions excluded (own-latent rule cannot be flipped): %zu\n", excluded_j0);

    free(samples);
    free(logits_host);
    free(o_modified);
    free(o_pristine);
    free(data);
    blt_arena_destroy(enc_arena);
    blt_arena_destroy(seg_arena);
    blt_arena_destroy(lm_arena);
    blt_arena_destroy(scratch);
    blt_arena_destroy(model_arena);
    return 0;
}
