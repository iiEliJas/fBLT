// Target-leak probe for the entropy patcher. Compares the repo's segmentation
// (patch starts at i iff entropy_data[i] > thr) against the paper-aligned one
// (patch starts at i iff entropy_data[i-1] > thr, i.e. H(x_i|x_<i) > thr).
// For each alignment it measures how much the segmentation reveals about the
// next-byte target x_{p+1}: target distribution on final rows, conditional
// entropy H(x_{p+1}|is_final), and a model-free lookup-table predictor
// target ~ f(is_final, patch_length, prev_byte) fit on one half of the windows
// and evaluated on the other. No model forward; entropy LM only.

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "core/allocator.h"
#include "core/backend.h"
#include "core/tensor.h"
#include "models/model_builder.h"
#include "models/entropy_lm.h"
#include "models/entropy.h"
#include "models/patcher.h"
#include "ops/softmax.h"

#define LEAK_MAX_PATCHES 4096
#define LEAK_MAX_SEQ 1024
#define LEAK_MAX_PATCH_LEN 16
#define LEAK_KEYS (2 * LEAK_MAX_PATCH_LEN * 256) // is_final x patch_len x prev_byte
#define LEAK_KEY_TARGETS 256

typedef struct {
    size_t n_windows;
    size_t n_bytes;
    size_t n_patches;
    size_t n_final_rows;       // natural final rows (p in [0,len-2], last byte of its patch)
    size_t n_all_rows;         // all p in [0,len-2]
    uint64_t target_hist[256]; // all rows
    uint64_t final_hist[256];  // final rows only
    uint64_t train_final_hist[256];
    uint64_t test_final_hist[256];
    size_t test_final_n;
    size_t test_final_correct; // lookup-table top-1 on test final rows
    size_t const_test_n;
    size_t const_test_correct; // most-common-train-target constant predictor
    uint32_t *lut;             // LEAK_KEYS x 256 counts (train half)
    double avg_patch_len;
} align_stats;

static void align_stats_init(align_stats *s) {
    memset(s, 0, sizeof(*s));
    s->lut = (uint32_t *)calloc((size_t)LEAK_KEYS * LEAK_KEY_TARGETS, sizeof(uint32_t));
    if (!s->lut) {
        fprintf(stderr, "FATAL: leak_probe: LUT alloc failed\n");
        exit(1);
    }
}

static void align_stats_free(align_stats *s) { free(s->lut); }

static long fsize(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fclose(f);
    return sz;
}

// Entropy LM over the bytes, softmax, per-position entropy. Byte-for-byte the
// same call sequence the trainer and greedy use (entropy_lm.c forward ->
// softmax -> blt_compute_entropy). ent[i] = H(x_{i+1} | x_{0..i}) in nats.
static void compute_entropy_vals(blt_arena *arena, const blt_entropy_lm *lm, const uint8_t *bytes, size_t len,
                                 float *ent_out) {
    size_t bytes_shape[1] = {len};
    blt_tensor bytes_in = blt_tensor_create(arena, bytes_shape, 1, BLT_DTYPE_UINT8);
    blt_tensor_upload(&bytes_in, bytes, len);

    size_t logits_shape[2] = {len, 256};
    blt_tensor logits = blt_tensor_create(arena, logits_shape, 2, BLT_DTYPE_FP32);
    size_t scalar_shape[1] = {1};
    blt_tensor discard_loss = blt_tensor_create(arena, scalar_shape, 1, BLT_DTYPE_FP32);
    blt_entropy_lm_forward(lm, &bytes_in, &logits, &discard_loss, arena);

    blt_tensor probs = blt_tensor_create(arena, logits_shape, 2, BLT_DTYPE_FP32);
    blt_softmax(&logits, &probs);

    size_t vals_shape[1] = {len};
    blt_tensor ent_t = blt_tensor_create(arena, vals_shape, 1, BLT_DTYPE_FP32);
    blt_entropy_config entropy_cfg = {.vocab_size = 256, .use_log2 = false};
    blt_compute_entropy(&probs, &ent_t, &entropy_cfg);

    if (ent_t.backend == BLT_BACKEND_CPU) {
        memcpy(ent_out, (const float *)ent_t.data, len * sizeof(float));
    } else {
        blt_tensor_download(&ent_t, ent_out, len * sizeof(float));
    }
}

// Paper-aligned segmentation: boundary at i uses entropy_data[i-1] (the paper's
// H(x_i|x_<i)) instead of entropy_data[i]. Same loop / emit structure and same
// max-patch-length cap as blt_segment_patches; the only change is the entropy
// index, isolating the off-by-one. reset_on_newline is false in the live config,
// so the newline branch is retained but inert.
static size_t segment_paper_aligned(const float *ent, const uint8_t *bytes, size_t len, float threshold,
                                    size_t max_patch_length, bool reset_on_newline, blt_patch_info *patches_out,
                                    size_t max_patches) {
    size_t patch_count = 0;
    size_t start = 0;
    size_t cur_len = 1;
    float peak = ent[0];

    for (size_t i = 1; i < len; ++i) {
        int boundary = 0;
        if (bytes && reset_on_newline && bytes[i] == 0x0A) {
            boundary = 1;
        } else if (cur_len >= max_patch_length) {
            boundary = 1;
        } else if (ent[i - 1] > threshold) {
            boundary = 1;
        }

        if (boundary) {
            if (patch_count >= max_patches) break;
            patches_out[patch_count].start_idx = start;
            patches_out[patch_count].length = cur_len;
            patches_out[patch_count].peak_entropy = peak;
            patch_count++;
            start = i;
            cur_len = 1;
            peak = ent[i];
        } else {
            cur_len++;
            if (ent[i] > peak) peak = ent[i];
        }
    }
    if (patch_count < max_patches) {
        patches_out[patch_count].start_idx = start;
        patches_out[patch_count].length = cur_len;
        patches_out[patch_count].peak_entropy = peak;
        patch_count++;
    }
    return patch_count;
}

// Flat LUT key: is_final (0/1) x patch_len (1..16) x prev_byte (0..255).
static size_t lut_key(int is_final, size_t patch_len, uint8_t prev_byte) {
    size_t pl = patch_len;
    if (pl < 1) pl = 1;
    if (pl > LEAK_MAX_PATCH_LEN) pl = LEAK_MAX_PATCH_LEN;
    return (size_t)is_final * (LEAK_MAX_PATCH_LEN * 256) + (pl - 1) * 256 + (size_t)prev_byte;
}

static size_t find_patch(const blt_patch_info *patches, size_t num_patches, size_t pos) {
    for (size_t j = 0; j < num_patches; j++) {
        size_t s = patches[j].start_idx;
        if (pos < s) break;
        if (pos < s + patches[j].length) return j;
    }
    return num_patches ? num_patches - 1 : 0;
}

static int is_final_row(const blt_patch_info *patches, size_t num_patches, size_t pos) {
    size_t j = find_patch(patches, num_patches, pos);
    return (pos + 1 == patches[j].start_idx + patches[j].length) ? 1 : 0;
}

// Accumulate one window's contribution into align_stats. ent is the cached
// per-byte entropy. is_train selects the fit half for the lookup table.
static void accumulate(align_stats *st, const uint8_t *bytes, size_t len, const blt_patch_info *patches,
                       size_t num_patches, int is_train) {
    st->n_windows++;
    st->n_bytes += len;
    st->n_patches += num_patches;
    if (num_patches) st->avg_patch_len += (double)len / (double)num_patches;

    for (size_t p = 0; p + 1 < len; ++p) {
        uint8_t target = bytes[p + 1];
        int fin = is_final_row(patches, num_patches, p);
        st->n_all_rows++;
        st->target_hist[target]++;

        if (fin) {
            st->n_final_rows++;
            st->final_hist[target]++;
            if (is_train) {
                st->train_final_hist[target]++;
                size_t j = find_patch(patches, num_patches, p);
                size_t key = lut_key(1, patches[j].length, bytes[p]);
                st->lut[key * LEAK_KEY_TARGETS + target]++;
            } else {
                st->test_final_hist[target]++;
                st->test_final_n++;
            }
        }
    }
}

static uint8_t hist_argmax(const uint64_t *h) {
    uint8_t best = 0;
    for (int v = 1; v < 256; v++)
        if (h[v] > h[best]) best = (uint8_t)v;
    return best;
}

static uint8_t lut_predict(const align_stats *st, size_t key) {
    const uint32_t *row = &st->lut[key * LEAK_KEY_TARGETS];
    uint8_t best = 0;
    uint32_t bestc = 0;
    for (int v = 0; v < 256; v++) {
        if (row[v] > bestc) {
            bestc = row[v];
            best = (uint8_t)v;
        }
    }
    return best;
}

static double hist_entropy_bits(const uint64_t *h, size_t n) {
    if (n == 0) return 0.0;
    double acc = 0.0;
    for (int v = 0; v < 256; v++) {
        if (h[v] == 0) continue;
        double pr = (double)h[v] / (double)n;
        acc -= pr * (log(pr) / log(2.0));
    }
    return acc;
}

static void top8(const uint64_t *h_src, size_t n) {
    uint64_t h[256];
    memcpy(h, h_src, sizeof(h));
    printf("    target distribution on final rows (top-8):\n");
    for (int rank = 0; rank < 8; rank++) {
        int best = -1;
        uint64_t bestc = 0;
        for (int v = 0; v < 256; v++) {
            if (h[v] > bestc) {
                bestc = h[v];
                best = v;
            }
        }
        if (best < 0 || bestc == 0) break;
        double share = (double)bestc / (double)n;
        printf("      0x%02X %3u  %6.2f%%\n", best, (unsigned)bestc, 100.0 * share);
        h[best] = 0;
    }
}

int main(int argc, char **argv) {
    const char *entropy_lm_path = NULL;
    const char *corpus = NULL;
    size_t window = 512;
    size_t num_windows = 256;
    float thr_repo = 2.5f;
    size_t max_patch_length = LEAK_MAX_PATCH_LEN;
    size_t calib_windows = 32;
    uint64_t seed = 7;
    int use_cuda = 0;
    long trace_window = -1;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--entropy-lm") && i + 1 < argc) entropy_lm_path = argv[++i];
        else if (!strcmp(argv[i], "--corpus") && i + 1 < argc) corpus = argv[++i];
        else if (!strcmp(argv[i], "--window") && i + 1 < argc) window = (size_t)atol(argv[++i]);
        else if (!strcmp(argv[i], "--windows") && i + 1 < argc) num_windows = (size_t)atol(argv[++i]);
        else if (!strcmp(argv[i], "--threshold-global") && i + 1 < argc) thr_repo = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--max-patch-length") && i + 1 < argc) max_patch_length = (size_t)atol(argv[++i]);
        else if (!strcmp(argv[i], "--calib-windows") && i + 1 < argc) calib_windows = (size_t)atol(argv[++i]);
        else if (!strcmp(argv[i], "--trace-window") && i + 1 < argc) trace_window = atol(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = (uint64_t)strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--backend") && i + 1 < argc) {
            i++;
            if (!strcmp(argv[i], "cuda")) use_cuda = 1;
            else if (strcmp(argv[i], "cpu") != 0) {
                fprintf(stderr, "Error: --backend must be cpu|cuda\n");
                return 1;
            }
        } else {
            fprintf(stderr, "Error: unknown/unpaired arg '%s'\n", argv[i]);
            return 1;
        }
    }

    if (!entropy_lm_path || !corpus) {
        fprintf(stderr,
                "Usage: leak_probe --entropy-lm FILE --corpus FILE [--window N] [--windows N] "
                "[--threshold-global F] [--max-patch-length N] [--calib-windows N] [--seed N] [--backend cpu]\n");
        return 1;
    }
    if (use_cuda) {
        fprintf(stderr, "FATAL: leak_probe supports only --backend cpu\n");
        return 1;
    }
    if (window < 2 || window > LEAK_MAX_SEQ || max_patch_length < 1 || max_patch_length > LEAK_MAX_PATCH_LEN) {
        fprintf(stderr, "Error: need 2 <= window <= %d, 1 <= max-patch-length <= %d\n", LEAK_MAX_SEQ,
                LEAK_MAX_PATCH_LEN);
        return 1;
    }
    if (num_windows < 4 || (num_windows % 2) != 0) {
        fprintf(stderr, "Error: --windows must be even and >= 4 (halves used for train/test split)\n");
        return 1;
    }

    long file_size = fsize(corpus);
    if (file_size <= 0) {
        fprintf(stderr, "Error: cannot read corpus '%s'\n", corpus);
        return 1;
    }
    size_t data_len = (size_t)file_size;
    uint8_t *data = (uint8_t *)malloc(data_len);
    if (!data) {
        fprintf(stderr, "FATAL: corpus alloc failed\n");
        return 1;
    }
    FILE *f = fopen(corpus, "rb");
    if (!f || fread(data, 1, data_len, f) != data_len) {
        fprintf(stderr, "Error: short read from '%s'\n", corpus);
        free(data);
        return 1;
    }
    fclose(f);

    const size_t usable = data_len / window;
    if (num_windows > usable) num_windows = usable - (usable % 2);
    if (num_windows < 4) {
        fprintf(stderr, "Error: corpus too small for 4 windows of %zu bytes\n", window);
        free(data);
        return 1;
    }

    blt_arena *lm_arena = blt_arena_create(64ULL * 1024 * 1024, BLT_BACKEND_CPU);
    blt_entropy_lm *lm = blt_make_entropy_lm(lm_arena, LEAK_MAX_SEQ, entropy_lm_path, seed);

    fprintf(stderr, "[LEAK] entropy LM: %s seed=%llu\n", entropy_lm_path, (unsigned long long)seed);
    fprintf(stderr, "[LEAK] corpus: %s (%zu bytes) window=%zu windows=%zu thr_repo=%.3f max_patch_len=%zu\n", corpus,
            data_len, window, num_windows, thr_repo, max_patch_length);

    // Cache per-byte entropy for every window once.
    float *ent_all = (float *)malloc(num_windows * window * sizeof(float));
    if (!ent_all) {
        fprintf(stderr, "FATAL: entropy cache alloc failed\n");
        return 1;
    }
    for (size_t w = 0; w < num_windows; w++) {
        size_t marker = lm_arena->offset;
        compute_entropy_vals(lm_arena, lm, data + w * window, window, &ent_all[w * window]);
        lm_arena->offset = marker; // bump arena: reclaim per-window temporaries
    }
    fprintf(stderr, "[LEAK] cached entropy for %zu windows\n", num_windows);

    blt_patcher_config pcfg;
    blt_make_patcher_cfg(&pcfg, 0, thr_repo, 1.0f, max_patch_length);

    // ---- Repo alignment ----
    align_stats repo;
    align_stats_init(&repo);
    blt_patch_info patches[LEAK_MAX_PATCHES];
    for (size_t w = 0; w < num_windows; w++) {
        blt_tensor ent_view;
        view_1d(&ent_view, &ent_all[w * window], window, BLT_DTYPE_FP32, BLT_BACKEND_CPU);
        size_t M = blt_segment_patches(&ent_view, data + w * window, patches, LEAK_MAX_PATCHES, &pcfg);
        accumulate(&repo, data + w * window, window, patches, M, w < num_windows / 2);
    }

    // ---- Optional trace of one window (both alignments) to verify indexing ----
    if (trace_window >= 0 && (size_t)trace_window < num_windows) {
        size_t w = (size_t)trace_window;
        const uint8_t *b = data + w * window;
        float *ent = &ent_all[w * window];
        blt_patch_info tp[LEAK_MAX_PATCHES];
        blt_tensor ent_view;
        view_1d(&ent_view, ent, window, BLT_DTYPE_FP32, BLT_BACKEND_CPU);
        size_t M = blt_segment_patches(&ent_view, b, tp, LEAK_MAX_PATCHES, &pcfg);
        fprintf(stderr, "\n[TRACE] window %zu  M=%zu  bytes: ", w, M);
        for (size_t i = 0; i < 48; i++) fprintf(stderr, "%c", (b[i] >= 32 && b[i] < 127) ? b[i] : '.');
        fprintf(stderr, "\n[TRACE] idx byte  ent[i]  ent[i-1]  is_final  target\n");
        for (size_t p = 0; p < 24; p++) {
            int fin = is_final_row(tp, M, p);
            char bc = (b[p] >= 32 && b[p] < 127) ? b[p] : '.';
            char tc = (b[p + 1] >= 32 && b[p + 1] < 127) ? b[p + 1] : '.';
            fprintf(stderr, "[TRACE] %3zu 0x%02X(%c)  %6.3f   %6.3f     %d     0x%02X(%c)%s\n", p, b[p], bc, ent[p],
                    ent[p ? p - 1 : 0], fin, b[p + 1], tc, fin ? "  <-- final" : "");
        }
    }

    // ---- Paper-aligned threshold: match repo avg patch length on a calib subset ----
    size_t calib = calib_windows < num_windows ? calib_windows : num_windows;
    double repo_avg_calib = 0.0;
    {
        size_t tot_b = 0, tot_p = 0;
        for (size_t w = 0; w < calib; w++) {
            blt_tensor ent_view;
            view_1d(&ent_view, &ent_all[w * window], window, BLT_DTYPE_FP32, BLT_BACKEND_CPU);
            size_t M = blt_segment_patches(&ent_view, data + w * window, patches, LEAK_MAX_PATCHES, &pcfg);
            tot_b += window;
            tot_p += M;
        }
        repo_avg_calib = tot_p ? (double)tot_b / (double)tot_p : 0.0;
    }
    float lo = 0.1f, hi = 8.0f;
    for (int it = 0; it < 40; it++) {
        float mid = 0.5f * (lo + hi);
        size_t tot_b = 0, tot_p = 0;
        for (size_t w = 0; w < calib; w++) {
            size_t M = segment_paper_aligned(&ent_all[w * window], data + w * window, window, mid, max_patch_length,
                                             pcfg.reset_on_newline, patches, LEAK_MAX_PATCHES);
            tot_b += window;
            tot_p += M;
        }
        double avg = tot_p ? (double)tot_b / (double)tot_p : 1e9;
        if (avg < repo_avg_calib) lo = mid;
        else hi = mid; // higher thr -> longer patches
    }
    float thr_paper = 0.5f * (lo + hi);
    fprintf(stderr, "[LEAK] repo avg patch len (calib %zu win) = %.3f -> paper thr tuned to %.4f\n", calib,
            repo_avg_calib, thr_paper);

    // ---- Paper alignment ----
    align_stats paper;
    align_stats_init(&paper);
    for (size_t w = 0; w < num_windows; w++) {
        size_t M = segment_paper_aligned(&ent_all[w * window], data + w * window, window, thr_paper, max_patch_length,
                                         pcfg.reset_on_newline, patches, LEAK_MAX_PATCHES);
        accumulate(&paper, data + w * window, window, patches, M, w < num_windows / 2);
    }

    // ---- Evaluate lookup-table predictors (train-fit, test-eval, final rows) ----
    align_stats *st[2] = {&repo, &paper};
    for (int a = 0; a < 2; a++) {
        align_stats *s = st[a];
        // Re-walk test windows to apply the fitted LUT.
        for (size_t w = num_windows / 2; w < num_windows; w++) {
            blt_patch_info tp[LEAK_MAX_PATCHES];
            size_t M;
            if (a == 0) {
                blt_tensor ent_view;
                view_1d(&ent_view, &ent_all[w * window], window, BLT_DTYPE_FP32, BLT_BACKEND_CPU);
                M = blt_segment_patches(&ent_view, data + w * window, tp, LEAK_MAX_PATCHES, &pcfg);
            } else {
                M = segment_paper_aligned(&ent_all[w * window], data + w * window, window, thr_paper, max_patch_length,
                                          pcfg.reset_on_newline, tp, LEAK_MAX_PATCHES);
            }
            for (size_t p = 0; p + 1 < window; p++) {
                if (!is_final_row(tp, M, p)) continue;
                size_t j = find_patch(tp, M, p);
                size_t key = lut_key(1, tp[j].length, data[w * window + p]);
                if (lut_predict(s, key) == data[w * window + p + 1]) s->test_final_correct++;
            }
        }
        // Constant predictor: most-common target among TRAIN final rows.
        uint8_t c = hist_argmax(s->train_final_hist);
        s->const_test_n = s->test_final_n;
        s->const_test_correct = 0;
        for (int v = 0; v < 256; v++)
            if (v == (int)c) s->const_test_correct += (size_t)s->test_final_hist[v];
    }

    // ---- Report ----
    const char *names[2] = {"REPO  (ent[i]   > thr)", "PAPER (ent[i-1] > thr)"};
    for (int a = 0; a < 2; a++) {
        align_stats *s = st[a];
        double avg = s->n_patches ? (double)s->n_bytes / (double)s->n_patches : 0.0;

        printf("\n===== %s =====\n", names[a]);
        printf("  threshold: %.4f   avg_patch_len: %.3f   windows: %zu   natural_final_rows: %zu (%.1f%% of %zu)\n",
               a == 0 ? thr_repo : thr_paper, avg, s->n_windows, s->n_final_rows,
               s->n_all_rows ? 100.0 * (double)s->n_final_rows / (double)s->n_all_rows : 0.0, s->n_all_rows);
        top8(s->final_hist, s->n_final_rows);

        double h_all = hist_entropy_bits(s->target_hist, s->n_all_rows);
        double h_fin = hist_entropy_bits(s->final_hist, s->n_final_rows);
        double drop = h_all - h_fin;
        printf("  H(x_{p+1}) over all rows   = %.4f bits\n", h_all);
        printf("  H(x_{p+1} | is_final)     = %.4f bits\n", h_fin);
        printf("  drop (all - |final)       = %+.4f bits  [info the segmentation carries about the target]\n", drop);

        double lut_acc = s->test_final_n ? (double)s->test_final_correct / (double)s->test_final_n : 0.0;
        double const_acc = s->const_test_n ? (double)s->const_test_correct / (double)s->const_test_n : 0.0;
        printf("  model-free lookup  f(is_final, patch_len, prev) top-1 on held-out final rows = %.4f (%zu/%zu)\n",
               lut_acc, s->test_final_correct, s->test_final_n);
        printf("  constant predictor (most-common train target)  top-1 on held-out final rows = %.4f\n", const_acc);
        printf("  [reference] trained checkpoint on natural final rows = 0.9938\n");
    }

    align_stats_free(&repo);
    align_stats_free(&paper);
    free(ent_all);
    free(data);
    blt_arena_destroy(lm_arena);
    return 0;
}
