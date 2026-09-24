// Per-row patch-final diagnostic: top-1 accuracy split by whether row i is the
// final byte of its patch, the patch closure type (natural/max/boundary), and
// the patch length bucket. Ground truth per row i is text[i+1].
//
// Usage: patch_final_split --checkpoint MODEL --corpus FILE --dump OUT [options]

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "core/allocator.h"
#include "core/backend.h"
#include "core/tensor.h"
#include "models/model.h"
#include "models/checkpoint.h"
#include "models/model_builder.h"
#include "train_eval.h"

// Patch closure types
enum { CLOSURE_NAT = 0, CLOSURE_MAX, CLOSURE_BND, CLOSURE_COUNT };

// Patch length buckets (1-2, 3-4, 5-8, 9-16, 17+)
#define LEN_BUCKET_COUNT 5
static const size_t len_bucket_threshold[LEN_BUCKET_COUNT] = {2, 4, 8, 16, 999999};

static long fsize(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fclose(f);
    return sz;
}

// Map a patch length to its bucket index.
static int len_bucket(size_t patch_len) {
    for (int s = 0; s < LEN_BUCKET_COUNT; s++) {
        if (patch_len <= len_bucket_threshold[s]) return s;
    }
    return LEN_BUCKET_COUNT - 1;
}

// Short closure label for dump lines and reports.
static const char *closure_str(int closure) {
    switch (closure) {
    case CLOSURE_NAT:
        return "nat";
    case CLOSURE_MAX:
        return "max";
    default:
        return "bnd";
    }
}

// Classify a patch's closure: boundary (last patch), max-length capped, or natural.
static int classify_closure(const blt_patch_info *patches, size_t M, size_t k, size_t max_patch_len) {
    if (k == M - 1) return CLOSURE_BND;
    if (patches[k].length >= max_patch_len) return CLOSURE_MAX;
    return CLOSURE_NAT;
}

// Find the patch index containing row i.
static size_t find_patch(const blt_patch_info *patches, size_t M, size_t i) {
    for (size_t k = 0; k < M; k++) {
        if (i >= patches[k].start_idx && i < patches[k].start_idx + patches[k].length) return k;
    }
    return 0;
}

// Accumulate one row into the 6 main cells and the 30 per-bucket cells.
// Cell index = is_final * CLOSURE_COUNT + closure.
static void accumulate_row(size_t *main_correct, size_t *main_count, size_t (*bucket_correct)[6],
                           size_t (*bucket_count)[6], size_t patch_len, int closure, int is_final, int correct) {
    int cell = is_final * CLOSURE_COUNT + closure;
    main_correct[cell] += (size_t)correct;
    main_count[cell]++;
    int lb = len_bucket(patch_len);
    bucket_correct[lb][cell] += (size_t)correct;
    bucket_count[lb][cell]++;
}

// Selftest assertion helper: prints the first failing field.
static int check_eq(const char *field, size_t expected, size_t got) {
    if (expected != got) {
        fprintf(stderr, "[SELFCHECK] FAIL %s: expected=%zu got=%zu\n", field, expected, got);
        return 0;
    }
    return 1;
}

// Classification-only self-check: synthetic 32-byte window, fixed_stride(32, 4),
// stub predictions exercise the accumulator without any model or disk I/O.
static int run_selftest(void) {
    size_t window = 32;
    size_t max_patch_len = 16;
    uint8_t text[32];
    for (size_t i = 0; i < window; i++) text[i] = (uint8_t)i;
    (void)text;

    blt_patch_info patches[128];
    size_t M = fixed_stride(window, 4, patches);

    size_t main_correct[6] = {0};
    size_t main_count[6] = {0};
    size_t bucket_correct[LEN_BUCKET_COUNT][6] = {{0}};
    size_t bucket_count[LEN_BUCKET_COUNT][6] = {{0}};
    size_t total_rows = 0;
    size_t windows = 0;

    for (size_t i = 0; i < window; i++) {
        size_t patch_k = find_patch(patches, M, i);
        int closure = classify_closure(patches, M, patch_k, max_patch_len);
        int is_final = (i == patches[patch_k].start_idx + patches[patch_k].length - 1);
        int correct = ((int)(i + patches[patch_k].start_idx) % 2 == 0);
        accumulate_row(main_correct, main_count, bucket_correct, bucket_count, patches[patch_k].length, closure,
                       is_final, correct);
        total_rows++;
    }
    windows++;

    if (!check_eq("final_nat", 7, main_count[1 * CLOSURE_COUNT + CLOSURE_NAT])) return 1;
    if (!check_eq("final_max", 0, main_count[1 * CLOSURE_COUNT + CLOSURE_MAX])) return 1;
    if (!check_eq("final_bnd", 1, main_count[1 * CLOSURE_COUNT + CLOSURE_BND])) return 1;
    if (!check_eq("nonfinal_nat", 21, main_count[0 * CLOSURE_COUNT + CLOSURE_NAT])) return 1;
    if (!check_eq("nonfinal_max", 0, main_count[0 * CLOSURE_COUNT + CLOSURE_MAX])) return 1;
    if (!check_eq("nonfinal_bnd", 3, main_count[0 * CLOSURE_COUNT + CLOSURE_BND])) return 1;
    if (!check_eq("windows", 1, windows)) return 1;
    if (!check_eq("total_rows", 32, total_rows)) return 1;

    fprintf(stderr, "[SELFCHECK] PASS\n");
    return 0;
}

int main(int argc, char **argv) {
    const char *checkpoint = NULL;
    const char *corpus = NULL;
    const char *entropy_lm_path = NULL;
    const char *dump_path = NULL;
    size_t window = 512;
    size_t embed = 256;
    size_t hidden = 512;
    size_t enc_layers = 2;
    size_t glob_layers = 6;
    size_t dec_layers = 2;
    int cross_last = 0;
    int diffusion = 1;
    size_t num_windows = 50;
    size_t skip = 0;
    size_t max_patch_len = 16;
    int selftest = 0;
    int use_cuda = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--checkpoint") && i + 1 < argc) checkpoint = argv[++i];
        else if (!strcmp(argv[i], "--corpus") && i + 1 < argc) corpus = argv[++i];
        else if (!strcmp(argv[i], "--window") && i + 1 < argc) window = (size_t)atol(argv[++i]);
        else if (!strcmp(argv[i], "--embed") && i + 1 < argc) embed = (size_t)atol(argv[++i]);
        else if (!strcmp(argv[i], "--hidden") && i + 1 < argc) hidden = (size_t)atol(argv[++i]);
        else if (!strcmp(argv[i], "--enc-layers") && i + 1 < argc) enc_layers = (size_t)atol(argv[++i]);
        else if (!strcmp(argv[i], "--glob-layers") && i + 1 < argc) glob_layers = (size_t)atol(argv[++i]);
        else if (!strcmp(argv[i], "--dec-layers") && i + 1 < argc) dec_layers = (size_t)atol(argv[++i]);
        else if (!strcmp(argv[i], "--cross-attn") && i + 1 < argc) {
            i++;
            if (!strcmp(argv[i], "last")) cross_last = 1;
        } else if (!strcmp(argv[i], "--diffusion") && i + 1 < argc) diffusion = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--num-windows") && i + 1 < argc) num_windows = (size_t)atol(argv[++i]);
        else if (!strcmp(argv[i], "--skip") && i + 1 < argc) skip = (size_t)atol(argv[++i]);
        else if (!strcmp(argv[i], "--entropy-lm") && i + 1 < argc) entropy_lm_path = argv[++i];
        else if (!strcmp(argv[i], "--max-patch-length") && i + 1 < argc) max_patch_len = (size_t)atol(argv[++i]);
        else if (!strcmp(argv[i], "--dump") && i + 1 < argc) dump_path = argv[++i];
        else if (!strcmp(argv[i], "--selftest")) selftest = 1;
        else if (!strcmp(argv[i], "--backend") && i + 1 < argc) {
            i++;
            if (!strcmp(argv[i], "cuda")) use_cuda = 1;
            else if (!strcmp(argv[i], "cpu")) use_cuda = 0;
            else {
                fprintf(stderr, "unknown --backend '%s'\n", argv[i]);
                return 1;
            }
        }
    }

    if (selftest) return run_selftest();

    if (!checkpoint || !corpus || !dump_path) {
        fprintf(stderr, "Usage: patch_final_split --checkpoint MODEL --corpus FILE --dump OUT [options]\n");
        fprintf(stderr, "Options:\n");
        fprintf(stderr, "  --window W           Window size (default 512)\n");
        fprintf(stderr, "  --num-windows N      Number of windows to evaluate (default 50)\n");
        fprintf(stderr, "  --skip S             Window offset before first window (default 0)\n");
        fprintf(stderr, "  --entropy-lm PATH    Entropy LM for patching\n");
        fprintf(stderr, "  --max-patch-length L Max patch length for classification (default 16)\n");
        fprintf(stderr, "  --dump PATH          Per-row dump output file (required)\n");
        fprintf(stderr, "  --embed/--hidden/--enc-layers/--glob-layers/--dec-layers  Model shape\n");
        fprintf(stderr, "  --cross-attn all|last  Cross-attention placement\n");
        fprintf(stderr, "  --diffusion 0|1      Diffusion mode (default 1)\n");
        fprintf(stderr, "  --backend cpu|cuda   Compute backend (default cpu)\n");
        fprintf(stderr, "  --selftest           Run classification self-check and exit\n");
        return 1;
    }

    long file_size = fsize(corpus);
    if (file_size <= 0) {
        fprintf(stderr, "Error: cannot read corpus '%s'\n", corpus);
        return 1;
    }
    size_t data_len = (size_t)file_size;
    uint8_t *data = (uint8_t *)malloc(data_len);
    FILE *f = fopen(corpus, "rb");
    fread(data, 1, data_len, f);
    fclose(f);

    blt_backend dev = use_cuda ? BLT_BACKEND_CUDA : BLT_BACKEND_CPU;
    blt_arena *arena = blt_arena_create(1024ULL * 1024 * 1024, dev);

    blt_model_config cfg;
    blt_model_config_defaults(&cfg, embed, hidden, enc_layers, glob_layers, dec_layers, window, cross_last);
    blt_model *model = blt_model_create(arena, &cfg);
    blt_model_load(model, checkpoint);
    fprintf(stderr, "[PATCH_FINAL] loaded checkpoint: %s\n", checkpoint);
    fprintf(stderr, "[PATCH_FINAL] config: embed=%zu hidden=%zu enc=%zu glob=%zu dec=%zu cross=%s diff=%d win=%zu\n",
            embed, hidden, enc_layers, glob_layers, dec_layers, cross_last ? "last" : "all", diffusion, window);
    fprintf(stderr, "[PATCH_FINAL] max_patch_length=%zu entropy_lm=%s backend=%s\n", max_patch_len,
            entropy_lm_path ? entropy_lm_path : "(none)", use_cuda ? "cuda" : "cpu");

    blt_entropy_lm *el = NULL;
    if (entropy_lm_path) {
        el = blt_make_entropy_lm(arena, window, entropy_lm_path, 11);
        fprintf(stderr, "[PATCH_FINAL] loaded entropy LM: %s\n", entropy_lm_path);
    }

    blt_arena *eval_arena = blt_arena_create(1024ULL * 1024 * 1024, dev);

    // 6 main cells: {final,nonfinal} x {nat,max,bnd}, each a (correct, count) pair
    size_t main_correct[6] = {0};
    size_t main_count[6] = {0};
    // 30 per-bucket cells: length bucket x the same 6 cells
    size_t bucket_correct[LEN_BUCKET_COUNT][6] = {{0}};
    size_t bucket_count[LEN_BUCKET_COUNT][6] = {{0}};
    size_t total_correct = 0;
    size_t total_rows = 0;
    size_t total_windows = 0;

    FILE *dump_file = fopen(dump_path, "w");
    if (!dump_file) {
        fprintf(stderr, "Error: cannot open dump file '%s'\n", dump_path);
        return 1;
    }
    fprintf(dump_file, "window\trow\tpatch_idx\tpatch_start\tpatch_len\tclosure\tis_final\tcorrect\n");

    for (size_t w = 0; w < num_windows; w++) {
        size_t offset = (skip + w) * window;
        if (offset + window + 1 > data_len) break;
        const uint8_t *text = data + offset;

        blt_patch_info patches[128];
        size_t M;
        if (el) {
            M = entropy_segment(eval_arena, el, text, window, patches, 128);
            if (M >= 128) {
                blt_arena_reset(eval_arena);
                continue;
            }
        } else {
            M = fixed_stride(window, 4, patches);
        }

        // Forward pass: encoder -> global -> decoder
        size_t bshape[1] = {window};
        blt_tensor bytes_in = blt_tensor_create(eval_arena, bshape, 1, BLT_DTYPE_UINT8);
        blt_tensor_upload(&bytes_in, text, window);

        size_t p_shape[2] = {M, model->config.encoder_config.embed_dim};
        blt_tensor P = blt_tensor_create(eval_arena, p_shape, 2, BLT_DTYPE_FP32);
        size_t h_shape[2] = {window, model->config.encoder_config.embed_dim};
        blt_tensor h = blt_tensor_create(eval_arena, h_shape, 2, BLT_DTYPE_FP32);
        blt_local_encoder_forward(model->encoder, &bytes_in, patches, M, NULL, 0, &P, &h, eval_arena);

        blt_tensor O = blt_tensor_create(eval_arena, p_shape, 2, BLT_DTYPE_FP32);
        blt_global_transformer_forward(model->global, &P, NULL, 0, &O, eval_arena);

        size_t lg[2] = {window, 256};
        blt_tensor logits = blt_tensor_create(eval_arena, lg, 2, BLT_DTYPE_FP32);
        size_t sc[1] = {1};
        blt_tensor loss = blt_tensor_create(eval_arena, sc, 1, BLT_DTYPE_FP32);
        blt_local_decoder_forward(model->decoder, &h, &O, patches, M, &bytes_in, NULL, NULL, 0, &logits, &loss,
                                  eval_arena);

        float *L = (float *)malloc(logits.numel * sizeof(float));
        blt_tensor_download(&logits, L, logits.numel * sizeof(float));

        // Classify every row 0..window-1 against ground truth text[i+1]
        for (size_t i = 0; i < window; i++) {
            size_t patch_k = find_patch(patches, M, i);
            int closure = classify_closure(patches, M, patch_k, max_patch_len);
            int is_final = (i == patches[patch_k].start_idx + patches[patch_k].length - 1);

            const float *row = L + i * 256;
            int best = 0;
            for (int v = 1; v < 256; v++)
                if (row[v] > row[best]) best = v;
            int actual = (int)text[i + 1];
            int correct = (best == actual);

            accumulate_row(main_correct, main_count, bucket_correct, bucket_count, patches[patch_k].length, closure,
                           is_final, correct);
            total_correct += (size_t)correct;
            total_rows++;

            fprintf(dump_file, "%zu\t%zu\t%zu\t%zu\t%zu\t%s\t%d\t%d\n", skip + w, i, patch_k,
                    patches[patch_k].start_idx, patches[patch_k].length, closure_str(closure), is_final, correct);
        }

        total_windows++;
        free(L);
        blt_arena_reset(eval_arena);

        if ((w + 1) % 25 == 0) fprintf(stderr, "[PATCH_FINAL] window %zu/%zu\n", w + 1, num_windows);
    }

    fclose(dump_file);
    fprintf(stderr, "[PATCH_FINAL] wrote %zu rows to %s\n", total_rows, dump_path);

    // Report
    fprintf(stderr, "\n=== PATCH-FINAL SPLIT (per-row top-1) ===\n");
    fprintf(stderr, "Windows evaluated: %zu\n", total_windows);
    fprintf(stderr, "Rows evaluated:    %zu\n", total_rows);
    fprintf(stderr, "Max patch length:  %zu\n\n", max_patch_len);

    fprintf(stderr, "Final-row accuracy by closure:\n");
    for (int c = 0; c < CLOSURE_COUNT; c++) {
        size_t cell = 1 * CLOSURE_COUNT + c;
        double acc = (main_count[cell] > 0) ? 100.0 * (double)main_correct[cell] / (double)main_count[cell] : 0.0;
        fprintf(stderr, "  %-10s %zu/%zu = %.2f%%\n", closure_str(c), main_correct[cell], main_count[cell], acc);
    }

    fprintf(stderr, "Non-final-row accuracy by closure:\n");
    for (int c = 0; c < CLOSURE_COUNT; c++) {
        size_t cell = 0 * CLOSURE_COUNT + c;
        double acc = (main_count[cell] > 0) ? 100.0 * (double)main_correct[cell] / (double)main_count[cell] : 0.0;
        fprintf(stderr, "  %-10s %zu/%zu = %.2f%%\n", closure_str(c), main_correct[cell], main_count[cell], acc);
    }

    size_t f_c = 0, f_n = 0, nf_c = 0, nf_n = 0;
    for (int c = 0; c < CLOSURE_COUNT; c++) {
        f_c += main_correct[1 * CLOSURE_COUNT + c];
        f_n += main_count[1 * CLOSURE_COUNT + c];
        nf_c += main_correct[0 * CLOSURE_COUNT + c];
        nf_n += main_count[0 * CLOSURE_COUNT + c];
    }
    double f_acc = (f_n > 0) ? 100.0 * (double)f_c / (double)f_n : 0.0;
    double nf_acc = (nf_n > 0) ? 100.0 * (double)nf_c / (double)nf_n : 0.0;
    fprintf(stderr, "\nOverall final:    %zu/%zu = %.2f%%\n", f_c, f_n, f_acc);
    fprintf(stderr, "Overall nonfinal: %zu/%zu = %.2f%%\n", nf_c, nf_n, nf_acc);

    fprintf(stderr, "\nKey gaps (final - nonfinal, pp):\n");
    for (int c = 0; c < CLOSURE_COUNT; c++) {
        size_t fcell = 1 * CLOSURE_COUNT + c;
        size_t ncell = 0 * CLOSURE_COUNT + c;
        double fa = (main_count[fcell] > 0) ? 100.0 * (double)main_correct[fcell] / (double)main_count[fcell] : 0.0;
        double na = (main_count[ncell] > 0) ? 100.0 * (double)main_correct[ncell] / (double)main_count[ncell] : 0.0;
        fprintf(stderr, "  %-10s %.2f pp\n", closure_str(c), fa - na);
    }

    fprintf(stderr, "\nLength-bucket accuracy (final / nonfinal):\n");
    for (int lb = 0; lb < LEN_BUCKET_COUNT; lb++) {
        size_t fc = 0, fn = 0, nc = 0, nn = 0;
        for (int cl = 0; cl < CLOSURE_COUNT; cl++) {
            fc += bucket_correct[lb][1 * CLOSURE_COUNT + cl];
            fn += bucket_count[lb][1 * CLOSURE_COUNT + cl];
            nc += bucket_correct[lb][0 * CLOSURE_COUNT + cl];
            nn += bucket_count[lb][0 * CLOSURE_COUNT + cl];
        }
        double fa = (fn > 0) ? 100.0 * (double)fc / (double)fn : 0.0;
        double na = (nn > 0) ? 100.0 * (double)nc / (double)nn : 0.0;
        fprintf(stderr, "  bucket %d: final %zu/%zu = %.2f%%   nonfinal %zu/%zu = %.2f%%\n", lb, fc, fn, fa, nc, nn,
                na);
    }

    // Machine-readable output (stdout reserved for this line)
    printf("final_nat=%zu final_max=%zu final_bnd=%zu final_nat_c=%zu final_max_c=%zu final_bnd_c=%zu "
           "nonfinal_nat=%zu nonfinal_max=%zu nonfinal_bnd=%zu nonfinal_nat_c=%zu nonfinal_max_c=%zu "
           "nonfinal_bnd_c=%zu total_correct=%zu total_rows=%zu windows=%zu\n",
           main_count[1 * CLOSURE_COUNT + CLOSURE_NAT], main_count[1 * CLOSURE_COUNT + CLOSURE_MAX],
           main_count[1 * CLOSURE_COUNT + CLOSURE_BND], main_correct[1 * CLOSURE_COUNT + CLOSURE_NAT],
           main_correct[1 * CLOSURE_COUNT + CLOSURE_MAX], main_correct[1 * CLOSURE_COUNT + CLOSURE_BND],
           main_count[0 * CLOSURE_COUNT + CLOSURE_NAT], main_count[0 * CLOSURE_COUNT + CLOSURE_MAX],
           main_count[0 * CLOSURE_COUNT + CLOSURE_BND], main_correct[0 * CLOSURE_COUNT + CLOSURE_NAT],
           main_correct[0 * CLOSURE_COUNT + CLOSURE_MAX], main_correct[0 * CLOSURE_COUNT + CLOSURE_BND], total_correct,
           total_rows, total_windows);

    free(data);
    return 0;
}