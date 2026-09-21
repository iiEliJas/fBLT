// Patch-truncation split v2: sharpened §7.3 diagnostic.
//
// Three-way classification of interior-row top-1 accuracy:
//   (a) Natural        — entropy-triggered boundary, patch length < max_patch_length
//   (b) Max-length     — patch length >= max_patch_length, not last patch (hit the cap)
//   (c) Buffer-boundary — last patch in window (k == M-1)
//
// Buffer-boundary patches are further sub-bucketed by length (1-2, 3-4, 5-8, 9-16, 17+)
// to reveal how harsh truncation at the window boundary really is.
//
// Usage: patch_trunc --checkpoint MODEL --corpus FILE [options]

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

// Patch classification buckets
enum { BUCKET_NATURAL = 0, BUCKET_MAXLEN, BUCKET_BOUNDARY, BUCKET_COUNT };

// Buffer-boundary sub-bucket boundaries (patch length ranges)
#define BB_SUB_COUNT 5
static const char *bb_sub_labels[BB_SUB_COUNT] = {"len 1-2", "len 3-4", "len 5-8", "len 9-16", "len 17+"};

static size_t bb_sub_threshold[BB_SUB_COUNT] = {2, 4, 8, 16, 999999};

static long fsize(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fclose(f);
    return sz;
}

// Classify a patch into one of three buckets.
// Returns BUCKET_NATURAL, BUCKET_MAXLEN, or BUCKET_BOUNDARY.
static int classify_patch(const blt_patch_info *patches, size_t M, size_t k, size_t max_patch_len) {
    if (k == M - 1) return BUCKET_BOUNDARY;
    if (patches[k].length >= max_patch_len) return BUCKET_MAXLEN;
    return BUCKET_NATURAL;
}

// Map patch length to a buffer-boundary sub-bucket index.
static int bb_sub_bucket(size_t patch_len) {
    for (int s = 0; s < BB_SUB_COUNT; s++) {
        if (patch_len <= bb_sub_threshold[s]) return s;
    }
    return BB_SUB_COUNT - 1;
}

int main(int argc, char **argv) {
    const char *checkpoint = NULL;
    const char *corpus = NULL;
    const char *entropy_lm_path = NULL;
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
    }

    if (!checkpoint || !corpus) {
        fprintf(stderr, "Usage: patch_trunc --checkpoint MODEL --corpus FILE [options]\n");
        fprintf(stderr, "Options:\n");
        fprintf(stderr, "  --window W           Window size (default 512)\n");
        fprintf(stderr, "  --num-windows N      Number of windows to evaluate (default 50)\n");
        fprintf(stderr, "  --skip S             Byte offset before first window (default 0)\n");
        fprintf(stderr, "  --entropy-lm PATH    Entropy LM for patching\n");
        fprintf(stderr, "  --max-patch-length L Max patch length for classification (default 16)\n");
        fprintf(stderr, "  --embed/--hidden/--enc-layers/--glob-layers/--dec-layers  Model shape\n");
        fprintf(stderr, "  --cross-attn all|last  Cross-attention placement\n");
        fprintf(stderr, "  --diffusion 0|1      Diffusion mode (default 1)\n");
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

    blt_backend dev = BLT_BACKEND_CPU;
    blt_arena *arena = blt_arena_create(1024ULL * 1024 * 1024, dev);

    blt_model_config cfg;
    blt_model_config_defaults(&cfg, embed, hidden, enc_layers, glob_layers, dec_layers, window, cross_last);
    blt_model *model = blt_model_create(arena, &cfg);
    blt_model_load(model, checkpoint);
    fprintf(stderr, "[PATCH_V2] loaded checkpoint: %s\n", checkpoint);
    fprintf(stderr, "[PATCH_V2] config: embed=%zu hidden=%zu enc=%zu glob=%zu dec=%zu cross=%s diff=%d win=%zu\n",
            embed, hidden, enc_layers, glob_layers, dec_layers, cross_last ? "last" : "all", diffusion, window);
    fprintf(stderr, "[PATCH_V2] max_patch_length=%zu entropy_lm=%s\n", max_patch_len,
            entropy_lm_path ? entropy_lm_path : "(none)");

    blt_entropy_lm *el = NULL;
    if (entropy_lm_path) {
        el = blt_make_entropy_lm(arena, window, entropy_lm_path, 11);
        fprintf(stderr, "[PATCH_V2] loaded entropy LM: %s\n", entropy_lm_path);
    }

    blt_arena *eval_arena = blt_arena_create(1024ULL * 1024 * 1024, dev);

    // Main bucket accumulators: correct and total for each of the 3 buckets
    size_t main_correct[BUCKET_COUNT];
    size_t main_count[BUCKET_COUNT];
    memset(main_correct, 0, sizeof(main_correct));
    memset(main_count, 0, sizeof(main_count));

    // Buffer-boundary sub-bucket accumulators
    size_t bb_correct[BB_SUB_COUNT];
    size_t bb_count[BB_SUB_COUNT];
    memset(bb_correct, 0, sizeof(bb_correct));
    memset(bb_count, 0, sizeof(bb_count));

    // Patch count diagnostics per window
    size_t total_patches[BUCKET_COUNT];
    memset(total_patches, 0, sizeof(total_patches));
    size_t total_windows = 0;

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

        // Count patches by type for this window
        for (size_t k = 0; k < M; k++) {
            int bucket = classify_patch(patches, M, k, max_patch_len);
            total_patches[bucket]++;
        }

        // Evaluate interior rows (0..N-2 predicting bytes 1..N-1)
        for (size_t i = 0; i + 1 < window; i++) {
            // Find which patch contains row i
            size_t patch_k = 0;
            for (size_t k = 0; k < M; k++) {
                if (i >= patches[k].start_idx && i < patches[k].start_idx + patches[k].length) {
                    patch_k = k;
                    break;
                }
            }

            int bucket = classify_patch(patches, M, patch_k, max_patch_len);

            // Top-1 accuracy at this row
            const float *row = L + i * 256;
            int best = 0;
            for (int v = 1; v < 256; v++)
                if (row[v] > row[best]) best = v;
            int actual = (int)text[i + 1];
            int correct = (best == actual);

            main_correct[bucket] += (size_t)correct;
            main_count[bucket]++;

            // Buffer-boundary sub-bucketing
            if (bucket == BUCKET_BOUNDARY) {
                int sub = bb_sub_bucket(patches[patch_k].length);
                bb_correct[sub] += (size_t)correct;
                bb_count[sub]++;
            }
        }

        total_windows++;
        free(L);
        blt_arena_reset(eval_arena);
    }

    // Totals
    size_t grand_correct = 0, grand_count = 0;
    for (int b = 0; b < BUCKET_COUNT; b++) {
        grand_correct += main_correct[b];
        grand_count += main_count[b];
    }

    // Report results
    fprintf(stderr, "\n=== PATCH-TRUNCATION SPLIT V2 (Sharpened §7.3) ===\n");
    fprintf(stderr, "Windows evaluated: %zu\n", total_windows);
    fprintf(stderr, "Max patch length:  %zu\n\n", max_patch_len);

    fprintf(stderr, "Patch counts by type:\n");
    size_t total_p = 0;
    for (int b = 0; b < BUCKET_COUNT; b++) total_p += total_patches[b];
    fprintf(stderr, "  Natural (entropy-triggered):   %zu\n", total_patches[BUCKET_NATURAL]);
    fprintf(stderr, "  Max-length capped (len=%zu):    %zu\n", max_patch_len, total_patches[BUCKET_MAXLEN]);
    fprintf(stderr, "  Buffer-boundary (last patch):  %zu\n", total_patches[BUCKET_BOUNDARY]);
    fprintf(stderr, "  Total patches:                 %zu\n\n", total_p);

    fprintf(stderr, "Main buckets (interior rows):\n");
    for (int b = 0; b < BUCKET_COUNT; b++) {
        const char *label;
        switch (b) {
        case BUCKET_NATURAL:
            label = "Natural (entropy-triggered)";
            break;
        case BUCKET_MAXLEN:
            label = "Max-length capped";
            break;
        case BUCKET_BOUNDARY:
            label = "Buffer-boundary (last patch)";
            break;
        default:
            label = "???";
            break;
        }
        double acc = (main_count[b] > 0) ? 100.0 * (double)main_correct[b] / (double)main_count[b] : 0.0;
        fprintf(stderr, "  %-32s %zu/%zu = %.2f%%\n", label, main_correct[b], main_count[b], acc);
    }
    double overall = (grand_count > 0) ? 100.0 * (double)grand_correct / (double)grand_count : 0.0;
    fprintf(stderr, "  %-32s %zu/%zu = %.2f%%\n", "Overall", grand_correct, grand_count, overall);

    fprintf(stderr, "\nBuffer-boundary sub-buckets by patch length:\n");
    for (int s = 0; s < BB_SUB_COUNT; s++) {
        if (bb_count[s] == 0) {
            fprintf(stderr, "  %-10s  (no samples)\n", bb_sub_labels[s]);
            continue;
        }
        double acc = 100.0 * (double)bb_correct[s] / (double)bb_count[s];
        fprintf(stderr, "  %-10s  %zu/%zu = %.2f%%\n", bb_sub_labels[s], bb_correct[s], bb_count[s], acc);
    }

    // Key gaps
    double nat_acc = (main_count[BUCKET_NATURAL] > 0)
                         ? 100.0 * (double)main_correct[BUCKET_NATURAL] / (double)main_count[BUCKET_NATURAL]
                         : 0.0;
    double len_acc = (main_count[BUCKET_MAXLEN] > 0)
                         ? 100.0 * (double)main_correct[BUCKET_MAXLEN] / (double)main_count[BUCKET_MAXLEN]
                         : 0.0;
    double bnd_acc = (main_count[BUCKET_BOUNDARY] > 0)
                         ? 100.0 * (double)main_correct[BUCKET_BOUNDARY] / (double)main_count[BUCKET_BOUNDARY]
                         : 0.0;

    fprintf(stderr, "\nKey gaps:\n");
    fprintf(stderr, "  Natural - Max-length:  %.2f pp\n", nat_acc - len_acc);
    fprintf(stderr, "  Natural - Boundary:    %.2f pp\n", nat_acc - bnd_acc);
    fprintf(stderr, "  Max-length - Boundary: %.2f pp\n", len_acc - bnd_acc);

    // Machine-readable output
    printf("windows=%zu max_patch_len=%zu nat_patches=%zu len_patches=%zu bnd_patches=%zu "
           "nat_acc=%.2f len_acc=%.2f bnd_acc=%.2f overall=%.2f\n",
           total_windows, max_patch_len, total_patches[BUCKET_NATURAL], total_patches[BUCKET_MAXLEN],
           total_patches[BUCKET_BOUNDARY], nat_acc, len_acc, bnd_acc, overall);

    free(data);
    return 0;
}
