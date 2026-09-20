// Evaluate prediction accuracy at row N-1 only (next byte after the prompt).
// Mirrors sanity_check.c forward pass, but only checks the last position.

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

static long fsize(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fclose(f);
    return sz;
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
    size_t num_windows = 20;
    size_t skip = 0;

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
    }

    if (!checkpoint || !corpus) {
        fprintf(stderr, "Usage: last_row_acc --checkpoint MODEL --corpus FILE [options]\n");
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
    fprintf(stderr, "[LAST_ROW] loaded checkpoint: %s\n", checkpoint);
    fprintf(stderr, "[LAST_ROW] config: embed=%zu hidden=%zu enc=%zu glob=%zu dec=%zu cross=%s diff=%d win=%zu\n",
            embed, hidden, enc_layers, glob_layers, dec_layers, cross_last ? "last" : "all", diffusion, window);

    blt_entropy_lm *el = NULL;
    if (entropy_lm_path) {
        el = blt_make_entropy_lm(arena, window, entropy_lm_path, 11);
        fprintf(stderr, "[LAST_ROW] loaded entropy LM: %s\n", entropy_lm_path);
    }

    blt_arena *eval_arena = blt_arena_create(1024ULL * 1024 * 1024, dev);

    double total_ce = 0.0;
    size_t total_samples = 0;
    size_t top1_correct = 0;
    size_t top5_correct = 0;
    size_t top10_correct = 0;

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

        // Evaluate ONLY the last row (row N-1) predicting text[window]
        size_t last_row = window - 1;
        const float *row = L + last_row * 256;
        int actual = (int)text[window];

        // CE at last position
        float max_logit = row[0];
        for (int v = 1; v < 256; v++)
            if (row[v] > max_logit) max_logit = row[v];
        float sum_exp = 0.0f;
        for (int v = 0; v < 256; v++) sum_exp += expf(row[v] - max_logit);
        float ce = -(row[actual] - max_logit) - logf(sum_exp);
        total_ce += ce;
        total_samples++;

        // Find top-10 at last position
        int top10_idx[10] = {0};
        float top10_val[10] = {0};
        for (int v = 0; v < 256; v++) {
            if (row[v] > top10_val[0]) {
                top10_val[0] = row[v];
                top10_idx[0] = v;
                for (int k = 1; k < 10; k++) {
                    if (top10_val[k - 1] > top10_val[k]) {
                        float tv = top10_val[k - 1];
                        top10_val[k - 1] = top10_val[k];
                        top10_val[k] = tv;
                        int ti = top10_idx[k - 1];
                        top10_idx[k - 1] = top10_idx[k];
                        top10_idx[k] = ti;
                    }
                }
            }
        }

        // Top-1 is the best (highest index after sort = index 9)
        if (top10_idx[9] == actual) top1_correct++;
        int in5 = 0, in10 = 0;
        for (int k = 0; k < 10; k++) {
            if (top10_idx[k] == actual) in10 = 1;
            if (k >= 5 && top10_idx[k] == actual) in5 = 1;
        }
        if (in5) top5_correct++;
        if (in10) top10_correct++;

        // Per-sample details for first 10 windows
        if (w < 10) {
            int best = top10_idx[9];
            char actual_ch = (actual >= 32 && actual < 127) ? (char)actual : '?';
            char pred_ch = (best >= 32 && best < 127) ? (char)best : '?';
            fprintf(stderr, "  [%3zu] actual='%c'(%3d)  pred='%c'(%3d)  top10=[", w, actual_ch, actual, pred_ch, best);
            for (int k = 9; k >= 0; k--) {
                char pc = (top10_idx[k] >= 32 && top10_idx[k] < 127) ? (char)top10_idx[k] : '?';
                fprintf(stderr, "%c(%d)", pc, top10_idx[k]);
                if (k > 0) fprintf(stderr, " ");
            }
            fprintf(stderr, "] %s\n", best == actual ? "OK" : "MISS");
        }

        free(L);
        blt_arena_reset(eval_arena);
    }

    double mean_ce = (total_samples > 0) ? total_ce / (double)total_samples : 0.0;
    double mean_bpb = mean_ce / log(2.0);
    double acc1 = (total_samples > 0) ? 100.0 * (double)top1_correct / (double)total_samples : 0.0;
    double acc5 = (total_samples > 0) ? 100.0 * (double)top5_correct / (double)total_samples : 0.0;
    double acc10 = (total_samples > 0) ? 100.0 * (double)top10_correct / (double)total_samples : 0.0;

    fprintf(stderr, "\n=== LAST-ROW ACCURACY (row N-1 only) ===\n");
    fprintf(stderr, "Samples evaluated: %zu\n", total_samples);
    fprintf(stderr, "Mean CE:    %.4f nats/sample\n", mean_ce);
    fprintf(stderr, "Mean BPB:   %.4f bits/sample\n", mean_bpb);
    fprintf(stderr, "Top-1:      %.2f%%\n", acc1);
    fprintf(stderr, "Top-5:      %.2f%%\n", acc5);
    fprintf(stderr, "Top-10:     %.2f%%\n", acc10);

    // Also print to stdout for machine parsing
    printf("samples=%zu ce=%.6f bpb=%.6f top1=%.2f top5=%.2f top10=%.2f\n", total_samples, mean_ce, mean_bpb, acc1,
           acc5, acc10);

    free(data);
    return 0;
}
