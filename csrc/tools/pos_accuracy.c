// Per-position prediction accuracy: reports top-1 accuracy at every row
// (0 through N-1) separately, to detect whether there's a cliff at the last row.

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
    size_t num_windows = 200;
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
        fprintf(stderr, "Usage: pos_accuracy --checkpoint MODEL --corpus FILE [options]\n");
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
    fprintf(stderr, "[POS_ACC] loaded checkpoint: %s\n", checkpoint);
    fprintf(stderr, "[POS_ACC] config: embed=%zu hidden=%zu enc=%zu glob=%zu dec=%zu cross=%s diff=%d win=%zu\n", embed,
            hidden, enc_layers, glob_layers, dec_layers, cross_last ? "last" : "all", diffusion, window);

    blt_entropy_lm *el = NULL;
    if (entropy_lm_path) {
        el = blt_make_entropy_lm(arena, window, entropy_lm_path, 11);
        fprintf(stderr, "[POS_ACC] loaded entropy LM: %s\n", entropy_lm_path);
    }

    blt_arena *eval_arena = blt_arena_create(1024ULL * 1024 * 1024, dev);

    size_t *pos_correct = (size_t *)calloc(window, sizeof(size_t));
    size_t *pos_count = (size_t *)calloc(window, sizeof(size_t));
    size_t actual_windows = 0;

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

        // Per-position top-1 accuracy
        for (size_t i = 0; i < window; i++) {
            const float *row = L + i * 256;
            int best = 0;
            for (int v = 1; v < 256; v++)
                if (row[v] > row[best]) best = v;
            int actual = (int)text[i + 1];
            pos_count[i]++;
            if (best == actual) pos_correct[i]++;
        }

        actual_windows++;
        free(L);
        blt_arena_reset(eval_arena);
    }

    fprintf(stderr, "\n=== PER-POSITION ACCURACY ===\n");
    fprintf(stderr, "Windows evaluated: %zu\n", actual_windows);

    for (size_t i = 0; i < window; i++) {
        double acc = (pos_count[i] > 0) ? 100.0 * (double)pos_correct[i] / (double)pos_count[i] : 0.0;
        fprintf(stderr, "pos=%-4zu acc=%.1f%%\n", i, acc);
    }

    // Summary stats
    double min_acc = 100.0, max_acc = 0.0, sum_acc = 0.0;
    size_t min_pos = 0, max_pos = 0;
    for (size_t i = 0; i < window; i++) {
        double acc = (pos_count[i] > 0) ? 100.0 * (double)pos_correct[i] / (double)pos_count[i] : 0.0;
        sum_acc += acc;
        if (acc < min_acc) {
            min_acc = acc;
            min_pos = i;
        }
        if (acc > max_acc) {
            max_acc = acc;
            max_pos = i;
        }
    }
    double mean_acc = sum_acc / (double)window;

    fprintf(stderr, "\nmin_acc=%.1f%% min_pos=%zu max_acc=%.1f%% max_pos=%zu mean_acc=%.1f%%\n", min_acc, min_pos,
            max_acc, max_pos, mean_acc);

    // Machine-readable output
    printf("min_acc=%.1f min_pos=%zu max_acc=%.1f max_pos=%zu mean_acc=%.1f\n", min_acc, min_pos, max_acc, max_pos,
           mean_acc);

    free(pos_correct);
    free(pos_count);
    free(data);
    return 0;
}
