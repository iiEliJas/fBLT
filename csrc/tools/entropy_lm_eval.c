// entropy_lm_eval: evaluate a trained entropy LM on a byte corpus.
//
// Reports:
//   - average next-byte cross-entropy (nats and bits)
//   - patch count statistics produced by the entropy patcher
//   - how often the patcher hits its patch cap
//
// This is a host-only tool: the entropy LM always runs on the CPU.

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "core/allocator.h"
#include "core/backend.h"
#include "core/tensor.h"
#include "models/entropy_lm.h"
#include "models/model_builder.h"
#include "models/checkpoint.h"
#include "models/patcher.h"

static long file_size(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fclose(f);
    return sz;
}

static void usage(void) {
    fprintf(stderr, "usage: entropy_lm_eval --entropy-lm FILE --corpus FILE\n"
                    "                      [--window N] [--windows N] [--skip N]\n"
                    "                      [--threshold F] [--max-patch-len N]\n");
}

int main(int argc, char **argv) {
    const char *lm_path = NULL;
    const char *corpus_path = NULL;
    size_t window = 512;
    size_t n_windows = 100;
    size_t skip = 0;
    float threshold = 3.0f;
    size_t max_patch_len = 16;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--entropy-lm") && i + 1 < argc) lm_path = argv[++i];
        else if (!strcmp(argv[i], "--corpus") && i + 1 < argc) corpus_path = argv[++i];
        else if (!strcmp(argv[i], "--window") && i + 1 < argc) window = (size_t)strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--windows") && i + 1 < argc) n_windows = (size_t)strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--skip") && i + 1 < argc) skip = (size_t)strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--threshold") && i + 1 < argc) threshold = strtof(argv[++i], NULL);
        else if (!strcmp(argv[i], "--max-patch-len") && i + 1 < argc)
            max_patch_len = (size_t)strtoull(argv[++i], NULL, 10);
        else {
            usage();
            return 2;
        }
    }

    if (!lm_path || !corpus_path || window < 2) {
        usage();
        return 2;
    }

    long sz = file_size(corpus_path);
    if (sz < 0) {
        fprintf(stderr, "entropy_lm_eval: cannot open corpus '%s'\n", corpus_path);
        return 1;
    }

    FILE *f = fopen(corpus_path, "rb");
    if (!f) {
        fprintf(stderr, "entropy_lm_eval: cannot open corpus '%s'\n", corpus_path);
        return 1;
    }
    const size_t avail = (size_t)sz > skip ? (size_t)sz - skip : 0;
    if (avail < window) {
        fprintf(stderr, "entropy_lm_eval: corpus too small for one window\n");
        fclose(f);
        return 1;
    }
    size_t possible = avail / window;
    if (n_windows > possible) n_windows = possible;
    if (n_windows == 0) {
        fprintf(stderr, "entropy_lm_eval: no windows available\n");
        fclose(f);
        return 1;
    }

    blt_arena *model_arena = blt_arena_create(64 * 1024 * 1024, BLT_BACKEND_CPU);
    blt_arena *scratch = blt_arena_create(64 * 1024 * 1024, BLT_BACKEND_CPU);

    blt_entropy_lm_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.embed_dim = 32;
    cfg.num_layers = 1;
    cfg.num_heads = 2;
    cfg.hidden_dim = 64;
    cfg.max_seq_len = window;
    cfg.rope_theta = 10000.0f;

    blt_entropy_lm *lm = blt_entropy_lm_create(model_arena, &cfg);
    blt_entropy_lm_load(lm, lm_path);

    blt_patcher_config pcfg;
    memset(&pcfg, 0, sizeof(pcfg));
    pcfg.threshold_global = threshold;
    pcfg.threshold_monotonic = 1.0f;
    pcfg.max_patch_length = max_patch_len;
    pcfg.rule = BLT_PATCH_RULE_GLOBAL;

    double loss_sum = 0.0;
    size_t patch_sum = 0;
    size_t patch_min = (size_t)-1;
    size_t patch_max = 0;
    size_t cap_hits = 0;
    size_t n_done = 0;

    uint8_t *buf = (uint8_t *)malloc(window);
    if (!buf) {
        fprintf(stderr, "entropy_lm_eval: out of memory\n");
        return 1;
    }

    for (size_t wi = 0; wi < n_windows; wi++) {
        fseek(f, (long)(skip + wi * window), SEEK_SET);
        if (fread(buf, 1, window, f) != window) {
            fprintf(stderr, "entropy_lm_eval: short read at window %zu\n", wi);
            break;
        }

        blt_arena_reset(scratch);
        size_t shape1[1] = {window};
        blt_tensor bytes_in = blt_tensor_create(scratch, shape1, 1, BLT_DTYPE_UINT8);
        memcpy(bytes_in.data, buf, window);

        size_t logits_shape[2] = {window, 256};
        blt_tensor logits = blt_tensor_create(scratch, logits_shape, 2, BLT_DTYPE_FP32);
        size_t scalar_shape[1] = {1};
        blt_tensor loss = blt_tensor_create(scratch, scalar_shape, 1, BLT_DTYPE_FP32);

        blt_entropy_lm_forward(lm, &bytes_in, &logits, &loss, scratch);

        float lv = 0.0f;
        blt_tensor_download(&loss, &lv, sizeof(float));
        loss_sum += (double)lv;

        // Softmax + entropy on the host.
        float *logits_host = (float *)malloc(window * 256 * sizeof(float));
        float *entropy_host = (float *)malloc(window * sizeof(float));
        if (!logits_host || !entropy_host) {
            fprintf(stderr, "entropy_lm_eval: out of memory\n");
            return 1;
        }
        blt_tensor_download(&logits, logits_host, window * 256 * sizeof(float));

        for (size_t i = 0; i < window; i++) {
            float *row = logits_host + i * 256;
            float mx = row[0];
            for (size_t v = 1; v < 256; v++)
                if (row[v] > mx) mx = row[v];
            float sum = 0.0f;
            for (size_t v = 0; v < 256; v++) sum += expf(row[v] - mx);
            float ent = 0.0f;
            for (size_t v = 0; v < 256; v++) {
                float p = expf(row[v] - mx) / sum;
                if (p > 1e-9f) ent -= p * logf(p);
            }
            entropy_host[i] = ent;
        }

        blt_tensor entropy_t;
        view_1d(&entropy_t, entropy_host, window, BLT_DTYPE_FP32, BLT_BACKEND_CPU);

        blt_patch_info patches[1024];
        size_t n_patches = blt_segment_patches(&entropy_t, buf, patches, 1024, &pcfg);

        patch_sum += n_patches;
        if (n_patches < patch_min) patch_min = n_patches;
        if (n_patches > patch_max) patch_max = n_patches;
        if (n_patches >= 1024) cap_hits++;

        free(logits_host);
        free(entropy_host);
        n_done++;
    }

    fclose(f);

    if (n_done == 0) {
        fprintf(stderr, "entropy_lm_eval: no windows processed\n");
        return 1;
    }

    const double avg_loss = loss_sum / (double)n_done;
    const double avg_patches = (double)patch_sum / (double)n_done;
    const double avg_bytes_per_patch = (double)window * (double)n_done / (double)patch_sum;

    printf("entropy_lm_eval: %s\n", lm_path);
    printf("corpus: %s\n", corpus_path);
    printf("windows: %zu (skip=%zu, window=%zu)\n", n_done, skip, window);
    printf("threshold: %.3f  max_patch_len: %zu\n", threshold, max_patch_len);
    printf("avg cross-entropy: %.4f nats  (%.4f bits)\n", avg_loss, avg_loss / 0.69314718056);
    printf("patches per window: avg=%.1f  min=%zu  max=%zu\n", avg_patches, patch_min, patch_max);
    printf("avg bytes per patch: %.2f\n", avg_bytes_per_patch);
    printf("patch-cap hits: %zu / %zu\n", cap_hits, n_done);

    return 0;
}
