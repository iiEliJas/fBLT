// Teacher-forced frontier-accuracy diagnostic: under teacher forcing, measure
// the top-1 accuracy of predictions made from the frontier (currently-open)
// patch during plain greedy generation, bucketed by frontier patch length.
// Every round re-segments the full prefix with the generation patcher config
// (same code path as csrc/core/generate_greedy.c), then a final full-window
// segmentation annotates each predicted position with its authoritative patch
// length, closure type (natural/max/boundary) and final-byte flag.
//
// Usage: greedy_tf_acc --checkpoint MODEL --corpus FILE --dump OUT [options]

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include "core/allocator.h"
#include "core/platform.h"
#include "core/backend.h"
#include "core/cuda_shim.h"
#include "core/tensor.h"
#include "models/model.h"
#include "models/checkpoint.h"
#include "models/model_builder.h"
#include "models/entropy_lm.h"
#include "models/patcher.h"
#include "models/entropy.h"
#include "ops/softmax.h"
#include "train_eval.h"

#define GREEDY_TF_MAX_PATCHES 4096

// Patch closure types (mirror patch_final_split.c)
enum { CLOSURE_NAT = 0, CLOSURE_MAX, CLOSURE_BND, CLOSURE_COUNT };

static long fsize(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fclose(f);
    return sz;
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

// Find the patch index containing byte position i.
static size_t find_patch(const blt_patch_info *patches, size_t M, size_t i) {
    for (size_t k = 0; k < M; k++) {
        if (i >= patches[k].start_idx && i < patches[k].start_idx + patches[k].length) return k;
    }
    return 0;
}

static uint8_t argmax_byte(const float *row, size_t vocab_size) {
    size_t best = 0;
    float best_val = row[0];
    for (size_t v = 1; v < vocab_size; v++) {
        if (row[v] > best_val) {
            best_val = row[v];
            best = v;
        }
    }
    return (uint8_t)best;
}

// CUDA weight upload: the entropy LM and model are created/loaded on a host
// arena (blt_fill_small_uniform and checkpoint I/O write tensor data directly,
// which is invalid for device-resident tensors), then uploaded to device twins.
static void upload_model_weights(blt_model *dst, const blt_model *src) {
    const size_t n = blt_model_num_tensors(src);
    BLT_REQUIRE(n == blt_model_num_tensors(dst), "upload_model_weights: twin models disagree on tensor count");
    float *stage = NULL;
    size_t cap = 0;
    for (size_t i = 0; i < n; i++) {
        const char *name_src;
        blt_tensor *t_src;
        const char *name_dst;
        blt_tensor *t_dst;
        blt_model_tensor_at(src, i, &name_src, &t_src);
        blt_model_tensor_at(dst, i, &name_dst, &t_dst);
        BLT_REQUIRE(strcmp(name_src, name_dst) == 0, "upload_model_weights: tensor order mismatch at %zu", i);
        if (cap < t_src->numel) {
            free(stage);
            cap = t_src->numel;
            stage = (float *)malloc(cap * sizeof(float));
            BLT_REQUIRE(stage != NULL, "upload_model_weights: staging alloc failed");
        }
        blt_tensor_download(t_src, stage, blt_tensor_bytes(t_src));
        blt_tensor_upload(t_dst, stage, blt_tensor_bytes(t_dst));
    }
    free(stage);
}

static void upload_entropy_lm_weights(blt_entropy_lm *dst, const blt_entropy_lm *src) {
    blt_tensor_upload(&dst->embedding_weight, src->embedding_weight.data, blt_tensor_bytes(&src->embedding_weight));
    blt_tensor_upload(&dst->lm_head_weight, src->lm_head_weight.data, blt_tensor_bytes(&src->lm_head_weight));
    for (size_t i = 0; i < src->stack.num_layers; i++) {
        const blt_transformer_layer_storage *s = &src->stack.layer_storage[i];
        blt_transformer_layer_storage *d = &dst->stack.layer_storage[i];
        blt_tensor_upload(&d->norm1_weight, s->norm1_weight.data, blt_tensor_bytes(&s->norm1_weight));
        blt_tensor_upload(&d->attn_qkv_w, s->attn_qkv_w.data, blt_tensor_bytes(&s->attn_qkv_w));
        blt_tensor_upload(&d->attn_proj_w, s->attn_proj_w.data, blt_tensor_bytes(&s->attn_proj_w));
        blt_tensor_upload(&d->norm2_weight, s->norm2_weight.data, blt_tensor_bytes(&s->norm2_weight));
        blt_tensor_upload(&d->ffn_up_w, s->ffn_up_w.data, blt_tensor_bytes(&s->ffn_up_w));
        blt_tensor_upload(&d->ffn_gate_w, s->ffn_gate_w.data, blt_tensor_bytes(&s->ffn_gate_w));
        blt_tensor_upload(&d->ffn_down_w, s->ffn_down_w.data, blt_tensor_bytes(&s->ffn_down_w));
    }
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
    int use_cuda = 0;
    size_t verify_windows = 0;

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
        else if (!strcmp(argv[i], "--backend") && i + 1 < argc) {
            i++;
            if (!strcmp(argv[i], "cuda")) use_cuda = 1;
            else if (!strcmp(argv[i], "cpu")) use_cuda = 0;
            else {
                fprintf(stderr, "unknown --backend '%s'\n", argv[i]);
                return 1;
            }
        } else if (!strcmp(argv[i], "--verify-num-windows") && i + 1 < argc) verify_windows = (size_t)atol(argv[++i]);
    }

    if (!checkpoint || !corpus || !dump_path) {
        fprintf(stderr, "Usage: greedy_tf_acc --checkpoint MODEL --corpus FILE --dump OUT [options]\n");
        fprintf(stderr, "Options:\n");
        fprintf(stderr, "  --window W           Window size (default 512)\n");
        fprintf(stderr, "  --num-windows N      Number of windows to evaluate (default 50)\n");
        fprintf(stderr, "  --skip S             Window offset before first window (default 0)\n");
        fprintf(stderr, "  --entropy-lm PATH    Entropy LM for patching\n");
        fprintf(stderr, "  --max-patch-length L Max patch length for patching/classification (default 16)\n");
        fprintf(stderr, "  --dump PATH          Per-prediction dump output file (required)\n");
        fprintf(stderr, "  --embed/--hidden/--enc-layers/--glob-layers/--dec-layers  Model shape\n");
        fprintf(stderr, "  --cross-attn all|last  Cross-attention placement\n");
        fprintf(stderr, "  --diffusion 0|1      Diffusion mode (default 1)\n");
        fprintf(stderr, "  --backend cpu|cuda   Compute backend (default cpu)\n");
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
    blt_arena *host_arena = blt_arena_create(64 * 1024 * 1024, BLT_BACKEND_CPU);

    blt_model_config cfg;
    blt_model_config_defaults(&cfg, embed, hidden, enc_layers, glob_layers, dec_layers, window, cross_last);

    blt_model *model;
    blt_entropy_lm *el;
    if (use_cuda) {
        // Host twin receives the checkpoint; weights are uploaded to device.
        blt_model *host_model = blt_model_create(host_arena, &cfg);
        blt_model_load(host_model, checkpoint);
        model = blt_model_create(arena, &cfg);
        upload_model_weights(model, host_model);
        blt_entropy_lm *host_lm = blt_make_entropy_lm(host_arena, window, entropy_lm_path, 11);
        el = blt_entropy_lm_create(arena, &host_lm->config);
        upload_entropy_lm_weights(el, host_lm);
    } else {
        model = blt_model_create(arena, &cfg);
        blt_model_load(model, checkpoint);
        el = blt_make_entropy_lm(arena, window, entropy_lm_path, 11);
    }
    fprintf(stderr, "[GREEDY_TF] loaded checkpoint: %s\n", checkpoint);
    fprintf(stderr, "[GREEDY_TF] config: embed=%zu hidden=%zu enc=%zu glob=%zu dec=%zu cross=%s diff=%d win=%zu\n",
            embed, hidden, enc_layers, glob_layers, dec_layers, cross_last ? "last" : "all", diffusion, window);
    fprintf(stderr, "[GREEDY_TF] max_patch_length=%zu entropy_lm=%s backend=%s\n", max_patch_len,
            entropy_lm_path ? entropy_lm_path : "(none)", use_cuda ? "cuda" : "cpu");
    if (entropy_lm_path) {
        fprintf(stderr, "[GREEDY_TF] loaded entropy LM: %s\n", entropy_lm_path);
    }

    // Generation patcher config (infer.c defaults: thr_global=2.5, thr_mono=1.0,
    // max_patch_length, fixed=0, rule=global). Identical to entropy_segment's.
    blt_patcher_config pcfg;
    blt_make_patcher_cfg(&pcfg, 0, 2.5f, 1.0f, max_patch_len);

    blt_arena *scratch = blt_arena_create(1024ULL * 1024 * 1024, dev);
    size_t vocab_size = model->config.decoder_config.vocab_size;

    // Host staging buffers (allocated once; the scratch arena may be device
    // memory, which is not host-writable).
    uint8_t *out = (uint8_t *)malloc(window);
    size_t *frontier_len = (size_t *)malloc(window * sizeof(size_t));
    uint8_t *correct = (uint8_t *)malloc(window);
    float *entropy_host = (float *)malloc(window * sizeof(float));
    float *last_row = (float *)malloc(vocab_size * sizeof(float));
    uint8_t *round_pred = (uint8_t *)malloc(window);
    uint8_t *batch_pred = (uint8_t *)malloc(window);
    BLT_REQUIRE(out != NULL && frontier_len != NULL && correct != NULL && entropy_host != NULL && last_row != NULL &&
                    round_pred != NULL && batch_pred != NULL,
                "greedy_tf_acc: staging alloc failed");

    // 6 main cells: {final,nonfinal} x {nat,max,bnd}, each a (correct, count) pair
    size_t main_correct[6] = {0};
    size_t main_count[6] = {0};
    size_t total_correct = 0;
    size_t total_rows = 0;
    size_t total_windows = 0;

    // Verification-mode counters: round-l argmax vs batch full-window argmax
    size_t v_agree_final = 0, v_total_final = 0;
    size_t v_agree_nonfinal = 0, v_total_nonfinal = 0;
    size_t v_agree_bucket[5] = {0};
    size_t v_total_bucket[5] = {0};
    static const size_t v_bucket_threshold[5] = {2, 4, 8, 16, 999999};
    FILE *verify_file = NULL;
    if (verify_windows > 0) {
        char vpath[1024];
        snprintf(vpath, sizeof(vpath), "%s.verify", dump_path);
        verify_file = fopen(vpath, "w");
        BLT_REQUIRE(verify_file != NULL, "greedy_tf_acc: cannot open verify dump '%s'", vpath);
        fprintf(verify_file,
                "window\tpos\tfrontier_len\tfinal_patch_len\tclosure\tis_final\tround_pred\tbatch_pred\tagree\n");
    }

    FILE *dump_file = fopen(dump_path, "w");
    if (!dump_file) {
        fprintf(stderr, "Error: cannot open dump file '%s'\n", dump_path);
        return 1;
    }
    fprintf(dump_file, "window\tpos\tfrontier_len\tfinal_patch_len\tclosure\tis_final\tcorrect\n");

    for (size_t w = 0; w < num_windows; w++) {
        size_t offset = (skip + w) * window;
        if (offset + window > data_len) break;
        const uint8_t *text = data + offset;

        double w0 = blt_time_sec();

        // Teacher-forced prefix: prompt is text[0..2), then the true byte is
        // appended every round.
        memcpy(out, text, 2);

        // Round loop: predict byte at position cur_len from the frontier patch
        // of the full prefix [0..cur_len). Never predicts position window.
        for (size_t cur_len = 2; cur_len < window; cur_len++) {
            size_t step_marker = scratch->offset;

            // 1. Entropy LM over the full prefix (same as generate_greedy.c)
            size_t bytes_shape[1] = {cur_len};
            blt_tensor bytes_in = blt_tensor_create(scratch, bytes_shape, 1, BLT_DTYPE_UINT8);
            blt_tensor_upload(&bytes_in, out, cur_len);

            size_t entropy_logits_shape[2] = {cur_len, 256};
            blt_tensor entropy_logits = blt_tensor_create(scratch, entropy_logits_shape, 2, BLT_DTYPE_FP32);
            size_t scalar_shape[1] = {1};
            blt_tensor entropy_loss = blt_tensor_create(scratch, scalar_shape, 1, BLT_DTYPE_FP32);
            blt_entropy_lm_forward(el, &bytes_in, &entropy_logits, &entropy_loss, scratch);

            blt_tensor probs = blt_tensor_create(scratch, entropy_logits_shape, 2, BLT_DTYPE_FP32);
            blt_softmax(&entropy_logits, &probs);

            size_t entropy_shape[1] = {cur_len};
            blt_tensor entropy_vals = blt_tensor_create(scratch, entropy_shape, 1, BLT_DTYPE_FP32);
            blt_entropy_config entropy_cfg = {.vocab_size = 256, .use_log2 = false};
            blt_compute_entropy(&probs, &entropy_vals, &entropy_cfg);

            // 2. Segment the full prefix with the generation patcher config
            blt_tensor_download(&entropy_vals, entropy_host, cur_len * sizeof(float));
            blt_tensor entropy_host_view;
            view_1d(&entropy_host_view, entropy_host, cur_len, BLT_DTYPE_FP32, BLT_BACKEND_CPU);

            blt_patch_info patches[GREEDY_TF_MAX_PATCHES];
            size_t num_patches = blt_segment_patches(&entropy_host_view, out, patches, GREEDY_TF_MAX_PATCHES, &pcfg);

            // 3. Full encoder-global-decoder forward pass (single document)
            size_t vocab_shape[2] = {cur_len, vocab_size};
            blt_tensor model_logits = blt_tensor_create(scratch, vocab_shape, 2, BLT_DTYPE_FP32);
            blt_tensor discard_loss = blt_tensor_create(scratch, scalar_shape, 1, BLT_DTYPE_FP32);
            blt_model_forward(model, &bytes_in, NULL, patches, num_patches, NULL, 0, &model_logits, &discard_loss,
                              scratch);

            // 4. Argmax of the last position's logits vs ground truth
            blt_tensor last_row_view;
            view_1d(&last_row_view, (float *)model_logits.data + (cur_len - 1) * vocab_size, vocab_size, BLT_DTYPE_FP32,
                    model_logits.backend);
            blt_tensor_download(&last_row_view, last_row, vocab_size * sizeof(float));
            uint8_t pred = argmax_byte(last_row, vocab_size);

            // 5. Record frontier patch length and correctness
            frontier_len[cur_len] = patches[num_patches - 1].length;
            correct[cur_len] = (pred == text[cur_len]) ? 1 : 0;
            round_pred[cur_len] = pred;

            // 6. Append the TRUE byte (teacher forcing)
            out[cur_len] = text[cur_len];

            // 7. Arena reset per round (mirror generate_greedy.c)
            scratch->offset = step_marker;
#ifdef BLT_WITH_CUDA
            // Ops (gather_scatter, patch_pool, row_stats, reductions) allocate
            // temporaries from the thread-local CUDA scratch arena; reset it
            // per round so it stays bounded (train_blt_d.c resets per step).
            blt_cuda_scratch_reset();
#endif
        }

        // Authoritative full-window segmentation over [0..window) for labels
        blt_arena_reset(scratch);
        blt_patch_info full_patches[GREEDY_TF_MAX_PATCHES];
        size_t M_full = entropy_segment(scratch, el, text, window, full_patches, GREEDY_TF_MAX_PATCHES);
        if (M_full >= GREEDY_TF_MAX_PATCHES) {
            blt_arena_reset(scratch);
            continue;
        }

        // Verification: batch-style full-window forward, argmax per row
        if (verify_windows > 0) {
            size_t bshape[1] = {window};
            blt_tensor bytes_in = blt_tensor_create(scratch, bshape, 1, BLT_DTYPE_UINT8);
            blt_tensor_upload(&bytes_in, text, window);
            size_t vshape[2] = {window, vocab_size};
            blt_tensor vlogits = blt_tensor_create(scratch, vshape, 2, BLT_DTYPE_FP32);
            size_t vscalar_shape[1] = {1};
            blt_tensor vloss = blt_tensor_create(scratch, vscalar_shape, 1, BLT_DTYPE_FP32);
            blt_model_forward(model, &bytes_in, NULL, full_patches, M_full, NULL, 0, &vlogits, &vloss, scratch);
            for (size_t p = 0; p < window; p++) {
                blt_tensor row_view;
                view_1d(&row_view, (float *)vlogits.data + p * vocab_size, vocab_size, BLT_DTYPE_FP32, vlogits.backend);
                blt_tensor_download(&row_view, last_row, vocab_size * sizeof(float));
                batch_pred[p] = argmax_byte(last_row, vocab_size);
            }
        }

        // Annotate every predicted position p (2..window-1)
        for (size_t p = 2; p < window; p++) {
            size_t patch_k = find_patch(full_patches, M_full, p);
            int closure = classify_closure(full_patches, M_full, patch_k, max_patch_len);
            int is_final = (p == full_patches[patch_k].start_idx + full_patches[patch_k].length - 1);

            int cell = is_final * CLOSURE_COUNT + closure;
            main_correct[cell] += (size_t)correct[p];
            main_count[cell]++;
            total_correct += (size_t)correct[p];
            total_rows++;

            if (verify_windows > 0) {
                int agree = (round_pred[p] == batch_pred[p]) ? 1 : 0;
                if (is_final) {
                    v_agree_final += (size_t)agree;
                    v_total_final++;
                } else {
                    v_agree_nonfinal += (size_t)agree;
                    v_total_nonfinal++;
                }
                for (int lb = 0; lb < 5; lb++) {
                    if (full_patches[patch_k].length <= v_bucket_threshold[lb]) {
                        v_agree_bucket[lb] += (size_t)agree;
                        v_total_bucket[lb]++;
                        break;
                    }
                }
                fprintf(verify_file, "%zu\t%zu\t%zu\t%zu\t%s\t%d\t%d\t%d\t%d\n", skip + w, p, frontier_len[p],
                        full_patches[patch_k].length, closure_str(closure), is_final, round_pred[p], batch_pred[p],
                        agree);
            }

            fprintf(dump_file, "%zu\t%zu\t%zu\t%zu\t%s\t%d\t%d\n", skip + w, p, frontier_len[p],
                    full_patches[patch_k].length, closure_str(closure), is_final, correct[p]);
        }

        blt_arena_reset(scratch);
        total_windows++;
        fprintf(stderr, "[GREEDY_TF] window %zu/%zu (%.2fs)\n", w + 1, num_windows, blt_time_sec() - w0);
    }

    fclose(dump_file);
    if (verify_file) fclose(verify_file);
    fprintf(stderr, "[GREEDY_TF] wrote %zu rows to %s\n", total_rows, dump_path);

    if (verify_windows > 0) {
        fprintf(stderr, "\n=== VERIFY: round-l argmax vs batch full-window argmax ===\n");
        double fa = (v_total_final > 0) ? 100.0 * (double)v_agree_final / (double)v_total_final : 0.0;
        double na = (v_total_nonfinal > 0) ? 100.0 * (double)v_agree_nonfinal / (double)v_total_nonfinal : 0.0;
        fprintf(stderr, "final rows:    %zu/%zu = %.2f%%\n", v_agree_final, v_total_final, fa);
        fprintf(stderr, "nonfinal rows: %zu/%zu = %.2f%%\n", v_agree_nonfinal, v_total_nonfinal, na);
        for (int lb = 0; lb < 5; lb++) {
            double ba =
                (v_total_bucket[lb] > 0) ? 100.0 * (double)v_agree_bucket[lb] / (double)v_total_bucket[lb] : 0.0;
            fprintf(stderr, "bucket %d (len<=%zu): %zu/%zu = %.2f%%\n", lb, v_bucket_threshold[lb], v_agree_bucket[lb],
                    v_total_bucket[lb], ba);
        }
    }

    // Report
    fprintf(stderr, "\n=== GREEDY-TF FRONTIER ACCURACY (teacher-forced) ===\n");
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

    // Machine-readable output (stdout reserved for this line)
    printf("greedy_tf_final_nat=%zu greedy_tf_final_max=%zu greedy_tf_final_bnd=%zu greedy_tf_final_nat_c=%zu "
           "greedy_tf_final_max_c=%zu greedy_tf_final_bnd_c=%zu greedy_tf_nonfinal_nat=%zu "
           "greedy_tf_nonfinal_max=%zu greedy_tf_nonfinal_bnd=%zu greedy_tf_nonfinal_nat_c=%zu "
           "greedy_tf_nonfinal_max_c=%zu greedy_tf_nonfinal_bnd_c=%zu greedy_tf_total_correct=%zu "
           "greedy_tf_total_rows=%zu greedy_tf_windows=%zu\n",
           main_count[1 * CLOSURE_COUNT + CLOSURE_NAT], main_count[1 * CLOSURE_COUNT + CLOSURE_MAX],
           main_count[1 * CLOSURE_COUNT + CLOSURE_BND], main_correct[1 * CLOSURE_COUNT + CLOSURE_NAT],
           main_correct[1 * CLOSURE_COUNT + CLOSURE_MAX], main_correct[1 * CLOSURE_COUNT + CLOSURE_BND],
           main_count[0 * CLOSURE_COUNT + CLOSURE_NAT], main_count[0 * CLOSURE_COUNT + CLOSURE_MAX],
           main_count[0 * CLOSURE_COUNT + CLOSURE_BND], main_correct[0 * CLOSURE_COUNT + CLOSURE_NAT],
           main_correct[0 * CLOSURE_COUNT + CLOSURE_MAX], main_correct[0 * CLOSURE_COUNT + CLOSURE_BND], total_correct,
           total_rows, total_windows);

    free(data);
    free(out);
    free(frontier_len);
    free(correct);
    free(entropy_host);
    free(last_row);
    free(round_pred);
    free(batch_pred);
    return 0;
}