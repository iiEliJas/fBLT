// Eval and patching helper functions extracted from train_blt_d.c.
// window_causal_ce: per-window causal cross-entropy for eval.
// fixed_stride: fixed-stride patching helper.
// entropy_segment: entropy-LM-based patching for --entropy-patches.

#include "train_eval.h"

#include <math.h>
#include <stdlib.h>
#include "core/backend.h"
#include "core/tensor.h"
#include "ops/softmax.h"
#include "models/entropy.h"
#include "models/local_encoder.h"
#include "models/global_transformer.h"
#include "models/model_builder.h"
#include "models/local_decoder.h"
#include "models/patcher.h"
#include "models/block_diffusion.h"

double window_causal_ce(blt_arena *arena, const blt_model *model, const uint8_t *text, size_t N,
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
        blt_local_decoder_forward_diffusion(model->decoder, &h, &O, patches, M, text, NULL, batch_or_null, d0m, &logits,
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
        blt_local_decoder_forward(model->decoder, &h, &O, patches, M, &bytes_in, NULL, NULL, 0, &logits, &loss, arena);
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

size_t fixed_stride(size_t seq_len, size_t patch_len, blt_patch_info *out) {
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

size_t entropy_segment(blt_arena *arena, blt_entropy_lm *lm, const uint8_t *bytes, size_t len, blt_patch_info *out,
                       size_t max_patches) {
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
