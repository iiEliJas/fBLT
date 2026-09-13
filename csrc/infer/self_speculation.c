// BLT-S: self-speculative greedy generation (Fast-BLT 5.1, Algorithm 2).
// Round structure: segment prefix -> freeze latents -> cache maintenance ->
// prefill chunk -> draft -> verify. Completed patches are immutable across
// rounds (prefix-stable patcher + causal models).
#include "infer/self_speculation.h"

#include "infer/kv_cache.h"
#include "core/backend.h"
#include "models/entropy.h"
#include "models/local_common.h"
#include "ops/softmax.h"
#include "ops/vecmath.h"
#include "ops/row_stats.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define BLT_SELFSPEC_MAX_PATCHES 4096

static uint8_t argmax_byte_host_row(const float *row, size_t vocab_size) {
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
#define argmax_byte_host_check(row, vocab) argmax_byte_host_row((row), (vocab))
static uint8_t argmax_byte_host(const uint32_t ids[1]) { return (uint8_t)ids[0]; }

// The patcher runs on the host, so device entropy tensors get staged through
// host memory. Returns a malloc'd buffer the caller must free.
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

// Public so the commit-selection rule stays unit-testable without a model.
size_t blt_aligned_commit_select(const blt_patch_info *patches, size_t num_patches, size_t lo, size_t hi) {
    size_t best = 0;
    for (size_t i = 0; i < num_patches; i++) {
        const size_t e = patches[i].start_idx + patches[i].length;
        if (e > lo && e <= hi && e > best) best = e;
    }
    return best;
}

size_t blt_verify_draft(const blt_model *model, const blt_entropy_lm *entropy_model,
                        const blt_patcher_config *patcher_config, uint8_t *x, size_t l, size_t r, size_t target_len,
                        blt_infer_stats *stats, blt_arena *arena) {
    BLT_REQUIRE(model != NULL && entropy_model != NULL && patcher_config != NULL && x != NULL && arena != NULL,
                "blt_verify_draft: arguments cannot be NULL");
    BLT_REQUIRE(l >= 1, "blt_verify_draft: committed length l must be >= 1");
    const size_t cand_len = l + r;
    BLT_REQUIRE(cand_len >= 2 && cand_len <= model->config.decoder_config.max_seq_len,
                "blt_verify_draft: candidate length must be in [2, max_seq_len]");

    const size_t vocab_size = model->config.decoder_config.vocab_size;

    // Segment candidate into M' patches (Algorithm 2 line 1).
    blt_tensor entropy_vals;
    compute_entropy_vals(arena, entropy_model, x, cand_len, &entropy_vals);

    blt_tensor entropy_host_view;
    float *entropy_host_buf = NULL;
    const blt_tensor *entropy_for_patcher = stage_entropy_host(&entropy_vals, &entropy_host_view, &entropy_host_buf);

    blt_patch_info patches[BLT_SELFSPEC_MAX_PATCHES];
    size_t num_patches = blt_segment_patches(entropy_for_patcher, x, patches, BLT_SELFSPEC_MAX_PATCHES, patcher_config);
    free(entropy_host_buf);
    entropy_host_buf = NULL;

    // Force patch boundary at commit point l — without this, the prefix
    // patch absorbs draft bytes and verification diverges from greedy.
    {
        blt_patch_info forced[BLT_SELFSPEC_MAX_PATCHES];
        size_t n2 = 0;
        for (size_t pi = 0; pi < num_patches && n2 < BLT_SELFSPEC_MAX_PATCHES; pi++) {
            const size_t s = patches[pi].start_idx;
            const size_t e = s + patches[pi].length;
            if (s < l && e > l) {
                forced[n2++] = patches[pi]; // prefix part [s, l)
                forced[n2 - 1].length = l - s;
                if (n2 < BLT_SELFSPEC_MAX_PATCHES) {
                    forced[n2] = patches[pi]; // draft part [l, e)
                    forced[n2].start_idx = l;
                    forced[n2].length = e - l;
                    forced[n2].peak_entropy = patches[pi].peak_entropy;
                    n2++;
                }
            } else {
                forced[n2++] = patches[pi];
            }
        }
        BLT_REQUIRE(n2 <= BLT_SELFSPEC_MAX_PATCHES, "blt_verify_draft: forced-boundary patch array overflow");
        num_patches = n2;
        memcpy(patches, forced, n2 * sizeof(blt_patch_info));
    }

    // Full forward: E(x'), O'=G(T'), y=D(x'; O') — Algorithm 2 line 2.
    size_t bytes_shape[1] = {cand_len};
    blt_tensor cand_bytes = blt_tensor_create(arena, bytes_shape, 1, BLT_DTYPE_UINT8);
    blt_tensor_upload(&cand_bytes, x, cand_len);

    blt_model_enc_out enc;
    blt_model_encode(model, &cand_bytes, patches, num_patches, NULL, 0, &enc, arena);
    if (stats != NULL) {
        stats->nfe_encoder_global++;
    }

    size_t logits_shape[2] = {cand_len, vocab_size};
    blt_tensor logits = blt_tensor_create(arena, logits_shape, 2, BLT_DTYPE_FP32);
    // logits-only: loss/bytes unused by verification
    blt_local_decoder_forward_ext(model->decoder, &enc.byte_hidden_out, &enc.global_out, patches, num_patches, NULL,
                                  NULL, 0, NULL, &logits, NULL, arena);
    if (stats != NULL) {
        stats->nfe_decoder++;
    }

    // Verification argmaxes run against a host copy of the logits.
    float *logits_host = (float *)malloc(logits.numel * sizeof(float));
    BLT_REQUIRE(logits_host != NULL, "blt_verify_draft: failed to stage logits");
    blt_tensor_download(&logits, logits_host, logits.numel * sizeof(float));
    const float *rows = logits_host;

    // Algorithm 2 lines 3-9: accept until first mismatch.
    for (size_t p = l; p < cand_len; p++) {
        uint8_t pred = argmax_byte_host_check(rows + (p - 1) * vocab_size, vocab_size);
        if (x[p] != pred) {
            x[p] = pred;
            if (stats != NULL) {
                stats->bytes_accepted += p - l;
            }
            free(logits_host);
            return p + 1;
        }
    }

    if (stats != NULL) {
        stats->bytes_accepted += r;
    }
    if (cand_len < target_len) {
        x[cand_len] = argmax_byte_host_check(rows + (cand_len - 1) * vocab_size, vocab_size);
        free(logits_host);
        return cand_len + 1;
    }
    free(logits_host);
    return cand_len;
}

size_t blt_verify_draft_aligned(const blt_model *model, const blt_entropy_lm *entropy_model,
                                const blt_patcher_config *patcher_config, uint8_t *x, size_t l, size_t r,
                                size_t target_len, blt_infer_stats *stats, blt_arena *arena) {
    BLT_REQUIRE(model != NULL && entropy_model != NULL && patcher_config != NULL && x != NULL && arena != NULL,
                "blt_verify_draft_aligned: arguments cannot be NULL");
    BLT_REQUIRE(l >= 1, "blt_verify_draft_aligned: committed length l must be >= 1");
    const size_t cand_len = l + r;
    BLT_REQUIRE(cand_len >= 2 && cand_len <= model->config.decoder_config.max_seq_len,
                "blt_verify_draft_aligned: candidate length must be in [2, max_seq_len]");

    const size_t vocab_size = model->config.decoder_config.vocab_size;

    // Natural segmentation — no forced split at l.
    blt_tensor entropy_vals;
    compute_entropy_vals(arena, entropy_model, x, cand_len, &entropy_vals);

    blt_tensor entropy_host_view;
    float *entropy_host_buf = NULL;
    const blt_tensor *entropy_for_patcher = stage_entropy_host(&entropy_vals, &entropy_host_view, &entropy_host_buf);

    blt_patch_info patches[BLT_SELFSPEC_MAX_PATCHES];
    const size_t num_patches =
        blt_segment_patches(entropy_for_patcher, x, patches, BLT_SELFSPEC_MAX_PATCHES, patcher_config);
    free(entropy_host_buf);
    entropy_host_buf = NULL;

    size_t bytes_shape[1] = {cand_len};
    blt_tensor cand_bytes = blt_tensor_create(arena, bytes_shape, 1, BLT_DTYPE_UINT8);
    blt_tensor_upload(&cand_bytes, x, cand_len);

    blt_model_enc_out enc;
    blt_model_encode(model, &cand_bytes, patches, num_patches, NULL, 0, &enc, arena);
    if (stats != NULL) {
        stats->nfe_encoder_global++;
    }

    size_t logits_shape[2] = {cand_len, vocab_size};
    blt_tensor logits = blt_tensor_create(arena, logits_shape, 2, BLT_DTYPE_FP32);
    blt_local_decoder_forward_ext(model->decoder, &enc.byte_hidden_out, &enc.global_out, patches, num_patches, NULL,
                                  NULL, 0, NULL, &logits, NULL, arena);
    if (stats != NULL) {
        stats->nfe_decoder++;
    }

    float *logits_host = (float *)malloc(logits.numel * sizeof(float));
    BLT_REQUIRE(logits_host != NULL, "blt_verify_draft_aligned: failed to stage logits");
    blt_tensor_download(&logits, logits_host, logits.numel * sizeof(float));
    const float *rows = logits_host;

    // Accept until mismatch; commit only at natural patch ends.
    for (size_t p = l; p < cand_len; p++) {
        const uint8_t pred = argmax_byte_host_check(rows + (p - 1) * vocab_size, vocab_size);
        if (x[p] != pred) {
            const size_t b = blt_aligned_commit_select(patches, num_patches, l, p);
            if (b > l) {
                if (stats != NULL) {
                    stats->bytes_accepted += b - l;
                }
                free(logits_host);
                return b;
            }
            x[p] = pred;
            if (stats != NULL) {
                stats->bytes_accepted += 1;
            }
            free(logits_host);
            return p + 1;
        }
    }

    if (stats != NULL) {
        stats->bytes_accepted += r;
    }
    if (cand_len < target_len) {
        x[cand_len] = argmax_byte_host_check(rows + (cand_len - 1) * vocab_size, vocab_size);
        free(logits_host);
        return cand_len + 1;
    }
    free(logits_host);
    return cand_len;
}

typedef struct {
    blt_kv_cache *cache;
    const blt_model *model;
} draft_ctx;

// Process rows [from_row..l) through incremental decoder; returns argmax
// of row l-1 (greedy prediction for position l).
static uint8_t prefill_argmax(draft_ctx *d, const blt_model_enc_out *enc, const blt_patch_info *patches,
                              size_t num_patches, size_t from_row, size_t l, blt_infer_stats *stats, blt_arena *arena) {
    const size_t E = d->model->config.encoder_config.embed_dim;
    const size_t V = d->model->config.decoder_config.vocab_size;
    const size_t n = l - from_row;
    BLT_REQUIRE(n >= 1, "prefill_argmax: empty chunk");

    size_t d0_shape[2] = {n, E};
    blt_tensor d0 = blt_tensor_create(arena, d0_shape, 2, BLT_DTYPE_FP32);
    blt_strided_copy(enc->byte_hidden_out.backend, (float *)d0.data, E,
                     (const float *)enc->byte_hidden_out.data + from_row * E, E, n, E);

    size_t logits_shape[2] = {n, V};
    blt_tensor logits = blt_tensor_create(arena, logits_shape, 2, BLT_DTYPE_FP32);
    blt_kv_decode_step(d->cache, &enc->global_out, patches, num_patches, &d0, &logits, arena);
    if (stats != NULL) {
        stats->nfe_decoder++;
    }

    blt_tensor last_row;
    blt_tensor_view_2d(&last_row, (float *)logits.data + (n - 1) * V, 1, V, logits.backend);
    uint32_t ids[1];
    blt_argmax_rows(&last_row, ids);
    return argmax_byte_host(ids);
}

// Single-row draft step. D_0 mode: ZEROS leaves it zero, LEARNED reads
// d0_embed_weight[token]. Logits predict position pos+1.
static uint8_t draft_step(draft_ctx *d, const blt_model_enc_out *enc, const blt_patch_info *patches, size_t num_patches,
                          const uint8_t *draft, size_t m, size_t pos, blt_d0_mode mode, blt_infer_stats *stats,
                          blt_arena *arena) {
    const size_t E = d->model->config.encoder_config.embed_dim;
    const size_t V = d->model->config.decoder_config.vocab_size;
    BLT_REQUIRE(pos == d->cache->self_len, "draft_step: cache length must equal the draft position");
    BLT_REQUIRE(m >= 1, "draft_step: no previously drafted byte");

    size_t d0_shape[2] = {1, E};
    blt_tensor d0 = blt_tensor_create(arena, d0_shape, 2, BLT_DTYPE_FP32); // zero-filled

    if (mode == BLT_D0_LEARNED) {
        const uint32_t token = (uint32_t)draft[m - 1];
        const blt_tensor *table = &d->model->decoder->d0_embed_weight;
        blt_strided_copy(table->backend, (float *)d0.data, E, (const float *)table->data + token * E, E, 1, E);
    }

    size_t logits_shape[2] = {1, V};
    blt_tensor logits = blt_tensor_create(arena, logits_shape, 2, BLT_DTYPE_FP32);
    blt_kv_decode_step(d->cache, &enc->global_out, patches, num_patches, &d0, &logits, arena);
    if (stats != NULL) {
        stats->nfe_decoder++;
    }

    uint32_t ids[1];
    blt_argmax_rows(&logits, ids);
    return argmax_byte_host(ids);
}

void blt_generate_greedy_selfspec(const blt_model *model, const blt_entropy_lm *entropy_model,
                                  const blt_patcher_config *patcher_config, const uint8_t *prompt_bytes,
                                  size_t prompt_len, size_t max_new_bytes, uint8_t *output_bytes,
                                  const blt_self_spec_config *config, blt_infer_stats *stats, blt_arena *arena) {
    BLT_REQUIRE(model != NULL && entropy_model != NULL && patcher_config != NULL && prompt_bytes != NULL &&
                    output_bytes != NULL && config != NULL && arena != NULL,
                "blt_generate_greedy_selfspec: arguments cannot be NULL");
    BLT_REQUIRE(prompt_len >= 2, "blt_generate_greedy_selfspec: prompt_len must be >= 2");
    BLT_REQUIRE(config->window_k >= 1, "blt_generate_greedy_selfspec: window_k must be >= 1");
    BLT_REQUIRE(config->window_k <= BLT_SELFSPEC_MAX_PATCHES,
                "blt_generate_greedy_selfspec: window_k must be <= BLT_SELFSPEC_MAX_PATCHES");

    memcpy(output_bytes, prompt_bytes, prompt_len);
    size_t l = prompt_len;
    const size_t target_len = prompt_len + max_new_bytes;

    // Cache buffers allocated before per-round arena markers so resets
    // never invalidate them.
    blt_kv_cache *cache = blt_kv_cache_create(arena, model->decoder, model->config.decoder_config.max_seq_len);

    draft_ctx d = {.cache = cache, .model = model};

    while (l < target_len) {
        const size_t round_marker = arena->offset;

        // Cap the window: l + k + 1 <= target_len.
        size_t r_eff = config->window_k;
        if (l + r_eff + 1 > target_len) {
            r_eff = target_len - l - 1;
        }

        // 1. Segment the committed prefix
        blt_tensor entropy_vals;
        compute_entropy_vals(arena, entropy_model, output_bytes, l, &entropy_vals);

        blt_tensor entropy_host_view;
        float *entropy_host_buf = NULL;
        const blt_tensor *entropy_for_patcher =
            stage_entropy_host(&entropy_vals, &entropy_host_view, &entropy_host_buf);

        blt_patch_info patches[BLT_SELFSPEC_MAX_PATCHES];
        size_t num_patches =
            blt_segment_patches(entropy_for_patcher, output_bytes, patches, BLT_SELFSPEC_MAX_PATCHES, patcher_config);
        free(entropy_host_buf);
        entropy_host_buf = NULL;

        // 2. Freeze latents (one encoder+global pass)
        size_t bytes_shape[1] = {l};
        blt_tensor prefix_bytes = blt_tensor_create(arena, bytes_shape, 1, BLT_DTYPE_UINT8);
        blt_tensor_upload(&prefix_bytes, output_bytes, l);

        blt_model_enc_out enc;
        blt_model_encode(model, &prefix_bytes, patches, num_patches, NULL, 0, &enc, arena);
        if (stats != NULL) {
            stats->nfe_encoder_global++;
        }

        // 3. Cache tier maintenance: LCP with cache, refresh trailing K/V.
        size_t common = blt_kv_cache_common_patches(cache, patches, num_patches);
        size_t self_valid = 0;
        if (common > 0) {
            self_valid = patches[common - 1].start_idx + patches[common - 1].length;
        }
        blt_kv_cache_truncate(cache, self_valid, common);
        blt_kv_cache_refresh_cross(cache, &enc.global_out, patches, num_patches, common, arena);

        // 4. Prefill chunk -> first draft byte
        uint8_t next = prefill_argmax(&d, &enc, patches, num_patches, self_valid, l, stats, arena);

        if (r_eff == 0) {
            // One byte left — plain AR step, no speculation.
            output_bytes[l] = next;
            l++;
            arena->offset = round_marker;
            break;
        }

        // 5. Draft past the boundary.
        uint8_t draft[BLT_SELFSPEC_MAX_PATCHES];
        draft[0] = next;
        if (stats != NULL) {
            stats->bytes_drafted++;
        }
        for (size_t m = 1; m < r_eff; m++) {
            draft[m] = draft_step(&d, &enc, patches, num_patches, draft, m, l + m - 1, config->d0_mode, stats, arena);
            if (stats != NULL) {
                stats->bytes_drafted++;
            }
        }
        memcpy(output_bytes + l, draft, r_eff);

        // 6. Verify (Algorithm 2): re-segment, full forward, accept/replace
        l = blt_verify_draft(model, entropy_model, patcher_config, output_bytes, l, r_eff, target_len, stats, arena);

        arena->offset = round_marker;
    }
}
