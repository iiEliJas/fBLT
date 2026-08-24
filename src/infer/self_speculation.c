// BLT-S: self-speculative greedy generation (Fast-BLT 5.1, Algorithm 2).
//
// Round structure:
//   1. entropy LM + patcher segment the committed prefix
//   2. one encoder+global call freezes the latents          [1 enc/global NFE]
//      (tier-3 global/encoder caching is deferred; see kv_cache.h)
//   3. tier maintenance: LCP vs the cached patch boundaries ->
//      truncate both tiers to the last completed patch, refresh the
//      trailing patch's cross-attn K/V
//   4. prefill chunk through the incremental decoder for the open patch's
//      rows -> argmax of row l-1 drafts the first byte     [1 decoder NFE]
//   5. single-row decoder steps draft up to window_k bytes [1 NFE each]
//   6. Algorithm 2 verification (dense full forward): accept until first
//      mismatch, replace it                                [1 enc/global NFE,
//                                                            1 decoder NFE]
//
// Completed patches are immutable across rounds (prefix-stable patcher +
// causal models), so tier entries below the truncation point stay valid and
// per-round decoder work shrinks to the open patch + drafts.
#include "blt/infer/self_speculation.h"

#include "blt/infer/kv_cache.h"
#include "blt/core/backend.h"
#include "blt/models/entropy.h"
#include "blt/models/local_common.h"
#include "blt/ops/softmax.h"

#include <stdint.h>
#include <string.h>


#define BLT_SELFSPEC_MAX_PATCHES 4096


static uint8_t argmax_byte(const float* row, size_t vocab_size) {
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


// Runs the entropy model over bytes[0..len) and writes per-byte entropies
// into *entropy_vals_out ([len] FP32). Same numerics as the plain greedy
// loop in tools/generate_greedy.c.
static void compute_entropy_vals(blt_arena* arena, const blt_entropy_lm* entropy_model,
                                 const uint8_t* bytes, size_t len,
                                 blt_tensor* entropy_vals_out) {
    size_t bytes_shape[1] = {len};
    blt_tensor bytes_in = blt_tensor_create(arena, bytes_shape, 1, BLT_DTYPE_UINT8);
    memcpy(bytes_in.data, bytes, len);

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


size_t blt_verify_draft(
    const blt_model* model,
    const blt_entropy_lm* entropy_model,
    const blt_patcher_config* patcher_config,
    uint8_t* x,
    size_t l,
    size_t r,
    size_t target_len,
    blt_infer_stats* stats,
    blt_arena* arena
) {
    BLT_REQUIRE(model != NULL && entropy_model != NULL && patcher_config != NULL &&
                x != NULL && arena != NULL,
        "blt_verify_draft: arguments cannot be NULL");
    BLT_REQUIRE(l >= 1, "blt_verify_draft: committed length l must be >= 1");
    const size_t cand_len = l + r;
    BLT_REQUIRE(cand_len >= 2 && cand_len <= model->config.decoder_config.max_seq_len,
        "blt_verify_draft: candidate length must be in [2, max_seq_len]");

    const size_t vocab_size = model->config.decoder_config.vocab_size;

    // -------------------------------------------------------------
    // Algorithm 2 line 1: segment the candidate x' into M' patches
    // -------------------------------------------------------------
    blt_tensor entropy_vals;
    compute_entropy_vals(arena, entropy_model, x, cand_len, &entropy_vals);

    blt_patch_info patches[BLT_SELFSPEC_MAX_PATCHES];
    size_t num_patches = blt_segment_patches(
        &entropy_vals, x, patches, BLT_SELFSPEC_MAX_PATCHES, patcher_config);

    // -------------------------------------------------------------
    // Algorithm 2 line 2: T' = E(x'); O' = G(T'); y = D(x'; O')
    // -------------------------------------------------------------
    size_t bytes_shape[1] = {cand_len};
    blt_tensor cand_bytes = blt_tensor_create(arena, bytes_shape, 1, BLT_DTYPE_UINT8);
    memcpy(cand_bytes.data, x, cand_len);

    blt_model_enc_out enc;
    blt_model_encode(model, &cand_bytes, patches, num_patches, NULL, 0, &enc, arena);
    if (stats != NULL) {
        stats->nfe_encoder_global++;
    }

    size_t logits_shape[2] = {cand_len, vocab_size};
    blt_tensor logits = blt_tensor_create(arena, logits_shape, 2, BLT_DTYPE_FP32);
    // logits-only decode: loss/bytes unused by verification
    blt_local_decoder_forward_ext(model->decoder, &enc.byte_hidden_out, &enc.global_out,
        patches, num_patches, NULL, NULL, 0, NULL, &logits, NULL, arena);
    if (stats != NULL) {
        stats->nfe_decoder++;
    }

    const float* rows = (const float*)logits.data;

    // -------------------------------------------------------------
    // Algorithm 2 lines 3-9: accept until first mismatch, else free byte.
    //
    // Prediction for position p is argmax(logits row p-1). Drafted byte at
    // position p is x[p]. On mismatch, position p is REPLACED with the
    // prediction and committed length becomes p+1. On full match, the free
    // byte y_{l+r-1} extends the sequence by one more (budget permitting).
    // -------------------------------------------------------------
    for (size_t p = l; p < cand_len; p++) {
        uint8_t pred = argmax_byte(rows + (p - 1) * vocab_size, vocab_size);
        if (x[p] != pred) {
            x[p] = pred;                     // reject drafted byte; replace first mismatch
            if (stats != NULL) {
                stats->bytes_accepted += p - l;
            }
            return p + 1;
        }
    }

    // full match: all r drafted bytes accepted
    if (stats != NULL) {
        stats->bytes_accepted += r;
    }
    if (cand_len < target_len) {
        x[cand_len] = argmax_byte(rows + (cand_len - 1) * vocab_size, vocab_size);   // free byte
        return cand_len + 1;
    }
    return cand_len;
}


//----------------------------------------------------------------------
// Cache-aware drafting helpers

typedef struct {
    blt_kv_cache* cache;
    const blt_model* model;
} draft_ctx;


// Processes decoder rows [from_row .. l) through the incremental decoder
// (prefill chunk), returning argmax of row l-1 -- i.e. the model's greedy
// prediction for position l given the committed prefix.
//
// D_0 rows come from the encoder h_final states (prefill rows are always
// real committed bytes).
static uint8_t prefill_argmax(draft_ctx* d, const blt_model_enc_out* enc,
                              const blt_patch_info* patches, size_t num_patches,
                              size_t from_row, size_t l, blt_infer_stats* stats,
                              blt_arena* arena) {
    const size_t E = d->model->config.encoder_config.embed_dim;
    const size_t V = d->model->config.decoder_config.vocab_size;
    const size_t n = l - from_row;
    BLT_REQUIRE(n >= 1, "prefill_argmax: empty chunk");

    size_t d0_shape[2] = {n, E};
    blt_tensor d0 = blt_tensor_create(arena, d0_shape, 2, BLT_DTYPE_FP32);
    memcpy(d0.data, (float*)enc->byte_hidden_out.data + from_row * E, n * E * sizeof(float));

    size_t logits_shape[2] = {n, V};
    blt_tensor logits = blt_tensor_create(arena, logits_shape, 2, BLT_DTYPE_FP32);
    blt_kv_decode_step(d->cache, &enc->global_out, patches, num_patches, &d0, &logits, arena);
    if (stats != NULL) {
        stats->nfe_decoder++;
    }

    return argmax_byte((const float*)logits.data + (n - 1) * V, V);
}

// One single-row drafting step: processes the row at absolute position
// `pos` (= cache->self_len) whose D_0 follows `mode` -- ZEROS leaves it
// zero, LEARNED reads d0_embed_weight[token = previously drafted byte].
// Its logits predict position pos+1, i.e. the next draft byte.
static uint8_t draft_step(draft_ctx* d, const blt_model_enc_out* enc,
                          const blt_patch_info* patches, size_t num_patches,
                          const uint8_t* draft, size_t m, size_t pos,
                          blt_d0_mode mode, blt_infer_stats* stats,
                          blt_arena* arena) {
    const size_t E = d->model->config.encoder_config.embed_dim;
    const size_t V = d->model->config.decoder_config.vocab_size;
    BLT_REQUIRE(pos == d->cache->self_len,
        "draft_step: cache length must equal the draft position");
    BLT_REQUIRE(m >= 1, "draft_step: no previously drafted byte");

    size_t d0_shape[2] = {1, E};
    blt_tensor d0 = blt_tensor_create(arena, d0_shape, 2, BLT_DTYPE_FP32);   // zero-filled

    if (mode == BLT_D0_LEARNED) {
        const uint32_t token = (uint32_t)draft[m - 1];
        memcpy(d0.data,
            (float*)d->model->decoder->d0_embed_weight.data + token * E,
            E * sizeof(float));
    }

    size_t logits_shape[2] = {1, V};
    blt_tensor logits = blt_tensor_create(arena, logits_shape, 2, BLT_DTYPE_FP32);
    blt_kv_decode_step(d->cache, &enc->global_out, patches, num_patches, &d0, &logits, arena);
    if (stats != NULL) {
        stats->nfe_decoder++;
    }

    return argmax_byte((const float*)logits.data, V);
}


void blt_generate_greedy_selfspec(
    const blt_model* model,
    const blt_entropy_lm* entropy_model,
    const blt_patcher_config* patcher_config,
    const uint8_t* prompt_bytes,
    size_t prompt_len,
    size_t max_new_bytes,
    uint8_t* output_bytes,
    const blt_self_spec_config* config,
    blt_infer_stats* stats,
    blt_arena* arena
) {
    BLT_REQUIRE(model != NULL && entropy_model != NULL && patcher_config != NULL &&
                prompt_bytes != NULL && output_bytes != NULL && config != NULL && arena != NULL,
        "blt_generate_greedy_selfspec: arguments cannot be NULL");
    BLT_REQUIRE(prompt_len >= 2, "blt_generate_greedy_selfspec: prompt_len must be >= 2");
    BLT_REQUIRE(config->window_k >= 1, "blt_generate_greedy_selfspec: window_k must be >= 1");
    BLT_REQUIRE(config->window_k <= BLT_SELFSPEC_MAX_PATCHES,
        "blt_generate_greedy_selfspec: window_k must be <= BLT_SELFSPEC_MAX_PATCHES");

    memcpy(output_bytes, prompt_bytes, prompt_len);
    size_t l = prompt_len;
    const size_t target_len = prompt_len + max_new_bytes;

    // Cache buffers are allocated once, BEFORE the per-round arena markers,
    // so the round resets below never invalidate them.
    blt_kv_cache* cache = blt_kv_cache_create(arena, model->decoder,
        model->config.decoder_config.max_seq_len);

    draft_ctx d = { .cache = cache, .model = model };

    while (l < target_len) {
        const size_t round_marker = arena->offset;

        // Cap the window so a fully accepted draft plus its free byte can
        // never exceed the output budget: l + k + 1 <= target_len.
        size_t r_eff = config->window_k;
        if (l + r_eff + 1 > target_len) {
            r_eff = target_len - l - 1;
        }

        // -------------------------------------------------------------
        // 1. Segment the committed prefix
        // -------------------------------------------------------------
        blt_tensor entropy_vals;
        compute_entropy_vals(arena, entropy_model, output_bytes, l, &entropy_vals);

        blt_patch_info patches[BLT_SELFSPEC_MAX_PATCHES];
        size_t num_patches = blt_segment_patches(
            &entropy_vals, output_bytes, patches, BLT_SELFSPEC_MAX_PATCHES, patcher_config);

        // -------------------------------------------------------------
        // 2. Freeze the latents (one encoder+global pass)
        // -------------------------------------------------------------
        size_t bytes_shape[1] = {l};
        blt_tensor prefix_bytes = blt_tensor_create(arena, bytes_shape, 1, BLT_DTYPE_UINT8);
        memcpy(prefix_bytes.data, output_bytes, l);

        blt_model_enc_out enc;
        blt_model_encode(model, &prefix_bytes, patches, num_patches, NULL, 0, &enc, arena);
        if (stats != NULL) {
            stats->nfe_encoder_global++;
        }

        // -------------------------------------------------------------
        // 3. Cache tier maintenance: roll back to the longest common patch
        //    prefix with what is cached, then refresh the trailing
        //    patches' cross-attn K/V against the frozen latents.
        //
        //    common < num_patches always holds (a tiling of [0,l) cannot
        //    equal a shorter cached tiling), so at least one row remains
        //    for the prefill chunk.
        // -------------------------------------------------------------
        size_t common = blt_kv_cache_common_patches(cache, patches, num_patches);
        size_t self_valid = 0;
        if (common > 0) {
            self_valid = patches[common - 1].start_idx + patches[common - 1].length;
        }
        blt_kv_cache_truncate(cache, self_valid, common);
        blt_kv_cache_refresh_cross(cache, &enc.global_out, patches, num_patches, common, arena);

        // -------------------------------------------------------------
        // 4. Prefill chunk -> first draft byte (argmax of row l-1)
        // -------------------------------------------------------------
        uint8_t next = prefill_argmax(&d, &enc, patches, num_patches,
            self_valid, l, stats, arena);

        if (r_eff == 0) {
            // Only one byte left in the budget: the prefill argmax IS the
            // final byte (plain AR step, no speculation).
            output_bytes[l] = next;
            l++;
            arena->offset = round_marker;
            break;
        }

        // -------------------------------------------------------------
        // 5. Draft: single-row decoder steps past the boundary. Byte
        //    draft[m] is the prediction for position l+m and comes from
        //    processing row l+m-1, whose D_0 encodes draft[m-1].
        // -------------------------------------------------------------
        uint8_t draft[BLT_SELFSPEC_MAX_PATCHES];
        draft[0] = next;
        if (stats != NULL) {
            stats->bytes_drafted++;
        }
        for (size_t m = 1; m < r_eff; m++) {
            draft[m] = draft_step(&d, &enc, patches, num_patches,
                draft, m, l + m - 1, config->d0_mode, stats, arena);
            if (stats != NULL) {
                stats->bytes_drafted++;
            }
        }
        memcpy(output_bytes + l, draft, r_eff);

        // -------------------------------------------------------------
        // 6. Verify (Algorithm 2): re-segment, full forward, accept/replace
        // -------------------------------------------------------------
        l = blt_verify_draft(model, entropy_model, patcher_config,
            output_bytes, l, r_eff, target_len, stats, arena);

        arena->offset = round_marker;
    }
}
