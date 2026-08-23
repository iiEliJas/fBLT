// blt/infer/self_speculation.c
//
// BLT-S: self-speculative greedy generation (Fast-BLT §5.1, Algorithm 2).
//
// Round structure (dense v1 -- no KV cache yet, Phase C adds it):
//   1. entropy LM + patcher segment the committed prefix
//   2. one encoder+global call freezes the latents          [1 enc/global NFE]
//   3. up to window_k decoder-only passes draft bytes past the last
//      patch boundary, conditioning on the last latent token
//      (draft rows join the last patch's cross-attention group --
//      built into blt_local_decoder_forward_ext)            [k decoder NFEs]
//   4. Algorithm 2 verification: re-segment the candidate, full forward,
//      accept until first mismatch, replace it              [1 enc/global NFE,
//                                                            1 decoder NFE]
#include "blt/infer/self_speculation.h"

#include "blt/core/backend.h"
#include "blt/models/entropy.h"
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
    blt_model_decode(model, &enc, patches, num_patches, NULL, NULL, 0,
        NULL, &logits, NULL, arena);
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


// One drafting step: decodes over committed prefix x[0..l) plus m drafted
// bytes d[0..m) against the frozen prefix latents, returns the argmax of
// the last logits row (the next draft byte).
//
// Draft rows have no encoder h_final; their D_0 comes from config->d0_mode.
// They condition on the LAST patch's latent sub-tokens via the group-id
// rule inside make_context -- the Fast-BLT "condition on last available
// latent token" mechanism.
static uint8_t draft_next_byte(
    const blt_model* model,
    const blt_model_enc_out* enc,
    const blt_patch_info* patches,
    size_t num_patches,
    const uint8_t* draft,
    size_t m,
    blt_d0_mode d0_mode,
    blt_infer_stats* stats,
    blt_arena* arena
) {
    const size_t l = enc->byte_hidden_out.shape[0];
    const size_t E = model->config.encoder_config.embed_dim;
    const size_t seq_len = l + m;
    const size_t vocab_size = model->config.decoder_config.vocab_size;

    // Assemble decoder input: h_final for the prefix rows, policy-filled
    // rows for the draft positions.
    size_t hidden_shape[2] = {seq_len, E};
    blt_tensor hidden_in = blt_tensor_create(arena, hidden_shape, 2, BLT_DTYPE_FP32);
    memcpy(hidden_in.data, enc->byte_hidden_out.data, l * E * sizeof(float));

    blt_local_decoder_d0_opts opts = {0};
    opts.d0_mode = d0_mode;
    opts.num_hfinal_rows = l;
    uint32_t tokens[BLT_SELFSPEC_MAX_PATCHES];
    BLT_REQUIRE(seq_len <= model->config.decoder_config.max_seq_len,
        "blt-s draft: seq_len exceeds decoder max_seq_len");
    if (m > 0) {
        // tensor_create zero-fills: ZEROS mode needs no extra work
        if (d0_mode == BLT_D0_LEARNED) {
            for (size_t i = 0; i < m; i++) {
                tokens[i] = (uint32_t)draft[i];
            }
            opts.d0_extra_tokens = tokens;
        }
    } else {
        opts.d0_mode = BLT_D0_HFINAL;   // no extra rows: legacy semantics
    }

    size_t logits_shape[2] = {seq_len, vocab_size};
    blt_tensor logits = blt_tensor_create(arena, logits_shape, 2, BLT_DTYPE_FP32);
    // Decode directly against the frozen latents: assembled D_0 rows for the
    // byte positions, frozen global_out O for cross-attention. (The
    // blt_model_decode wrapper always uses enc->byte_hidden_out verbatim and
    // cannot express appended draft rows.)
    blt_local_decoder_forward_ext(model->decoder, &hidden_in, &enc->global_out,
        patches, num_patches, NULL, NULL, 0, &opts, &logits, NULL, arena);
    if (stats != NULL) {
        stats->nfe_decoder++;
    }

    return argmax_byte((const float*)logits.data + (seq_len - 1) * vocab_size, vocab_size);
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

    while (l < target_len) {
        const size_t round_marker = arena->offset;

        // Cap the window so a fully accepted draft plus its free byte can
        // never exceed the output budget: l + k + 1 <= target_len.
        size_t r_eff = config->window_k;
        if (l + r_eff + 1 > target_len) {
            r_eff = target_len - l - 1;
        }

        // -------------------------------------------------------------
        // 1+2. Segment the committed prefix and freeze its latents
        // -------------------------------------------------------------
        blt_tensor entropy_vals;
        compute_entropy_vals(arena, entropy_model, output_bytes, l, &entropy_vals);

        blt_patch_info patches[BLT_SELFSPEC_MAX_PATCHES];
        size_t num_patches = blt_segment_patches(
            &entropy_vals, output_bytes, patches, BLT_SELFSPEC_MAX_PATCHES, patcher_config);

        size_t bytes_shape[1] = {l};
        blt_tensor prefix_bytes = blt_tensor_create(arena, bytes_shape, 1, BLT_DTYPE_UINT8);
        memcpy(prefix_bytes.data, output_bytes, l);

        blt_model_enc_out enc;
        blt_model_encode(model, &prefix_bytes, patches, num_patches, NULL, 0, &enc, arena);
        if (stats != NULL) {
            stats->nfe_encoder_global++;
        }

        if (r_eff == 0) {
            // Only one byte left in the budget: skip speculation entirely.
            // The m=0 draft decode below IS the plain AR step (full pipeline
            // over the prefix), so its last-row argmax is final.
            uint8_t next = draft_next_byte(model, &enc, patches, num_patches,
                NULL, 0, config->d0_mode, stats, arena);
            output_bytes[l] = next;
            l++;
            arena->offset = round_marker;
            break;
        }

        // -------------------------------------------------------------
        // 3. Draft: autoregressive decoder-only passes past the boundary
        // -------------------------------------------------------------
        uint8_t draft[BLT_SELFSPEC_MAX_PATCHES];
        for (size_t m = 0; m < r_eff; m++) {
            draft[m] = draft_next_byte(model, &enc, patches, num_patches,
                draft, m, config->d0_mode, stats, arena);
            if (stats != NULL) {
                stats->bytes_drafted++;
            }
        }
        memcpy(output_bytes + l, draft, r_eff);

        // -------------------------------------------------------------
        // 4. Verify (Algorithm 2): re-segment, full forward, accept/replace
        // -------------------------------------------------------------
        l = blt_verify_draft(model, entropy_model, patcher_config,
            output_bytes, l, r_eff, target_len, stats, arena);

        arena->offset = round_marker;
    }
}
