#include "ops/mask_builder.h"
#include "core/allocator.h"
#include "core/backend.h"

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdalign.h>

// Avoids rescanning doc_boundaries for every pair of positions.
static void precompute_doc_ids(const size_t *doc_boundaries, size_t num_docs, size_t seq_len, size_t *doc_id_out) {
    size_t doc_id = 0;
    for (size_t pos = 0; pos < seq_len; pos++) {
        while (doc_id + 1 < num_docs && pos >= doc_boundaries[doc_id]) {
            doc_id++;
        }
        doc_id_out[pos] = doc_id;
    }
}

void blt_build_attention_mask_cpu(const blt_mask_config *config, blt_tensor *out_mask, blt_arena *arena) {
    BLT_REQUIRE(arena->backend == BLT_BACKEND_CPU,
                "blt_build_attention_mask: CPU implementation called with non-CPU arena");
    BLT_REQUIRE(config != NULL, "blt_build_attention_mask: config is NULL");
    BLT_REQUIRE(out_mask != NULL, "blt_build_attention_mask: out_mask is NULL");
    BLT_REQUIRE(arena != NULL, "blt_build_attention_mask: arena is NULL");

    const size_t seq_len_q = config->seq_len_q;
    const size_t seq_len_kv = config->seq_len_kv;
    BLT_REQUIRE(seq_len_q > 0 && seq_len_kv > 0, "blt_build_attention_mask: seq_len_q/seq_len_kv must be > 0");

    const bool has_docs = (config->doc_boundaries != NULL && config->num_docs > 0);
    const bool has_groups = (config->query_group_ids != NULL && config->kv_group_ids != NULL);

    size_t shape[2] = {seq_len_q, seq_len_kv};
    *out_mask = blt_tensor_create(arena, shape, 2, BLT_DTYPE_FP32);
    float *mask_data = (float *)out_mask->data;
    const float neg_inf = -INFINITY;

    size_t *doc_id_q = NULL;
    size_t *doc_id_kv = NULL;
    if (has_docs) {
        doc_id_q = (size_t *)blt_arena_alloc(arena, seq_len_q * sizeof(size_t), alignof(size_t));
        doc_id_kv = (size_t *)blt_arena_alloc(arena, seq_len_kv * sizeof(size_t), alignof(size_t));
        precompute_doc_ids(config->doc_boundaries, config->num_docs, seq_len_q, doc_id_q);
        precompute_doc_ids(config->doc_boundaries, config->num_docs, seq_len_kv, doc_id_kv);
    }

    for (size_t i = 0; i < seq_len_q; i++) {
        bool row_has_valid_key = false;
        for (size_t j = 0; j < seq_len_kv; j++) {
            bool allowed = true;

            // 1. Causal check (with optional offset for cached prefixes)
            if (config->is_causal && j > i + config->causal_offset) {
                allowed = false;
            }

            // 2. Sliding window check. With a causal offset (cached prefix)
            // the distance is measured from the offset diagonal, matching
            // absolute positions in incremental decode.
            if (config->sliding_window > 0) {
                if (config->is_causal) {
                    const size_t diag = i + config->causal_offset;
                    size_t back = diag >= j ? diag - j : j - diag;
                    if (back >= config->sliding_window) {
                        allowed = false;
                    }
                } else {
                    ptrdiff_t diff = (ptrdiff_t)i - (ptrdiff_t)j;
                    if (diff < 0) diff = -diff;
                    if ((size_t)diff >= config->sliding_window) {
                        allowed = false;
                    }
                }
            }

            if (has_docs && doc_id_q[i] != doc_id_kv[j]) {
                allowed = false;
            }

            if (has_groups) {
                if (config->query_group_ids[i] != config->kv_group_ids[j]) {
                    allowed = false;
                } else if (!config->bidirectional_within_group && j > i) {
                    allowed = false;
                }
            }

            mask_data[i * seq_len_kv + j] = allowed ? 0.0f : neg_inf;
            row_has_valid_key = row_has_valid_key || allowed;
        }
        BLT_REQUIRE(row_has_valid_key, "blt_build_attention_mask: query position has no valid key");
    }
}

void blt_build_block_diffusion_mask_cpu(const blt_block_diffusion_config *config, blt_tensor *out_mask,
                                        blt_arena *arena) {
    BLT_REQUIRE(arena->backend == BLT_BACKEND_CPU,
                "blt_build_block_diffusion_mask: CPU implementation called with non-CPU arena");
    BLT_REQUIRE(config != NULL && out_mask != NULL && arena != NULL,
                "blt_build_block_diffusion_mask: config, out_mask and arena cannot be NULL");
    const size_t S = config->seq_len;
    const size_t N = config->num_clean;
    BLT_REQUIRE(S >= 1 && N >= 1 && N <= S, "blt_build_block_diffusion_mask: need 1 <= num_clean <= seq_len");

    if (config->mode == BLT_BDM_TRAIN) {
        BLT_REQUIRE(config->block_size >= 1, "blt_build_block_diffusion_mask: TRAIN mode needs block_size >= 1");
        BLT_REQUIRE((S - N) % config->block_size == 0,
                    "blt_build_block_diffusion_mask: block section must tile exactly (B * (M-1))");
    }

    size_t shape[2] = {S, S};
    *out_mask = blt_tensor_create(arena, shape, 2, BLT_DTYPE_FP32);
    float *m = (float *)out_mask->data;
    const float neg_inf = -INFINITY;

    for (size_t i = 0; i < S; i++) {
        bool row_ok = false;

        for (size_t j = 0; j < S; j++) {
            bool allowed;
            if (i < N) {
                // Clean prefix: causal among clean rows only; never see blocks.
                allowed = (j < N) && (j <= i);
            } else if (config->mode == BLT_BDM_INFER) {
                // Single live block: all clean + whole block section.
                allowed = true;
            } else {
                // TRAIN: prose rule -- clean rows causal; block row i sees all
                // clean bytes plus every block with block-index <= i's
                // block-index (bidirectional within own block). Paper prose
                // (3.2.2) vs Figure 5 matrix ambiguity: prose rule adopted.
                const size_t B = config->block_size;
                allowed = (j < N) || ((j - N) / B <= (i - N) / B);
            }

            m[i * S + j] = allowed ? 0.0f : neg_inf;
            row_ok = row_ok || allowed;
        }
        BLT_REQUIRE(row_ok, "blt_build_block_diffusion_mask: query position has no valid key");
    }
}
