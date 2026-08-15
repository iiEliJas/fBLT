#include "blt/ops/mask_builder.h"
#include "blt/core/allocator.h"
#include "blt/core/backend.h"

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdalign.h>


// Precompute doc IDs for all pairs in one call
// Avoids rescanning doc_boundaries for every pair of positions
static void precompute_doc_ids(const size_t* doc_boundaries, size_t num_docs,
                                size_t seq_len, size_t* doc_id_out) {
    size_t doc_id = 0;
    for (size_t pos = 0; pos < seq_len; pos++) {
        while (doc_id + 1 < num_docs && pos >= doc_boundaries[doc_id + 1]) {
            doc_id++;
        }
        doc_id_out[pos] = doc_id;
    }
}



void blt_build_attention_mask(const blt_mask_config* config, blt_tensor* out_mask, blt_arena* arena) {
    BLT_REQUIRE(config != NULL, "blt_build_attention_mask: config is NULL");
    BLT_REQUIRE(out_mask != NULL, "blt_build_attention_mask: out_mask is NULL");
    BLT_REQUIRE(arena != NULL, "blt_build_attention_mask: arena is NULL");

    const size_t seq_len_q = config->seq_len_q;
    const size_t seq_len_kv = config->seq_len_kv;
    BLT_REQUIRE(seq_len_q > 0 && seq_len_kv > 0, "blt_build_attention_mask: seq_len_q/seq_len_kv must be > 0");

    const bool has_docs = (config->doc_boundaries != NULL && config->num_docs > 0);
    const bool has_groups = (config->query_group_ids != NULL && config->kv_group_ids != NULL);

    // Allocate mask tensor [seq_len_q, seq_len_kv] in FP32
    size_t shape[2] = {seq_len_q, seq_len_kv};
    *out_mask = blt_tensor_create(arena, shape, 2, BLT_DTYPE_FP32);
    float* mask_data = (float*)out_mask->data;
    const float neg_inf = -INFINITY;

    // precompute per position doc IDs once instead of rescanning inside loop
    size_t* doc_id_q = NULL;
    size_t* doc_id_kv = NULL;
    if (has_docs) {
        doc_id_q = (size_t*)blt_arena_alloc(arena, seq_len_q * sizeof(size_t), alignof(size_t));
        doc_id_kv = (size_t*)blt_arena_alloc(arena, seq_len_kv * sizeof(size_t), alignof(size_t));
        precompute_doc_ids(config->doc_boundaries, config->num_docs, seq_len_q, doc_id_q);
        precompute_doc_ids(config->doc_boundaries, config->num_docs, seq_len_kv, doc_id_kv);
    }

    for (size_t i = 0; i < seq_len_q; i++) {
        bool row_has_valid_key = false;
        for (size_t j = 0; j < seq_len_kv; j++) {
            bool allowed = true;

            // 1. Causal check
            if (config->is_causal && j > i) {
                allowed = false;
            }

            // 2. Sliding window check
            if (config->sliding_window > 0) {
                ptrdiff_t diff = (ptrdiff_t)i - (ptrdiff_t)j;
                if (diff < 0) diff = -diff;
                if ((size_t)diff >= config->sliding_window) {
                    allowed = false;
                }
            }

            // 3. Document boundary check: only attend within the same document
            if (has_docs && doc_id_q[i] != doc_id_kv[j]) {
                allowed = false;
            }

            // 4. Group check (block-diagonal).
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
        // Guard against fully-masked rows
        BLT_REQUIRE(row_has_valid_key, "blt_build_attention_mask: query position has no valid key");
    }
}