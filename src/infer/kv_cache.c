// Two-tier KV cache + cache-aware incremental decoder
// See kv_cache.h for the design contract. Every computation here mirrors
// the dense composition in src/models/local_decoder.c / attention.c /
// cross_attention.c op-for-op so results are bit-identical:
//   - matmul/rmsnorm/swiglu are row-wise, so batch size never changes
//     per-row numerics;
//   - attention uses blt_vec_dot + blt_softmax_masked_row_inplace (the same
//     inline the dense paths use) with additive 0/-inf mask rows built from
//     absolute positions / group ids.
#include "blt/infer/kv_cache.h"

#include "blt/core/backend.h"
#include "blt/models/local_common.h"
#include "blt/ops/matmul.h"
#include "blt/ops/rmsnorm.h"
#include "blt/ops/swiglu.h"
#include "blt/ops/rope.h"
#include "blt/ops/vecmath.h"
#include "blt/ops/elementwise.h"
#include "blt/infer/rope_gather.h"

#include <math.h>
#include <string.h>


//----------------------------------------------------------------------
// Create / reset / truncate / LCP

blt_kv_cache* blt_kv_cache_create(blt_arena* arena, const blt_local_decoder* decoder,
                                  size_t max_seq_len) {
    BLT_REQUIRE(arena != NULL && decoder != NULL,
        "blt_kv_cache_create: arena and decoder cannot be NULL");
    BLT_REQUIRE(max_seq_len >= 2 && max_seq_len <= decoder->config.max_seq_len,
        "blt_kv_cache_create: max_seq_len must be in [2, decoder max_seq_len]");

    const size_t E = decoder->config.embed_dim;
    const size_t patch_dim = decoder->config.patch_dim ? decoder->config.patch_dim : E;
    BLT_REQUIRE(patch_dim % E == 0, "blt_kv_cache_create: patch_dim must be a multiple of embed_dim");

    blt_kv_cache* c = (blt_kv_cache*)blt_arena_alloc(arena, sizeof(blt_kv_cache), sizeof(void*));
    c->decoder = decoder;
    c->max_seq_len = max_seq_len;
    c->max_subtokens = max_seq_len;   // patches have >= 1 byte each
    c->embed_dim = E;
    c->patch_dim = patch_dim;
    c->k = patch_dim / E;

    const size_t L = decoder->config.num_layers;
    c->self_k = (blt_tensor*)blt_arena_alloc(arena, L * sizeof(blt_tensor), sizeof(void*));
    c->self_v = (blt_tensor*)blt_arena_alloc(arena, L * sizeof(blt_tensor), sizeof(void*));
    c->cross_k = (blt_tensor*)blt_arena_alloc(arena, L * sizeof(blt_tensor), sizeof(void*));
    c->cross_v = (blt_tensor*)blt_arena_alloc(arena, L * sizeof(blt_tensor), sizeof(void*));

    size_t self_shape[2] = {max_seq_len, E};
    size_t cross_shape[2] = {c->max_subtokens, E};
    for (size_t l = 0; l < L; l++) {
        c->self_k[l] = blt_tensor_create(arena, self_shape, 2, BLT_DTYPE_FP32);
        c->self_v[l] = blt_tensor_create(arena, self_shape, 2, BLT_DTYPE_FP32);
        c->cross_k[l] = blt_tensor_create(arena, cross_shape, 2, BLT_DTYPE_FP32);
        c->cross_v[l] = blt_tensor_create(arena, cross_shape, 2, BLT_DTYPE_FP32);
    }

    c->cached_patches = (blt_patch_info*)blt_arena_alloc(
        arena, max_seq_len * sizeof(blt_patch_info), sizeof(void*));
    memset(c->cached_patches, 0, max_seq_len * sizeof(blt_patch_info));

    c->self_len = 0;
    c->cross_num_patches = 0;
    return c;
}

void blt_kv_cache_reset(blt_kv_cache* cache) {
    cache->self_len = 0;
    cache->cross_num_patches = 0;
}

void blt_kv_cache_truncate(blt_kv_cache* cache, size_t self_len, size_t num_patches) {
    BLT_REQUIRE(self_len <= cache->max_seq_len,
        "blt_kv_cache_truncate: self_len out of range");
    BLT_REQUIRE(num_patches <= cache->max_subtokens,
        "blt_kv_cache_truncate: num_patches out of range");
    cache->self_len = self_len;
    cache->cross_num_patches = num_patches;
}

size_t blt_kv_cache_common_patches(const blt_kv_cache* cache,
                                   const blt_patch_info* patches, size_t num_patches) {
    size_t common = 0;
    while (common < cache->cross_num_patches && common < num_patches &&
           cache->cached_patches[common].start_idx == patches[common].start_idx &&
           cache->cached_patches[common].length == patches[common].length) {
        common++;
    }
    return common;
}


//----------------------------------------------------------------------
// Tier-2 refresh: cross-attn K/V for patches [from_patch, num_patches)
//
// Mirrors the dense kv-side projection (patch_in_split @ W_k / W_v).
// Row-wise matmuls make partial rewrites bit-exact.

static void refresh_cross_layer(blt_kv_cache* cache, size_t layer,
                                const blt_cross_attention_weights* xw,
                                const blt_tensor* patch_in_split,
                                size_t from_subtoken, size_t num_new_subtokens) {
    // The sub-token block is contiguous inside patch_in_split: view it as
    // its own [num_new_subtokens, E] matrix and write straight into the
    // cache at the matching offset.
    blt_tensor block;
    blt_tensor_view_2d(&block,
        (float*)patch_in_split->data + from_subtoken * cache->embed_dim,
        num_new_subtokens, cache->embed_dim, patch_in_split->backend);

    blt_tensor k_out, v_out;
    blt_tensor_view_2d(&k_out,
        (float*)cache->cross_k[layer].data + from_subtoken * cache->embed_dim,
        num_new_subtokens, cache->embed_dim, patch_in_split->backend);
    blt_tensor_view_2d(&v_out,
        (float*)cache->cross_v[layer].data + from_subtoken * cache->embed_dim,
        num_new_subtokens, cache->embed_dim, patch_in_split->backend);

    blt_matmul(&block, xw->weight_k, &k_out);
    blt_matmul(&block, xw->weight_v, &v_out);
}

void blt_kv_cache_refresh_cross(blt_kv_cache* cache,
                                const blt_tensor* patch_in,
                                const blt_patch_info* patches, size_t num_patches,
                                size_t from_patch,
                                blt_arena* arena) {
    const blt_local_decoder* dec = cache->decoder;
    BLT_REQUIRE(patch_in != NULL && patches != NULL && arena != NULL,
        "blt_kv_cache_refresh_cross: arguments cannot be NULL");
    BLT_REQUIRE(num_patches >= 1 && from_patch < num_patches,
        "blt_kv_cache_refresh_cross: need from_patch < num_patches");
    BLT_REQUIRE(from_patch <= cache->cross_num_patches,
        "blt_kv_cache_refresh_cross: from_patch must not skip uncached patches "
        "(truncate to the LCP first)");
    blt_check_nd_fp32(patch_in, 2, (const size_t[]){num_patches, cache->patch_dim},
        "blt_kv_cache_refresh_cross: patch_in must be [num_patches, patch_dim] FP32");
    BLT_REQUIRE(num_patches * cache->k <= cache->max_subtokens,
        "blt_kv_cache_refresh_cross: sub-token count exceeds cache capacity");

    blt_tensor patch_in_split;
    blt_tensor_view_2d(&patch_in_split, patch_in->data, num_patches * cache->k,
        cache->embed_dim, patch_in->backend);

    const size_t L = dec->config.num_layers;
    for (size_t l = 0; l < L; l++) {
        if (!blt_local_cross_attn_fires(dec->config.cross_attn_placement,
                                        dec->config.cross_attn_all_layers,
                                        dec->config.num_layers, l)) {
            continue;
        }
        const blt_local_decoder_layer_storage* s = &dec->layers[l];
        blt_cross_attention_weights xw = blt_local_cross_weights_view(s);
        refresh_cross_layer(cache, l, &xw, &patch_in_split,
            from_patch * cache->k, (num_patches - from_patch) * cache->k);
    }

    // Snapshot boundaries and extend validity
    memcpy(&cache->cached_patches[from_patch], &patches[from_patch],
        (num_patches - from_patch) * sizeof(blt_patch_info));
    cache->cross_num_patches = num_patches;
}


//----------------------------------------------------------------------
// Per-head attention kernels (used by the decode step)
//
// Both render their mask as additive 0/-inf rows and stabilize through
// blt_softmax_masked_row_inplace -- the identical inline the dense paths
// use -- so masked-out entries become exactly 0.0f weights and are skipped
// in the V accumulation exactly like attention.c does.

typedef struct {
    float* scores;     // scratch [kv_len]
    float* mask_row;   // scratch [kv_len]
} head_scratch;


// Decoder byte self-attention against tier 1. One head. Causal + sliding
// window by ABSOLUTE position: query row r sits at p = base + r and may
// attend cached keys j <= p (within the window when one is configured).
static void self_attention_head(blt_kv_cache* cache, size_t layer,
                                const head_scratch* hs,
                                size_t n, size_t base, size_t total,
                                float scale, size_t head,
                                const float* q_full,   // [n, E] rotated queries
                                float* combined) {     // [n, E]
    const size_t E = cache->embed_dim;
    const size_t H = cache->decoder->config.num_heads;
    const size_t hd = E / H;
    const size_t offset = head * hd;
    const size_t window = cache->decoder->config.local_window;

    const float* kc = (const float*)cache->self_k[layer].data;
    const float* vc = (const float*)cache->self_v[layer].data;

    for (size_t r = 0; r < n; r++) {
        const size_t p = base + r;
        const float* q = q_full + r * E + offset;
        float* scores = hs->scores;
        float* mask_row = hs->mask_row;

        for (size_t j = 0; j < total; j++) {
            bool allowed = (j <= p);
            if (allowed && window > 0 && p - j >= window) {
                allowed = false;
            }
            if (allowed) {
                scores[j] = blt_vec_dot(q, kc + j * E + offset, hd);
                mask_row[j] = 0.0f;
            } else {
                scores[j] = 0.0f;
                mask_row[j] = -INFINITY;
            }
        }

        blt_softmax_masked_row_inplace(scores, total, r, false, mask_row, scale);

        float* out = combined + r * E + offset;
        for (size_t d = 0; d < hd; d++) {
            out[d] = 0.0f;
        }
        for (size_t j = 0; j < total; j++) {
            const float weight = scores[j];
            if (weight == 0.0f) {
                continue;
            }
            const float* v_j = vc + j * E + offset;
            for (size_t d = 0; d < hd; d++) {
                out[d] += weight * v_j[d];
            }
        }
    }
}

// Cross-attention against tier 2 with the block-diagonal group mask:
// query row r attends exactly the sub-tokens whose patch id equals its own
// group id (draft/MASK rows carry the LAST patch's id). Bidirectional
// within the group, mirroring ctx.cross_mask in local_decoder.c.
static void cross_attention_head(blt_kv_cache* cache, size_t layer,
                                 const float* q_rows,        // [n, E]
                                 const size_t* q_group_ids,  // [n]
                                 const size_t* kv_group_ids, // [total_sub]
                                 size_t n, size_t total_sub, float scale,
                                 float* scores, float* mask_row,
                                 size_t head,
                                 float* combined) {          // [n, E]
    const size_t E = cache->embed_dim;
    const size_t XH = cache->decoder->config.cross_attn_heads;
    const size_t hd = E / XH;
    const size_t offset = head * hd;

    const float* kc = (const float*)cache->cross_k[layer].data;
    const float* vc = (const float*)cache->cross_v[layer].data;

    for (size_t r = 0; r < n; r++) {
        const float* q = q_rows + r * E + offset;
        const size_t g = q_group_ids[r];

        for (size_t j = 0; j < total_sub; j++) {
            if (kv_group_ids[j] == g) {
                scores[j] = blt_vec_dot(q, kc + j * E + offset, hd);
                mask_row[j] = 0.0f;
            } else {
                scores[j] = 0.0f;
                mask_row[j] = -INFINITY;
            }
        }

        blt_softmax_masked_row_inplace(scores, total_sub, r, false, mask_row, scale);

        float* out = combined + r * E + offset;
        for (size_t d = 0; d < hd; d++) {
            out[d] = 0.0f;
        }
        for (size_t j = 0; j < total_sub; j++) {
            const float weight = scores[j];
            if (weight == 0.0f) {
                continue;
            }
            const float* v_j = vc + j * E + offset;
            for (size_t d = 0; d < hd; d++) {
                out[d] += weight * v_j[d];
            }
        }
    }
}


//----------------------------------------------------------------------
// Unified write-through incremental decode step

void blt_kv_decode_step(blt_kv_cache* cache,
                        const blt_tensor* patch_in,
                        const blt_patch_info* patches, size_t num_patches,
                        const blt_tensor* d0_rows,
                        blt_tensor* logits_out,
                        blt_arena* arena) {
    const blt_local_decoder* dec = cache->decoder;
    const blt_local_decoder_config* cfg = &dec->config;
    const size_t E = cache->embed_dim;
    const size_t V = cfg->vocab_size;
    const size_t H = cfg->num_heads;
    const size_t XH = cfg->cross_attn_heads;
    const size_t hd = E / H;
    const size_t xhd = E / XH;
    BLT_REQUIRE(hd > 0 && hd % 2 == 0, "blt_kv_decode_step: RoPE requires an even head_dim");

    BLT_REQUIRE(cache != NULL && patch_in != NULL && patches != NULL &&
                d0_rows != NULL && logits_out != NULL && arena != NULL,
        "blt_kv_decode_step: arguments cannot be NULL");
    BLT_REQUIRE(num_patches >= 1 && num_patches == cache->cross_num_patches,
        "blt_kv_decode_step: tier-2 cache must cover all patches (refresh first)");
    blt_check_nd_fp32(d0_rows, 2, (const size_t[]){0, E},
        "blt_kv_decode_step: d0_rows must be [n, embed_dim] FP32");

    const size_t n = d0_rows->shape[0];
    BLT_REQUIRE(n >= 1, "blt_kv_decode_step: need at least one row");
    const size_t base = cache->self_len;
    const size_t total = base + n;
    BLT_REQUIRE(total <= cache->max_seq_len, "blt_kv_decode_step: exceeds cache capacity");
    blt_check_nd_fp32(logits_out, 2, (const size_t[]){n, V},
        "blt_kv_decode_step: logits_out must be [n, vocab_size] FP32");
    BLT_REQUIRE(num_patches * cache->k <= cache->max_subtokens,
        "blt_kv_decode_step: sub-token count exceeds cache capacity");

    // Group bookkeeping: a new row takes the group of the byte position it
    // sits at (rows past the last patch end resolve to the LAST patch --
    // the Fast-BLT draft-conditioning rule). The kv side is the standard
    // per-sub-token expansion.
    size_t* q_group_ids = (size_t*)blt_arena_alloc(arena, n * sizeof(size_t), 64);
    for (size_t r = 0; r < n; r++) {
        const size_t pos = base + r;
        size_t g = num_patches - 1;
        for (size_t pi = 0; pi < num_patches; pi++) {
            if (pos >= patches[pi].start_idx &&
                pos < patches[pi].start_idx + patches[pi].length) {
                g = pi;
                break;
            }
        }
        q_group_ids[r] = g;
    }
    size_t* kv_group_ids = (size_t*)blt_arena_alloc(
        arena, num_patches * cache->k * sizeof(size_t), 64);
    for (size_t pi = 0; pi < num_patches; pi++) {
        for (size_t s = 0; s < cache->k; s++) {
            kv_group_ids[pi * cache->k + s] = pi;
        }
    }

    const size_t total_sub = num_patches * cache->k;
    head_scratch hs;
    hs.scores = (float*)blt_arena_alloc(arena, total * sizeof(float), sizeof(float));
    hs.mask_row = (float*)blt_arena_alloc(arena, total * sizeof(float), sizeof(float));
    float* x_scores = (float*)blt_arena_alloc(arena, total_sub * sizeof(float), sizeof(float));
    float* x_mask_row = (float*)blt_arena_alloc(arena, total_sub * sizeof(float), sizeof(float));

    const float self_scale = 1.0f / sqrtf((float)hd);
    const float cross_scale = 1.0f / sqrtf((float)xhd);

    // RoPE tables for the new rows' absolute positions.
    size_t* positions = (size_t*)blt_arena_alloc(arena, n * sizeof(size_t), 64);
    for (size_t r = 0; r < n; r++) {
        positions[r] = base + r;
    }
    blt_tensor rope_cos, rope_sin;
    blt_rope_position_gather(&dec->rope_cos_cache, &dec->rope_sin_cache,
        positions, n, &rope_cos, &rope_sin, arena);

    // Split view of patch_in: [num_patches*k, E]
    blt_tensor patch_in_split;
    blt_tensor_view_2d(&patch_in_split, patch_in->data,
        num_patches * cache->k, E, patch_in->backend);

    // Workspaces
    float* q_head = (float*)blt_arena_alloc(arena, n * hd * sizeof(float), sizeof(float));
    float* k_head = (float*)blt_arena_alloc(arena, n * hd * sizeof(float), sizeof(float));
    float* q_rot  = (float*)blt_arena_alloc(arena, n * hd * sizeof(float), sizeof(float));
    float* k_rot  = (float*)blt_arena_alloc(arena, n * hd * sizeof(float), sizeof(float));
    float* q_full = (float*)blt_arena_alloc(arena, n * E * sizeof(float), sizeof(float));
    float* combined = (float*)blt_arena_alloc(arena, n * E * sizeof(float), sizeof(float));

    // Running decoder states for the new rows; reassigned per layer.
    blt_tensor cur = *d0_rows;

    const size_t L = cfg->num_layers;
    for (size_t l = 0; l < L; l++) {
        const blt_local_decoder_layer_storage* s = &dec->layers[l];
        blt_transformer_weights w = blt_local_byte_weights_view(s);
        size_t shape2[2] = {n, E};
        blt_tensor b;

        // -------------------------------------------------------------
        // Cross-attention block (fires per placement): RMSNorm -> Q ->
        // masked attention against tier 2 -> output proj -> residual
        //
        if (blt_local_cross_attn_fires(cfg->cross_attn_placement, cfg->cross_attn_all_layers, L, l)) {
            blt_cross_attention_weights xw = blt_local_cross_weights_view(s);

            blt_tensor normed_q = blt_tensor_create(arena, shape2, 2, BLT_DTYPE_FP32);
            blt_rmsnorm_forward(&cur, &s->cross_norm_weight, &normed_q);

            blt_tensor q_rows = blt_tensor_create(arena, shape2, 2, BLT_DTYPE_FP32);
            blt_matmul(&normed_q, xw.weight_q, &q_rows);

            memset(combined, 0, n * E * sizeof(float));
            for (size_t h = 0; h < XH; h++) {
                cross_attention_head(cache, l, q_rows.data,
                    q_group_ids, kv_group_ids, n, total_sub, cross_scale,
                    x_scores, x_mask_row, h, combined);
            }

            blt_tensor comb_t;
            blt_tensor_view_2d(&comb_t, combined, n, E, cur.backend);
            blt_tensor cross_out = blt_tensor_create(arena, shape2, 2, BLT_DTYPE_FP32);
            blt_matmul(&comb_t, xw.weight_proj, &cross_out);

            b = blt_tensor_create(arena, shape2, 2, BLT_DTYPE_FP32);
            blt_add(&cur, &cross_out, &b);
        } else {
            b = cur;
        }

        // -------------------------------------------------------------
        // Byte transformer block (RMSNorm -> causal RoPE'd self-attn with
        // tier-1 write-through -> proj -> residual -> RMSNorm -> SwiGLU FFN
        // -> residual), mirroring blt_transformer_forward exactly.
        //
        blt_tensor normed1 = blt_tensor_create(arena, shape2, 2, BLT_DTYPE_FP32);
        blt_rmsnorm_forward(&b, w.norm1_weight, &normed1);

        size_t qkv_shape[2] = {n, 3 * E};
        blt_tensor qkv = blt_tensor_create(arena, qkv_shape, 2, BLT_DTYPE_FP32);
        blt_matmul(&normed1, w.attn_qkv_w, &qkv);

        // Rotate Q/K per head (mirror attention.c's apply_rope_to_all_heads),
        // write rotated K and raw V through to tier 1, collect rotated Q.
        float* qkv_data = (float*)qkv.data;
        float* kd = (float*)cache->self_k[l].data;
        float* vd = (float*)cache->self_v[l].data;
        const size_t stride3 = 3 * E;

        blt_tensor q_head_t, q_rot_t, k_head_t, k_rot_t;
        for (size_t h = 0; h < H; h++) {
            const size_t q_off = h * hd;
            const size_t k_off = E + h * hd;
            const size_t v_off = 2 * E + h * hd;

            for (size_t r = 0; r < n; r++) {
                memcpy(q_head + r * hd, qkv_data + r * stride3 + q_off, hd * sizeof(float));
                memcpy(k_head + r * hd, qkv_data + r * stride3 + k_off, hd * sizeof(float));
            }

            blt_tensor_view_3d(&q_head_t, q_head, n, 1, hd, cur.backend);
            blt_tensor_view_3d(&q_rot_t, q_rot, n, 1, hd, cur.backend);
            blt_tensor_view_3d(&k_head_t, k_head, n, 1, hd, cur.backend);
            blt_tensor_view_3d(&k_rot_t, k_rot, n, 1, hd, cur.backend);
            blt_rope_apply(&q_head_t, &rope_cos, &rope_sin, &q_rot_t);
            blt_rope_apply(&k_head_t, &rope_cos, &rope_sin, &k_rot_t);

            for (size_t r = 0; r < n; r++) {
                memcpy(q_full + r * E + q_off, q_rot + r * hd, hd * sizeof(float));
                memcpy(kd + (base + r) * E + q_off, k_rot + r * hd, hd * sizeof(float));
                memcpy(vd + (base + r) * E + q_off, qkv_data + r * stride3 + v_off, hd * sizeof(float));
            }
        }

        memset(combined, 0, n * E * sizeof(float));
        for (size_t h = 0; h < H; h++) {
            self_attention_head(cache, l, &hs, n, base, total, self_scale, h, q_full, combined);
        }

        blt_tensor comb_t;
        blt_tensor_view_2d(&comb_t, combined, n, E, cur.backend);
        blt_tensor attn_out = blt_tensor_create(arena, shape2, 2, BLT_DTYPE_FP32);
        blt_matmul(&comb_t, w.attn_proj_w, &attn_out);

        blt_tensor resid1 = blt_tensor_create(arena, shape2, 2, BLT_DTYPE_FP32);
        blt_add(&b, &attn_out, &resid1);

        // FFN sub-layer: up/gate projections -> SwiGLU -> down projection
        blt_tensor normed2 = blt_tensor_create(arena, shape2, 2, BLT_DTYPE_FP32);
        blt_rmsnorm_forward(&resid1, w.norm2_weight, &normed2);

        const size_t hidden_dim = w.ffn_up_w->shape[1];
        size_t hidden_shape[2] = {n, hidden_dim};
        blt_tensor up_proj = blt_tensor_create(arena, hidden_shape, 2, BLT_DTYPE_FP32);
        blt_matmul(&normed2, w.ffn_up_w, &up_proj);
        blt_tensor gate_proj = blt_tensor_create(arena, hidden_shape, 2, BLT_DTYPE_FP32);
        blt_matmul(&normed2, w.ffn_gate_w, &gate_proj);
        blt_tensor activated = blt_tensor_create(arena, hidden_shape, 2, BLT_DTYPE_FP32);
        blt_swiglu_forward(&gate_proj, &up_proj, &activated);
        blt_tensor ffn_out = blt_tensor_create(arena, shape2, 2, BLT_DTYPE_FP32);
        blt_matmul(&activated, w.ffn_down_w, &ffn_out);

        blt_tensor next = blt_tensor_create(arena, shape2, 2, BLT_DTYPE_FP32);
        blt_add(&resid1, &ffn_out, &next);
        cur = next;
    }

    // LM head over the new rows' final states
    blt_matmul(&cur, &dec->lm_head_weight, logits_out);

    cache->self_len = total;
}
