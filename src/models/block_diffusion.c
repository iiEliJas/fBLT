// BLT-D block-wise diffusion training (Fast-BLT 3.2).
//
// Composition notes:
//   - Cross-attention reuses blt_cross_attention_forward/backward verbatim;
//     the corrupted-block rows are expressible through query group ids, so
//     no bespoke cross-attention code exists here.
//   - Self-attention is bespoke (Figure 5 mask is row-type dependent) but
//     mirrors attention.c op-for-op; its backward mirrors attention.c's
//     backward with the post-softmax weights saved during recomputation.
#include "blt/models/block_diffusion.h"

#include "blt/core/backend.h"
#include "blt/models/local_common.h"
#include "blt/models/cross_attention.h"
#include "blt/ops/matmul.h"
#include "blt/ops/rmsnorm.h"
#include "blt/ops/swiglu.h"
#include "blt/ops/rope.h"
#include "blt/ops/vecmath.h"
#include "blt/ops/elementwise.h"
#include "blt/ops/mask_builder.h"
#include "blt/ops/cross_entropy.h"
#include "blt/infer/rope_gather.h"

#include <math.h>
#include <string.h>


//----------------------------------------------------------------------
// RNG (splitmix64; deterministic, libc-independent)

static uint64_t rng_next(uint64_t* state) {
    uint64_t z = (*state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static float rng_uniform01(uint64_t* state) {
    return (float)(rng_next(state) >> 40) / (float)(1u << 24);
}


//----------------------------------------------------------------------
// Preprocessing

void blt_block_batch_build(blt_block_batch* out, blt_arena* arena,
                           const uint8_t* bytes, size_t N,
                           const blt_patch_info* patches, size_t num_patches,
                           size_t B, uint64_t rng_seed) {
    BLT_REQUIRE(out != NULL && arena != NULL && bytes != NULL && patches != NULL,
        "blt_block_batch_build: arguments cannot be NULL");
    BLT_REQUIRE(N >= 2, "blt_block_batch_build: N must be >= 2");
    BLT_REQUIRE(num_patches >= 2, "blt_block_batch_build: need at least 2 patches to form blocks");
    BLT_REQUIRE(B >= 1, "blt_block_batch_build: block_size must be >= 1");

    const size_t NB = num_patches - 1;      // blocks (first patch excluded)
    const size_t R = B * NB;

    memset(out, 0, sizeof(*out));
    out->num_clean = N;
    out->block_size = B;
    out->num_blocks = NB;
    out->n_block_rows = R;
    out->loss_scale = 1.0f;   // paper objective by default

    out->tokens = (uint32_t*)blt_arena_alloc(arena, R * sizeof(uint32_t), sizeof(uint32_t));
    out->positions = (size_t*)blt_arena_alloc(arena, R * sizeof(size_t), sizeof(size_t));
    out->targets = (uint8_t*)blt_arena_alloc(arena, R * sizeof(uint8_t), sizeof(uint8_t));
    out->cell_valid = (uint8_t*)blt_arena_alloc(arena, R * sizeof(uint8_t), sizeof(uint8_t));
    out->cell_masked = (uint8_t*)blt_arena_alloc(arena, R * sizeof(uint8_t), sizeof(uint8_t));
    out->groups = (size_t*)blt_arena_alloc(arena, R * sizeof(size_t), sizeof(size_t));

    uint64_t rng = rng_seed;
    out->t = rng_uniform01(&rng);
    if (out->t <= 1e-6f) {
        out->t = 1e-6f;   // guard the 1/t loss scaling
    }

    size_t r = 0;
    for (size_t i = 1; i < num_patches; i++) {          // block j = i-1
        const size_t s = patches[i].start_idx;
        for (size_t k = 0; k < B; k++, r++) {
            const size_t pos = s + k;
            const bool valid = pos < N;
            out->cell_valid[r] = valid ? 1 : 0;
            out->positions[r] = valid ? pos : (N - 1);   // PAD rows keep an in-range position
            out->targets[r] = valid ? bytes[pos] : 0;
            // PAD cells reuse the MASK embedding id; excluded from L_mask.
            out->tokens[r] = (!valid || rng_uniform01(&rng) < out->t)
                ? BLT_MASK_TOKEN_ID : (uint32_t)bytes[pos];
            out->cell_masked[r] = (out->cell_valid[r] && out->tokens[r] == BLT_MASK_TOKEN_ID) ? 1 : 0;
            out->groups[r] = i - 1;                      // attends latent o_j (paper's o_{i-1})
        }
    }
}


//----------------------------------------------------------------------
// Shared step context

typedef struct {
    size_t N, R, S, E, V, H, XH, hd, xhd, half, L, k, pdim;
    blt_tensor d0;                   // [S, E]
    blt_tensor rope_cos, rope_sin;   // [S, half], gathered by position
    blt_tensor self_mask;            // [S, S] additive 0/-inf (Figure 5)
    size_t* q_group_ids;             // [S]
    size_t* kv_group_ids;            // [M*k]
    blt_tensor patch_in_split;       // [M*k, E]
} diff_ctx;


static void diff_ctx_build(diff_ctx* c, const blt_local_decoder* model,
                           const blt_tensor* byte_hidden_in,
                           const blt_tensor* patch_in,
                           const blt_patch_info* patches, size_t num_patches,
                           const blt_block_batch* batch,
                           blt_d0_mode d0_mode,
                           blt_block_diffusion_mode mask_mode,
                           blt_arena* arena) {
    const blt_local_decoder_config* cfg = &model->config;
    c->N = batch->num_clean;
    c->R = batch->n_block_rows;
    c->S = c->N + c->R;
    c->E = cfg->embed_dim;
    c->V = cfg->vocab_size;
    c->H = cfg->num_heads;
    c->XH = cfg->cross_attn_heads;
    c->hd = c->E / c->H;
    c->xhd = c->E / c->XH;
    c->half = c->hd / 2;
    c->L = cfg->num_layers;
    c->pdim = cfg->patch_dim ? cfg->patch_dim : c->E;
    c->k = c->pdim / c->E;

    BLT_REQUIRE(c->hd > 0 && c->hd % 2 == 0, "diffusion: RoPE requires an even head_dim");
    BLT_REQUIRE(byte_hidden_in != NULL && patch_in != NULL && batch != NULL && arena != NULL,
        "diffusion: arguments cannot be NULL");
    BLT_REQUIRE(byte_hidden_in->shape[0] == c->N && byte_hidden_in->shape[1] == c->E,
        "diffusion: byte_hidden_in must be [N, embed_dim]");
    BLT_REQUIRE(patch_in->shape[0] == num_patches && patch_in->shape[1] == c->pdim,
        "diffusion: patch_in must be [num_patches, patch_dim]");
    BLT_REQUIRE(c->S >= 2 && c->S <= cfg->max_seq_len,
        "diffusion: total sequence length outside [2, decoder max_seq_len]");

    // D_0: clean rows carry h_final content; block rows follow the policy.
    size_t d0_shape[2] = {c->S, c->E};
    c->d0 = blt_tensor_create(arena, d0_shape, 2, BLT_DTYPE_FP32);   // zero-filled
    memcpy(c->d0.data, byte_hidden_in->data, c->N * c->E * sizeof(float));
    if (d0_mode == BLT_D0_LEARNED) {
        const float* table = (const float*)model->d0_embed_weight.data;
        float* dst = (float*)c->d0.data + c->N * c->E;
        for (size_t r = 0; r < c->R; r++) {
            memcpy(dst + r * c->E, table + batch->tokens[r] * c->E, c->E * sizeof(float));
        }
    }
    // ZEROS (default): keep the zero-fill.

    // Positions: identity for clean rows, recorded originals for block rows.
    size_t* positions = (size_t*)blt_arena_alloc(arena, c->S * sizeof(size_t), 64);
    for (size_t i = 0; i < c->N; i++) {
        positions[i] = i;
    }
    for (size_t r = 0; r < c->R; r++) {
        positions[c->N + r] = batch->positions[r];
    }
    blt_rope_position_gather(&model->rope_cos_cache, &model->rope_sin_cache,
        positions, c->S, &c->rope_cos, &c->rope_sin, arena);

    // Query group ids: clean rows take their own patch id (repo BLT rule);
    // block rows take their assigned latent group (paper o_{i-1} rule).
    c->q_group_ids = (size_t*)blt_arena_alloc(arena, c->S * sizeof(size_t), 64);
    for (size_t i = 0; i < c->N; i++) {
        size_t g = num_patches - 1;
        for (size_t pi = 0; pi < num_patches; pi++) {
            if (i >= patches[pi].start_idx &&
                i < patches[pi].start_idx + patches[pi].length) {
                g = pi;
                break;
            }
        }
        c->q_group_ids[i] = g;
    }
    for (size_t r = 0; r < c->R; r++) {
        c->q_group_ids[c->N + r] = batch->groups[r];
    }

    c->kv_group_ids = (size_t*)blt_arena_alloc(
        arena, num_patches * c->k * sizeof(size_t), 64);
    for (size_t pi = 0; pi < num_patches; pi++) {
        for (size_t s2 = 0; s2 < c->k; s2++) {
            c->kv_group_ids[pi * c->k + s2] = pi;
        }
    }

    blt_block_diffusion_config mc = {
        .mode = mask_mode,
        .seq_len = c->S,
        .num_clean = c->N,
        .block_size = batch->block_size,
    };
    blt_build_block_diffusion_mask(&mc, &c->self_mask, arena);

    blt_tensor_view_2d(&c->patch_in_split, patch_in->data,
        num_patches * c->k, c->E, patch_in->backend);
}

static void diff_cross_config(const diff_ctx* c, size_t num_patches,
                              blt_cross_attention_config* xc, blt_mask_config* xm) {
    memset(xc, 0, sizeof(*xc));
    memset(xm, 0, sizeof(*xm));
    xm->seq_len_q = c->S;
    xm->seq_len_kv = num_patches * c->k;
    xm->query_group_ids = c->q_group_ids;
    xm->kv_group_ids = c->kv_group_ids;
    xm->bidirectional_within_group = true;
    xm->is_causal = false;

    xc->embed_dim = c->E;
    xc->patch_dim = c->pdim;
    xc->split_mode = (c->k == 1) ? BLT_CROSS_ATTN_NO_SPLIT : BLT_CROSS_ATTN_SPLIT_KV;
    xc->num_heads = c->XH;
    xc->mask_config = xm;
}


//----------------------------------------------------------------------
// Forward machinery

typedef struct {
    bool has_cross;
    blt_tensor d_in;         // [S, E] entering this layer
    blt_tensor normed_q;     // cross rmsnorm output
    blt_tensor b_in;         // input to the byte block (post-cross residual)
    blt_tensor normed1;      // self-attn rmsnorm output
    blt_tensor qkv;          // post-RoPE [S, 3E]
    blt_tensor weights_all;  // [H, S, S] post-softmax self-attn weights
    blt_tensor combined;     // [S, E] view over the combined-heads buffer
    blt_tensor resid1;       // [S, E]
    blt_tensor normed2;      // ffn rmsnorm output
    blt_tensor gate, up, act;// [S, hidden]
} diff_layer_cache;


// One bespoke self-attention over the full sequence against the Figure 5
// mask. Rotates Q/K per head (gathered tables), saves post-softmax weights
// for backward, writes the projected output into attn_out.
static void diffusion_self_attention(const diff_ctx* c,
                                     const blt_transformer_weights* w,
                                     const blt_tensor* normed1,
                                     blt_tensor* attn_out,
                                     blt_tensor* qkv_out,
                                     blt_tensor* weights_all,
                                     blt_tensor* combined_out,
                                     blt_arena* arena) {
    const size_t S = c->S, E = c->E, H = c->H, hd = c->hd;
    const float scale = 1.0f / sqrtf((float)hd);

    size_t qkv_shape[2] = {S, 3 * E};
    *qkv_out = blt_tensor_create(arena, qkv_shape, 2, BLT_DTYPE_FP32);
    blt_matmul(normed1, w->attn_qkv_w, qkv_out);

    // Rotate Q/K per head in place (mirror attention.c).
    float* qkv_data = (float*)qkv_out->data;
    const size_t stride3 = 3 * E;
    float* q_head = (float*)blt_arena_alloc(arena, S * hd * sizeof(float), sizeof(float));
    float* k_head = (float*)blt_arena_alloc(arena, S * hd * sizeof(float), sizeof(float));
    float* q_rot = (float*)blt_arena_alloc(arena, S * hd * sizeof(float), sizeof(float));
    float* k_rot = (float*)blt_arena_alloc(arena, S * hd * sizeof(float), sizeof(float));

    blt_tensor q_head_t, q_rot_t, k_head_t, k_rot_t;
    for (size_t h = 0; h < H; h++) {
        const size_t q_off = h * hd;
        const size_t k_off = E + h * hd;
        for (size_t i = 0; i < S; i++) {
            memcpy(q_head + i * hd, qkv_data + i * stride3 + q_off, hd * sizeof(float));
            memcpy(k_head + i * hd, qkv_data + i * stride3 + k_off, hd * sizeof(float));
        }
        blt_tensor_view_3d(&q_head_t, q_head, S, 1, hd, normed1->backend);
        blt_tensor_view_3d(&q_rot_t, q_rot, S, 1, hd, normed1->backend);
        blt_tensor_view_3d(&k_head_t, k_head, S, 1, hd, normed1->backend);
        blt_tensor_view_3d(&k_rot_t, k_rot, S, 1, hd, normed1->backend);
        blt_rope_apply(&q_head_t, &c->rope_cos, &c->rope_sin, &q_rot_t);
        blt_rope_apply(&k_head_t, &c->rope_cos, &c->rope_sin, &k_rot_t);
        for (size_t i = 0; i < S; i++) {
            memcpy(qkv_data + i * stride3 + q_off, q_rot + i * hd, hd * sizeof(float));
            memcpy(qkv_data + i * stride3 + k_off, k_rot + i * hd, hd * sizeof(float));
        }
    }

    // Per-head masked attention against the Figure 5 mask; save post-softmax
    // weights for the backward pass.
    size_t wa_shape[3] = {H, S, S};
    *weights_all = blt_tensor_create(arena, wa_shape, 3, BLT_DTYPE_FP32);
    float* combined = (float*)blt_arena_alloc(arena, S * E * sizeof(float), sizeof(float));
    memset(combined, 0, S * E * sizeof(float));

    const float* mask_base = (const float*)c->self_mask.data;

    for (size_t h = 0; h < H; h++) {
        const size_t q_off = h * hd;
        const size_t v_off = 2 * E + h * hd;
        float* W = (float*)weights_all->data + h * S * S;

        for (size_t i = 0; i < S; i++) {
            const float* q_i = qkv_data + i * stride3 + q_off;
            float* scores = W + i * S;
            const float* mask_row = mask_base + i * S;

            for (size_t j = 0; j < S; j++) {
                scores[j] = blt_vec_dot(q_i, qkv_data + j * stride3 + E + q_off, hd);
            }
            blt_softmax_masked_row_inplace(scores, S, i, false, mask_row, scale);
        }

        for (size_t i = 0; i < S; i++) {
            float* out_i = combined + i * E + q_off;
            for (size_t d = 0; d < hd; d++) {
                out_i[d] = 0.0f;
            }
            const float* scores = W + i * S;
            for (size_t j = 0; j < S; j++) {
                const float weight = scores[j];
                if (weight == 0.0f) {
                    continue;
                }
                const float* v_j = qkv_data + j * stride3 + v_off;
                for (size_t d = 0; d < hd; d++) {
                    out_i[d] += weight * v_j[d];
                }
            }
        }
    }

    blt_tensor_view_2d(combined_out, combined, S, E, normed1->backend);
    blt_matmul(combined_out, w->attn_proj_w, attn_out);
}

// Recomputes the forward pass, caching per-layer intermediates plus the
// final states (needed by the LM-head backward).
static void forward_with_caches(const blt_local_decoder* model,
                                const diff_ctx* c,
                                size_t num_patches,
                                blt_tensor* logits,
                                blt_tensor* final_states,
                                diff_layer_cache** caches_out,
                                blt_arena* arena) {
    *caches_out = (diff_layer_cache*)blt_arena_alloc(
        arena, c->L * sizeof(diff_layer_cache), sizeof(void*));
    memset(*caches_out, 0, c->L * sizeof(diff_layer_cache));
    diff_layer_cache* caches = *caches_out;

    blt_tensor cur = c->d0;

    for (size_t l = 0; l < c->L; l++) {
        const blt_local_decoder_layer_storage* s = &model->layers[l];
        blt_transformer_weights w = blt_local_byte_weights_view(s);
        diff_layer_cache* ca = &caches[l];
        ca->d_in = cur;
        size_t shape2[2] = {c->S, c->E};

        blt_tensor b;
        if (blt_local_cross_attn_fires(model->config.cross_attn_placement,
                                       model->config.cross_attn_all_layers, c->L, l)) {
            ca->has_cross = true;
            blt_cross_attention_weights xw = blt_local_cross_weights_view(s);

            ca->normed_q = blt_tensor_create(arena, shape2, 2, BLT_DTYPE_FP32);
            blt_rmsnorm_forward(&cur, &s->cross_norm_weight, &ca->normed_q);

            blt_cross_attention_config xc;
            blt_mask_config xm;
            diff_cross_config(c, num_patches, &xc, &xm);

            blt_tensor cross_out = blt_tensor_create(arena, shape2, 2, BLT_DTYPE_FP32);
            blt_cross_attention_forward(&ca->normed_q, &c->patch_in_split, &xw,
                &cross_out, &xc, arena);

            ca->b_in = blt_tensor_create(arena, shape2, 2, BLT_DTYPE_FP32);
            blt_add(&cur, &cross_out, &ca->b_in);
            b = ca->b_in;
        } else {
            ca->b_in = cur;
            b = cur;
        }

        blt_tensor attn_out = blt_tensor_create(arena, shape2, 2, BLT_DTYPE_FP32);
        ca->normed1 = blt_tensor_create(arena, shape2, 2, BLT_DTYPE_FP32);
        blt_rmsnorm_forward(&b, w.norm1_weight, &ca->normed1);
        diffusion_self_attention(c, &w, &ca->normed1, &attn_out,
            &ca->qkv, &ca->weights_all, &ca->combined, arena);

        ca->resid1 = blt_tensor_create(arena, shape2, 2, BLT_DTYPE_FP32);
        blt_add(&b, &attn_out, &ca->resid1);

        ca->normed2 = blt_tensor_create(arena, shape2, 2, BLT_DTYPE_FP32);
        blt_rmsnorm_forward(&ca->resid1, w.norm2_weight, &ca->normed2);

        const size_t hidden_dim = w.ffn_up_w->shape[1];
        size_t hidden_shape[2] = {c->S, hidden_dim};
        ca->up = blt_tensor_create(arena, hidden_shape, 2, BLT_DTYPE_FP32);
        blt_matmul(&ca->normed2, w.ffn_up_w, &ca->up);
        ca->gate = blt_tensor_create(arena, hidden_shape, 2, BLT_DTYPE_FP32);
        blt_matmul(&ca->normed2, w.ffn_gate_w, &ca->gate);
        ca->act = blt_tensor_create(arena, hidden_shape, 2, BLT_DTYPE_FP32);
        blt_swiglu_forward(&ca->gate, &ca->up, &ca->act);
        blt_tensor ffn_out = blt_tensor_create(arena, shape2, 2, BLT_DTYPE_FP32);
        blt_matmul(&ca->act, w.ffn_down_w, &ffn_out);

        blt_tensor next = blt_tensor_create(arena, shape2, 2, BLT_DTYPE_FP32);
        blt_add(&ca->resid1, &ffn_out, &next);
        cur = next;
    }

    *final_states = cur;
    size_t logits_shape[2] = {c->S, c->V};
    *logits = blt_tensor_create(arena, logits_shape, 2, BLT_DTYPE_FP32);
    blt_matmul(final_states, &model->lm_head_weight, logits);
}

// Builds a UINT8 tensor view companion for the CE targets (bytes[1..N)).
static void make_bytes_tensor(blt_arena* arena, const uint8_t* bytes, size_t len,
                              blt_tensor* out) {
    size_t shape[1] = {len};
    *out = blt_tensor_create(arena, shape, 1, BLT_DTYPE_UINT8);
    memcpy(out->data, bytes, len);
}


//----------------------------------------------------------------------
// Public forward

void blt_local_decoder_forward_diffusion(
    const blt_local_decoder* model,
    const blt_tensor* byte_hidden_in,
    const blt_tensor* patch_in,
    const blt_patch_info* patches, size_t num_patches,
    const uint8_t* clean_bytes,
    const blt_block_batch* batch,
    blt_d0_mode d0_mode,
    blt_tensor* logits_out,
    blt_tensor* loss_out,
    blt_arena* arena
) {
    BLT_REQUIRE(model != NULL && clean_bytes != NULL && logits_out != NULL &&
                loss_out != NULL && arena != NULL,
        "forward_diffusion: arguments cannot be NULL");

    diff_ctx c;
    diff_ctx_build(&c, model, byte_hidden_in, patch_in, patches, num_patches,
        batch, d0_mode, BLT_BDM_TRAIN, arena);

    size_t out_shape[2] = {c.S, c.V};
    blt_check_nd_fp32(logits_out, 2, out_shape,
        "forward_diffusion: logits_out must be [N + n_block_rows, vocab_size] FP32");

    diff_layer_cache* caches;
    blt_tensor logits, final_states;
    forward_with_caches(model, &c, num_patches, &logits, &final_states,
        &caches, arena);
    memcpy(logits_out->data, logits.data, c.S * c.V * sizeof(float));

    // Losses (Eq. 7): L_clean via the legacy CE path over clean rows +
    // L_mask/t accumulated over masked block cells.
    float loss = 0.0f;

    if (c.N >= 2) {
        blt_tensor logits_view;
        blt_tensor_view_2d(&logits_view, logits.data, c.N - 1, c.V, logits.backend);

        blt_tensor bytes_t;
        make_bytes_tensor(arena, clean_bytes, c.N, &bytes_t);
        blt_tensor targets_view;
        view_1d_offset(&targets_view, &bytes_t, 1, c.N - 1);

        blt_cross_entropy_forward(&logits_view, &targets_view, loss_out);
        loss = ((const float*)loss_out->data)[0];
    }

    if (batch->t > 0.0f) {
        float l_mask = 0.0f;
        for (size_t r = 0; r < c.R; r++) {
            if (!batch->cell_masked[r]) continue;
            const float* row = (const float*)logits.data + (c.N + r) * c.V;
            // -log p[target], stable
            float max_val = row[0];
            for (size_t v = 1; v < c.V; v++) {
                if (row[v] > max_val) max_val = row[v];
            }
            float sum = 0.0f, pt = 0.0f;
            for (size_t v = 0; v < c.V; v++) {
                float e = expf(row[v] - max_val);
                sum += e;
                if (v == (size_t)batch->targets[r]) pt = e;
            }
            // floor guards -log(0) when the target probability underflows
            l_mask += -logf(fmaxf(pt / sum, 1e-9f));
        }
        loss += batch->loss_scale * l_mask / batch->t;
    }

    ((float*)loss_out->data)[0] = loss;
}


//----------------------------------------------------------------------
// Backward

void blt_local_decoder_backward_diffusion(
    const blt_local_decoder* model,
    const blt_tensor* byte_hidden_in,
    const blt_tensor* patch_in,
    const blt_patch_info* patches, size_t num_patches,
    const uint8_t* clean_bytes,
    const blt_block_batch* batch,
    blt_d0_mode d0_mode,
    blt_tensor* grad_byte_hidden_in,
    blt_tensor* grad_patch_in,
    blt_local_decoder_grad* grad,
    blt_arena* arena
) {
    BLT_REQUIRE(model != NULL && clean_bytes != NULL && grad_byte_hidden_in != NULL &&
                grad_patch_in != NULL && grad != NULL && arena != NULL,
        "backward_diffusion: arguments cannot be NULL");

    diff_ctx c;
    diff_ctx_build(&c, model, byte_hidden_in, patch_in, patches, num_patches,
        batch, d0_mode, BLT_BDM_TRAIN, arena);

    blt_check_nd_fp32(grad_byte_hidden_in, 2, (const size_t[]){c.N, c.E},
        "backward_diffusion: grad_byte_hidden_in must be [N, embed_dim] FP32");
    blt_check_nd_fp32(grad_patch_in, 2, (const size_t[]){num_patches, c.pdim},
        "backward_diffusion: grad_patch_in must be [num_patches, patch_dim] FP32");
    blt_check_nd_fp32(&grad->lm_head_grad, 2, (const size_t[]){c.E, c.V},
        "backward_diffusion: lm_head_grad must be [embed_dim, vocab_size] FP32");

    // STEP 1: recompute the forward, caching intermediates.
    diff_layer_cache* caches;
    blt_tensor logits, final_states;
    forward_with_caches(model, &c, num_patches, &logits, &final_states,
        &caches, arena);

    // STEP 2: terminal gradient dL/d(logits).
    blt_tensor grad_logits = blt_tensor_create(arena,
        (size_t[2]){c.S, c.V}, 2, BLT_DTYPE_FP32);   // zero-init

    if (c.N >= 2) {
        blt_tensor logits_view;
        blt_tensor_view_2d(&logits_view, logits.data, c.N - 1, c.V, logits.backend);

        blt_tensor bytes_t;
        make_bytes_tensor(arena, clean_bytes, c.N, &bytes_t);
        blt_tensor targets_view;
        view_1d_offset(&targets_view, &bytes_t, 1, c.N - 1);

        size_t gv_shape[2] = {c.N - 1, c.V};
        blt_tensor grad_view = blt_tensor_create(arena, gv_shape, 2, BLT_DTYPE_FP32);
        blt_cross_entropy_backward(&logits_view, &targets_view, &grad_view);
        memcpy(grad_logits.data, grad_view.data, (c.N - 1) * c.V * sizeof(float));
    }

    if (batch->t > 0.0f && batch->loss_scale != 0.0f) {
        const float inv_t = batch->loss_scale / batch->t;
        float* gl = (float*)grad_logits.data;
        for (size_t r = 0; r < c.R; r++) {
            if (!batch->cell_masked[r]) continue;
            const size_t row = c.N + r;
            const float* lr = (const float*)logits.data + row * c.V;
            const uint8_t tgt = batch->targets[r];

            float max_val = lr[0];
            for (size_t v = 1; v < c.V; v++) {
                if (lr[v] > max_val) max_val = lr[v];
            }
            float sum = 0.0f;
            for (size_t v = 0; v < c.V; v++) {
                sum += expf(lr[v] - max_val);
            }
            float* g = gl + row * c.V;
            for (size_t v = 0; v < c.V; v++) {
                const float p = expf(lr[v] - max_val) / sum;
                g[v] += inv_t * (p - (v == (size_t)tgt ? 1.0f : 0.0f));
            }
        }
    }

    // STEP 3: LM head backward -> dL/d(final states).
    blt_tensor dh = blt_tensor_create(arena, (size_t[2]){c.S, c.E}, 2, BLT_DTYPE_FP32);
    blt_matmul_backward(&final_states, &model->lm_head_weight, &grad_logits,
        &dh, &grad->lm_head_grad);

    // STEP 4: reverse layer loop.
    blt_tensor dP_split = blt_tensor_create(arena,
        (size_t[2]){num_patches * c.k, c.E}, 2, BLT_DTYPE_FP32);   // zero-init

    const size_t stride3 = 3 * c.E;
    const float self_scale = 1.0f / sqrtf((float)c.hd);

    for (size_t li = c.L; li-- > 0;) {
        const blt_local_decoder_layer_storage* s = &model->layers[li];
        blt_transformer_weights w = blt_local_byte_weights_view(s);
        blt_local_decoder_layer_grad* lg = &grad->layer_grads[li];
        const diff_layer_cache* ca = &caches[li];

        size_t shape2[2] = {c.S, c.E};
        const size_t hidden_dim = w.ffn_up_w->shape[1];
        size_t hidden_shape[2] = {c.S, hidden_dim};

        // ---------------- FFN backward ----------------
        // next = resid1 + ffn_out ; ffn_out = act @ ffn_down_w
        blt_tensor g_act = blt_tensor_create(arena, hidden_shape, 2, BLT_DTYPE_FP32);
        blt_matmul_backward(&ca->act, w.ffn_down_w, &dh, &g_act, &lg->ffn_down_w);

        blt_tensor g_gate = blt_tensor_create(arena, hidden_shape, 2, BLT_DTYPE_FP32);
        blt_tensor g_up = blt_tensor_create(arena, hidden_shape, 2, BLT_DTYPE_FP32);
        blt_swiglu_backward(&g_act, &ca->gate, &ca->up, &g_gate, &g_up);

        blt_tensor g_n2g = blt_tensor_create(arena, shape2, 2, BLT_DTYPE_FP32);
        blt_matmul_backward(&ca->normed2, w.ffn_gate_w, &g_gate, &g_n2g, &lg->ffn_gate_w);
        blt_tensor g_n2u = blt_tensor_create(arena, shape2, 2, BLT_DTYPE_FP32);
        blt_matmul_backward(&ca->normed2, w.ffn_up_w, &g_up, &g_n2u, &lg->ffn_up_w);

        blt_tensor g_normed2 = blt_tensor_create(arena, shape2, 2, BLT_DTYPE_FP32);
        blt_add(&g_n2g, &g_n2u, &g_normed2);

        blt_tensor g_r1_norm = blt_tensor_create(arena, shape2, 2, BLT_DTYPE_FP32);
        blt_rmsnorm_backward(&g_normed2, &ca->resid1, w.norm2_weight,
            &g_r1_norm, &lg->norm2_weight);

        // resid1 receives the direct path AND the norm path
        blt_tensor g_resid1 = blt_tensor_create(arena, shape2, 2, BLT_DTYPE_FP32);
        blt_add(&dh, &g_r1_norm, &g_resid1);

        // ---------------- self-attention backward ----------------
        // resid1 = b_in + attn_out ; attn_out = combined @ proj
        blt_tensor g_combined = blt_tensor_create(arena, shape2, 2, BLT_DTYPE_FP32);
        blt_matmul_backward(&ca->combined, w.attn_proj_w, &g_resid1,
            &g_combined, &lg->attn_proj_w);

        // mirror attention.c backward using the cached post-RoPE qkv and
        // post-softmax weights
        blt_tensor g_qkv = blt_tensor_create(arena,
            (size_t[2]){c.S, stride3}, 2, BLT_DTYPE_FP32);   // zero-init
        float* gqkv = (float*)g_qkv.data;
        const float* qkv = (const float*)ca->qkv.data;
        const float* gcomb = (const float*)g_combined.data;

        float* gW_buf = (float*)blt_arena_alloc(arena, c.S * c.S * sizeof(float), sizeof(float));
        float* gs_buf = (float*)blt_arena_alloc(arena, c.S * c.S * sizeof(float), sizeof(float));

        for (size_t h = 0; h < c.H; h++) {
            const size_t q_off = h * c.hd;
            const size_t k_off = c.E + h * c.hd;
            const size_t v_off = 2 * c.E + h * c.hd;
            const float* W = (const float*)ca->weights_all.data + h * c.S * c.S;

            // grad wrt V accumulation + grad wrt softmax weights
            memset(gW_buf, 0, c.S * c.S * sizeof(float));
            for (size_t i = 0; i < c.S; i++) {
                for (size_t j = 0; j < c.S; j++) {
                    float dot = 0.0f;
                    for (size_t d = 0; d < c.hd; d++) {
                        dot += gcomb[i * c.E + q_off + d] * qkv[j * stride3 + v_off + d];
                    }
                    gW_buf[i * c.S + j] = dot;

                    const float wgt = W[i * c.S + j];
                    if (wgt != 0.0f) {
                        float* gv = gqkv + j * stride3 + v_off;
                        for (size_t d = 0; d < c.hd; d++) {
                            gv[d] += wgt * gcomb[i * c.E + q_off + d];
                        }
                    }
                }
            }

            // softmax backward per row
            for (size_t i = 0; i < c.S; i++) {
                float dotGW = 0.0f;
                for (size_t j = 0; j < c.S; j++) {
                    dotGW += gW_buf[i * c.S + j] * W[i * c.S + j];
                }
                for (size_t j = 0; j < c.S; j++) {
                    gs_buf[i * c.S + j] =
                        W[i * c.S + j] * (gW_buf[i * c.S + j] - dotGW);
                }
            }

            // scores backward
            for (size_t i = 0; i < c.S; i++) {
                for (size_t j = 0; j < c.S; j++) {
                    const float gs = gs_buf[i * c.S + j];
                    if (gs == 0.0f) continue;
                    float* gq = gqkv + i * stride3 + q_off;
                    const float* k_j = qkv + j * stride3 + k_off;
                    for (size_t d = 0; d < c.hd; d++) {
                        gq[d] += self_scale * gs * k_j[d];
                    }
                    float* gk = gqkv + j * stride3 + k_off;
                    const float* q_i = qkv + i * stride3 + q_off;
                    for (size_t d = 0; d < c.hd; d++) {
                        gk[d] += self_scale * gs * q_i[d];
                    }
                }
            }
        }

        // RoPE backward on the Q/K gradients (same gathered tables)
        {
            float* gq_head = (float*)blt_arena_alloc(arena, c.S * c.hd * sizeof(float), sizeof(float));
            float* gk_head = (float*)blt_arena_alloc(arena, c.S * c.hd * sizeof(float), sizeof(float));
            float* gq_unrot = (float*)blt_arena_alloc(arena, c.S * c.hd * sizeof(float), sizeof(float));
            float* gk_unrot = (float*)blt_arena_alloc(arena, c.S * c.hd * sizeof(float), sizeof(float));
            blt_tensor gqh_t, gqu_t, gkh_t, gku_t;
            for (size_t h = 0; h < c.H; h++) {
                const size_t q_off = h * c.hd;
                const size_t k_off = c.E + h * c.hd;
                for (size_t i = 0; i < c.S; i++) {
                    memcpy(gq_head + i * c.hd, gqkv + i * stride3 + q_off, c.hd * sizeof(float));
                    memcpy(gk_head + i * c.hd, gqkv + i * stride3 + k_off, c.hd * sizeof(float));
                }
                blt_tensor_view_3d(&gqh_t, gq_head, c.S, 1, c.hd, dh.backend);
                blt_tensor_view_3d(&gqu_t, gq_unrot, c.S, 1, c.hd, dh.backend);
                blt_tensor_view_3d(&gkh_t, gk_head, c.S, 1, c.hd, dh.backend);
                blt_tensor_view_3d(&gku_t, gk_unrot, c.S, 1, c.hd, dh.backend);
                blt_rope_apply_backward(&gqh_t, &c.rope_cos, &c.rope_sin, &gqu_t);
                blt_rope_apply_backward(&gkh_t, &c.rope_cos, &c.rope_sin, &gku_t);
                for (size_t i = 0; i < c.S; i++) {
                    memcpy(gqkv + i * stride3 + q_off, gq_unrot + i * c.hd, c.hd * sizeof(float));
                    memcpy(gqkv + i * stride3 + k_off, gk_unrot + i * c.hd, c.hd * sizeof(float));
                }
            }
        }

        blt_tensor g_normed1 = blt_tensor_create(arena, shape2, 2, BLT_DTYPE_FP32);
        blt_matmul_backward(&ca->normed1, w.attn_qkv_w, &g_qkv,
            &g_normed1, &lg->attn_qkv_w);

        blt_tensor g_b_norm = blt_tensor_create(arena, shape2, 2, BLT_DTYPE_FP32);
        blt_rmsnorm_backward(&g_normed1, &ca->b_in, w.norm1_weight,
            &g_b_norm, &lg->norm1_weight);

        blt_tensor g_b = blt_tensor_create(arena, shape2, 2, BLT_DTYPE_FP32);
        blt_add(&g_resid1, &g_b_norm, &g_b);

        // ---------------- cross-attention backward ----------------
        if (ca->has_cross) {
            blt_cross_attention_weights xw = blt_local_cross_weights_view(s);
            blt_cross_attention_grad xg = {0};
            xg.grad_weight_q = lg->cross_weight_q;
            xg.grad_weight_k = lg->cross_weight_k;
            xg.grad_weight_v = lg->cross_weight_v;
            xg.grad_weight_proj = lg->cross_weight_proj;

            blt_cross_attention_config xc;
            blt_mask_config xm;
            diff_cross_config(&c, num_patches, &xc, &xm);

            blt_tensor grad_normed_q = blt_tensor_create(arena, shape2, 2, BLT_DTYPE_FP32);
            blt_tensor grad_kv_tmp = blt_tensor_create(arena,
                (size_t[2]){num_patches * c.k, c.E}, 2, BLT_DTYPE_FP32);
            blt_cross_attention_backward(&ca->normed_q, &c.patch_in_split, &xw,
                &g_b, &grad_normed_q, &grad_kv_tmp, &xg, &xc, arena);

            blt_tensor dP_new = blt_tensor_create(arena,
                (size_t[2]){num_patches * c.k, c.E}, 2, BLT_DTYPE_FP32);
            blt_add(&dP_split, &grad_kv_tmp, &dP_new);
            dP_split = dP_new;

            blt_tensor g_din_norm = blt_tensor_create(arena, shape2, 2, BLT_DTYPE_FP32);
            blt_rmsnorm_backward(&grad_normed_q, &ca->d_in, &s->cross_norm_weight,
                &g_din_norm, &lg->cross_norm_weight);

            blt_tensor dh_prev = blt_tensor_create(arena, shape2, 2, BLT_DTYPE_FP32);
            blt_add(&g_b, &g_din_norm, &dh_prev);
            dh = dh_prev;
        } else {
            dh = g_b;
        }
    }

    // STEP 5: terminal gradients.
    memcpy(grad_byte_hidden_in->data, dh.data, c.N * c.E * sizeof(float));

    if (d0_mode == BLT_D0_LEARNED) {
        float* tbl_grad = (float*)grad->d0_embed_grad.data;
        const float* dh_data = (const float*)dh.data;
        for (size_t r = 0; r < c.R; r++) {
            const size_t tok = batch->tokens[r];
            float* dst = tbl_grad + tok * c.E;
            const float* src = dh_data + (c.N + r) * c.E;
            for (size_t e = 0; e < c.E; e++) {
                dst[e] += src[e];
            }
        }
    }

    // Reinterpret dP_split back as [num_patches, patch_dim]
    blt_tensor dP;
    blt_tensor_view_2d(&dP, dP_split.data, num_patches, c.pdim, dP_split.backend);
    memcpy(grad_patch_in->data, dP.data, num_patches * c.pdim * sizeof(float));
}

//----------------------------------------------------------------------
// Public inference forward (Fast-BLT section 3.1.1 / Algorithm 1)
//
// Same decoder pass as blt_local_decoder_forward_diffusion but with the
// INFERENCE self-attention pattern: clean rows causal; every block row
// attends all clean positions plus the whole block section
// bidirectionally. No loss is computed -- the caller drives iterative
// unmasking over batch->tokens / cell_masked between passes.
//
// The caller owns the batch: fill tokens (byte ids or BLT_MASK_TOKEN_ID),
// positions (true future positions for RoPE), groups (latent index each
// block row cross-attends; the paper's rule is o_M for ALL block rows),
// cell_masked, num_clean = N, n_block_rows = B, t = 0.
void blt_local_decoder_forward_diffusion_infer(
    const blt_local_decoder* model,
    const blt_tensor* byte_hidden_in,
    const blt_tensor* patch_in,
    const blt_patch_info* patches, size_t num_patches,
    const blt_block_batch* batch,
    blt_d0_mode d0_mode,
    blt_tensor* logits_out,
    blt_arena* arena
) {
    BLT_REQUIRE(model != NULL && logits_out != NULL && arena != NULL &&
                batch != NULL,
        "forward_diffusion_infer: arguments cannot be NULL");

    diff_ctx c;
    diff_ctx_build(&c, model, byte_hidden_in, patch_in, patches, num_patches,
        batch, d0_mode, BLT_BDM_INFER, arena);

    size_t out_shape[2] = {c.S, c.V};
    blt_check_nd_fp32(logits_out, 2, out_shape,
        "forward_diffusion_infer: logits_out must be [N + n_block_rows, vocab_size] FP32");

    diff_layer_cache* caches;
    blt_tensor logits, final_states;
    forward_with_caches(model, &c, num_patches, &logits, &final_states,
        &caches, arena);
    memcpy(logits_out->data, logits.data, c.S * c.V * sizeof(float));
}
