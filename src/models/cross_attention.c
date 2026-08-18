#include <math.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "blt/models/cross_attention.h"
#include "blt/core/backend.h"
#include "blt/core/tensor.h"
#include "blt/core/allocator.h"
#include "blt/ops/matmul.h"
#include "blt/ops/vecmath.h"
#include "blt/ops/elementwise.h"
#include "blt/ops/mask_builder.h"



//----------------------------------------------------------------
// Validation

static void validate_cross_attention_call(
    const blt_tensor* query_in, const blt_tensor* kv_in,
    const blt_cross_attention_weights* weights,
    const blt_cross_attention_config* config, const blt_arena* arena,
    size_t* out_num_patches, size_t* out_seq_len, size_t* out_head_dim
) {
    BLT_REQUIRE(config != NULL && arena != NULL, "blt_cross_attention: config and arena cannot be NULL");
    BLT_REQUIRE(config->mask_config != NULL,
                "blt_cross_attention: mask_config is required (block-diagonal patch mask)");

    size_t embed_dim = config->embed_dim;
    size_t num_heads = config->num_heads;
    BLT_REQUIRE(num_heads != 0 && embed_dim % num_heads == 0,
                "blt_cross_attention: embed_dim must be divisible by num_heads");

    blt_check_nd_fp32(query_in, 2, (const size_t[]){0, embed_dim},
                      "blt_cross_attention: query_in must be [num_patches, embed_dim] FP32");
    size_t num_patches = query_in->shape[0];

    blt_check_nd_fp32(kv_in, 2, (const size_t[]){0, embed_dim},
                      "blt_cross_attention: kv_in must be [seq_len, embed_dim] FP32");
    size_t seq_len = kv_in->shape[0];

    blt_check_nd_fp32(weights->weight_q, 2, (const size_t[]){embed_dim, embed_dim},
                      "blt_cross_attention: weight_q must be [embed_dim, embed_dim] FP32");
    blt_check_nd_fp32(weights->weight_k, 2, (const size_t[]){embed_dim, embed_dim},
                      "blt_cross_attention: weight_k must be [embed_dim, embed_dim] FP32");
    blt_check_nd_fp32(weights->weight_v, 2, (const size_t[]){embed_dim, embed_dim},
                      "blt_cross_attention: weight_v must be [embed_dim, embed_dim] FP32");
    blt_check_nd_fp32(weights->weight_proj, 2, (const size_t[]){embed_dim, embed_dim},
                      "blt_cross_attention: weight_proj must be [embed_dim, embed_dim] FP32");

    size_t head_dim = (config->head_dim != 0) ? config->head_dim : (embed_dim / num_heads);
    BLT_REQUIRE(head_dim * num_heads == embed_dim,
                "blt_cross_attention: head_dim * num_heads must equal embed_dim");

    *out_num_patches = num_patches;
    *out_seq_len = seq_len;
    *out_head_dim = head_dim;
}



//----------------------------------------------------------------
// Helper: Builds attention mask

static const float* build_mask(const blt_cross_attention_config* config,
                               size_t num_patches, size_t seq_len, blt_arena* arena) {
    BLT_REQUIRE(config->mask_config->seq_len_q == num_patches &&
                config->mask_config->seq_len_kv == seq_len,
                "blt_cross_attention: mask_config seq_len_q/seq_len_kv must match "
                "num_patches (query rows) and seq_len (kv rows)");

    blt_tensor mask_tensor = {0};
    blt_build_attention_mask(config->mask_config, &mask_tensor, arena);

    return (const float*)mask_tensor.data;
}



//----------------------------------------------------------------
// Helper: Compute softmax of a row in-place with masking
// (same helper as self-attention) [Note] Maybe add it to softmax.c

static void softmax_row_inplace(float* row, size_t seq_len, size_t row_idx, bool is_causal, const float* mask_row, float scale) {
    float max_val = -INFINITY;

    for (size_t col = 0; col < seq_len; ++col) {
        if (mask_row != NULL) {
            row[col] = row[col] * scale + mask_row[col];
        } else if (is_causal && col > row_idx) {
            row[col] = -INFINITY;
        } else {
            row[col] *= scale;
        }

        if (isfinite(row[col]) && row[col] > max_val) {
            max_val = row[col];
        }
    }

    float sum = 0.0f;
    for (size_t col = 0; col < seq_len; ++col) {
        if (isfinite(row[col])) {
            row[col] = expf(row[col] - max_val);
            sum += row[col];
        } else {
            row[col] = 0.0f;
        }
    }

    if (sum > 0.0f) {
        for (size_t col = 0; col < seq_len; ++col) {
            row[col] /= sum;
        }
    }
}



//----------------------------------------------------------------
// Helper: Process a single cross-attention head
//
// Queries come from the patch sequence (length num_patches), keys/values from
// the byte sequence (length seq_len). Head h occupies the column block
// [h*head_dim, (h+1)*head_dim) of each [rows, embed_dim] buffer

static void cross_attention_head(const float* q_data, const float* k_data, const float* v_data,
                                 size_t head_idx, size_t num_patches, size_t seq_len,
                                 size_t embed_dim, size_t head_dim,
                                 const float* mask, float scale,
                                 float* scores_buf, float* combined_out) {
    size_t head_offset = head_idx * head_dim;

    for (size_t i = 0; i < num_patches; ++i) {
        const float* q_i = q_data + i * embed_dim + head_offset;
        float* scores_i = scores_buf + i * seq_len;

        for (size_t j = 0; j < seq_len; ++j) {
            const float* k_j = k_data + j * embed_dim + head_offset;
            scores_i[j] = blt_vec_dot(q_i, k_j, head_dim);
        }

        const float* mask_row = mask + i * seq_len;
        softmax_row_inplace(scores_i, seq_len, i, false, mask_row, scale);
    }

    for (size_t i = 0; i < num_patches; ++i) {
        float* out_i = combined_out + i * embed_dim + head_offset;
        const float* scores_i = scores_buf + i * seq_len;

        for (size_t d = 0; d < head_dim; ++d) {
            out_i[d] = 0.0f;
        }

        for (size_t j = 0; j < seq_len; ++j) {
            float weight = scores_i[j];
            if (weight == 0.0f) {
                continue;
            }
            const float* v_j = v_data + j * embed_dim + head_offset;
            for (size_t d = 0; d < head_dim; ++d) {
                out_i[d] += weight * v_j[d];
            }
        }
    }
}



//----------------------------------------------------------------
// Cross-attention function
//
// Pipeline:
// 1. Projections:       Q = query_in * W_q    [num_patches, embed_dim]
//                       K = kv_in * W_k       [seq_len, embed_dim]
//                       V = kv_in * W_v       [seq_len, embed_dim]
// 2. Scaled Attention:  Scores = (Q * K^T) / sqrt(d_k)
//                       Attn   = Softmax(Scores + block-diagonal patch mask)
//                       Head   = Attn * V     [num_patches, embed_dim]
// 3. Output Projection: Output = Combined * W_proj
//
// No positional embeddings (paper-explicit) and no residual connection
// inside this op — the caller computes P_l = P_{l-1} + output.

void blt_cross_attention_forward(
    const blt_tensor* query_in,
    const blt_tensor* kv_in,
    const blt_cross_attention_weights* weights,
    blt_tensor* output,
    const blt_cross_attention_config* config,
    blt_arena* arena
) {
    size_t num_patches, seq_len, head_dim;
    validate_cross_attention_call(query_in, kv_in, weights, config, arena,
                                  &num_patches, &seq_len, &head_dim);

    size_t embed_dim = config->embed_dim;
    size_t num_heads = config->num_heads;

    blt_check_nd_fp32(output, 2, (const size_t[]){num_patches, embed_dim},
                      "blt_cross_attention: output must be [num_patches, embed_dim] FP32");

    // Scale factor for scaled dot-product attention: 1 / sqrt(head_dim)
    float scale = 1.0f / sqrtf((float)head_dim);

    // -----------------------------------------------------------------
    // STEP 0: Arena Memory Allocations
    // Workspace for the Q/K/V projections, concatenated multi-head
    // output, and the per-head [num_patches, seq_len] score matrix
    // -----------------------------------------------------------------
    size_t q_numel = num_patches * embed_dim;
    size_t kv_numel = seq_len * embed_dim;
    size_t scores_numel = num_patches * seq_len;

    float* q_data = (float*)blt_arena_alloc(arena, q_numel * sizeof(float), sizeof(float));
    float* k_data = (float*)blt_arena_alloc(arena, kv_numel * sizeof(float), sizeof(float));
    float* v_data = (float*)blt_arena_alloc(arena, kv_numel * sizeof(float), sizeof(float));
    float* combined_data = (float*)blt_arena_alloc(arena, q_numel * sizeof(float), sizeof(float));
    float* scores_buf = (float*)blt_arena_alloc(arena, scores_numel * sizeof(float), sizeof(float));
    BLT_REQUIRE(q_data && k_data && v_data && combined_data && scores_buf,
                "blt_cross_attention: failed to allocate temporary buffers from arena");

    memset(combined_data, 0, q_numel * sizeof(float));


    // -----------------------------------------------------------------
    // STEP 1: Linear Projections
    // Q and K/V come from different input tensors, so there is no shared
    // QKV projection — three separate matmuls against W_q, W_k, W_v.
    // -----------------------------------------------------------------
    blt_tensor q_tensor, k_tensor, v_tensor;
    blt_tensor_view_2d(&q_tensor, q_data, num_patches, embed_dim, query_in->backend);
    blt_tensor_view_2d(&k_tensor, k_data, seq_len, embed_dim, kv_in->backend);
    blt_tensor_view_2d(&v_tensor, v_data, seq_len, embed_dim, kv_in->backend);

    blt_matmul(query_in, weights->weight_q, &q_tensor);
    blt_matmul(kv_in, weights->weight_k, &k_tensor);
    blt_matmul(kv_in, weights->weight_v, &v_tensor);


    // No RoPE in cross attention

    const float* mask_data = build_mask(config, num_patches, seq_len, arena);


    // -----------------------------------------------------------------
    // STEP 2: Per-Head Scaled Dot-Product Attention
    // For each head h in [0, num_heads - 1]:
    //   a. Raw scores:        S_ij = Q_i * K_j        (patch i, byte j)
    //   b. Scale + mask:      S_ij / sqrt(d_k), -inf outside patch i
    //   c. Softmax over rows: A_ij
    //   d. Aggregate values:  Head_out_i = SUM_j (A_ij * V_j)
    // Writes head outputs into combined_data [num_patches, embed_dim]
    // -----------------------------------------------------------------
    for (size_t head = 0; head < num_heads; ++head) {
        cross_attention_head(q_data, k_data, v_data, head, num_patches, seq_len,
                             embed_dim, head_dim, mask_data, scale,
                             scores_buf, combined_data);
    }


    // -----------------------------------------------------------------
    // STEP 3: Output Linear Projection
    // Output = Combined * W_proj
    // Output tensor shape: [num_patches, embed_dim]  (pre-residual)
    // -----------------------------------------------------------------
    blt_tensor combined_tensor;
    blt_tensor_view_2d(&combined_tensor, combined_data, num_patches, embed_dim, query_in->backend);
    blt_matmul(&combined_tensor, weights->weight_proj, output);

    // Arena cleanup is the callers responsibility
}




//-----------------------------------------------------------------------------------------------
// CROSS-ATTENTION BACKWARD PASS

typedef struct {
    float* q_data;        // [num_patches, embed_dim]
    float* k_data;        // [seq_len, embed_dim]
    float* v_data;        // [seq_len, embed_dim]
    float* combined_data; // [num_patches, embed_dim]
    float* weights_all;   // [num_heads, num_patches, seq_len] softmax weights
} attn_bwd_forward_cache;



static void attn_bwd_recompute_forward(
    const blt_tensor* query_in, const blt_tensor* kv_in,
    const blt_cross_attention_weights* weights, const blt_cross_attention_config* config,
    blt_arena* arena, size_t num_patches, size_t seq_len, size_t embed_dim,
    size_t num_heads, size_t head_dim, float scale,
    attn_bwd_forward_cache* cache) {

    size_t q_numel = num_patches * embed_dim;
    size_t kv_numel = seq_len * embed_dim;
    size_t weights_numel = num_heads * num_patches * seq_len;

    cache->q_data = (float*)blt_arena_alloc(arena, q_numel * sizeof(float), sizeof(float));
    cache->k_data = (float*)blt_arena_alloc(arena, kv_numel * sizeof(float), sizeof(float));
    cache->v_data = (float*)blt_arena_alloc(arena, kv_numel * sizeof(float), sizeof(float));
    cache->combined_data = (float*)blt_arena_alloc(arena, q_numel * sizeof(float), sizeof(float));
    cache->weights_all = (float*)blt_arena_alloc(arena, weights_numel * sizeof(float), sizeof(float));
    BLT_REQUIRE(cache->q_data && cache->k_data && cache->v_data &&
                cache->combined_data && cache->weights_all,
                "blt_cross_attention_backward: failed to allocate forward-recompute buffers");
    memset(cache->combined_data, 0, q_numel * sizeof(float));

    blt_tensor q_tensor, k_tensor, v_tensor;
    blt_tensor_view_2d(&q_tensor, cache->q_data, num_patches, embed_dim, query_in->backend);
    blt_tensor_view_2d(&k_tensor, cache->k_data, seq_len, embed_dim, kv_in->backend);
    blt_tensor_view_2d(&v_tensor, cache->v_data, seq_len, embed_dim, kv_in->backend);

    blt_matmul(query_in, weights->weight_q, &q_tensor);
    blt_matmul(kv_in, weights->weight_k, &k_tensor);
    blt_matmul(kv_in, weights->weight_v, &v_tensor);

    const float* mask_data = build_mask(config, num_patches, seq_len, arena);

    // Per-head softmax attention weights + combined output
    // cross_attention_head writes the softmaxed weights into the scores buffer
    // so each heads slice of weights_all doubles as that buffer
    for (size_t head = 0; head < num_heads; head++) {
        float* W = cache->weights_all + head * num_patches * seq_len;
        cross_attention_head(cache->q_data, cache->k_data, cache->v_data, head,
                             num_patches, seq_len, embed_dim, head_dim,
                             mask_data, scale, W, cache->combined_data);
    }
}




void blt_cross_attention_backward(
    const blt_tensor* query_in, const blt_tensor* kv_in,
    const blt_cross_attention_weights* weights, const blt_tensor* grad_out,
    blt_tensor* grad_query_in, blt_tensor* grad_kv_in,
    blt_cross_attention_grad* grad_weights,
    const blt_cross_attention_config* config, blt_arena* arena) {

    BLT_REQUIRE(query_in != NULL && kv_in != NULL && weights != NULL && grad_out != NULL &&
                grad_query_in != NULL && grad_kv_in != NULL && grad_weights != NULL,
                "blt_cross_attention_backward: arguments must not be NULL");
    BLT_REQUIRE(config != NULL && arena != NULL,
                "blt_cross_attention_backward: config and arena must not be NULL");
    BLT_REQUIRE(config->mask_config != NULL,
                "blt_cross_attention_backward: mask_config is required (block-diagonal patch mask)");

    size_t embed_dim = config->embed_dim;
    size_t num_heads = config->num_heads;
    BLT_REQUIRE(num_heads != 0 && embed_dim % num_heads == 0,
                "blt_cross_attention_backward: embed_dim must be divisible by num_heads");
    size_t head_dim = (config->head_dim != 0) ? config->head_dim : (embed_dim / num_heads);
    BLT_REQUIRE(head_dim * num_heads == embed_dim,
                "blt_cross_attention_backward: head_dim * num_heads must equal embed_dim");

    blt_check_nd_fp32(query_in, 2, (const size_t[]){0, embed_dim},
                      "blt_cross_attention_backward: query_in must be [num_patches, embed_dim] FP32");
    size_t num_patches = query_in->shape[0];
    blt_check_nd_fp32(kv_in, 2, (const size_t[]){0, embed_dim},
                      "blt_cross_attention_backward: kv_in must be [seq_len, embed_dim] FP32");
    size_t seq_len = kv_in->shape[0];

    blt_check_nd_fp32(weights->weight_q, 2, (const size_t[]){embed_dim, embed_dim},
                      "blt_cross_attention_backward: weight_q must be [embed_dim, embed_dim] FP32");
    blt_check_nd_fp32(weights->weight_k, 2, (const size_t[]){embed_dim, embed_dim},
                      "blt_cross_attention_backward: weight_k must be [embed_dim, embed_dim] FP32");
    blt_check_nd_fp32(weights->weight_v, 2, (const size_t[]){embed_dim, embed_dim},
                      "blt_cross_attention_backward: weight_v must be [embed_dim, embed_dim] FP32");
    blt_check_nd_fp32(weights->weight_proj, 2, (const size_t[]){embed_dim, embed_dim},
                      "blt_cross_attention_backward: weight_proj must be [embed_dim, embed_dim] FP32");

    blt_check_nd_fp32(grad_out, 2, (const size_t[]){num_patches, embed_dim},
                      "blt_cross_attention_backward: grad_out must be [num_patches, embed_dim] FP32");
    blt_check_nd_fp32(grad_query_in, 2, (const size_t[]){num_patches, embed_dim},
                      "blt_cross_attention_backward: grad_query_in must be [num_patches, embed_dim] FP32");
    blt_check_nd_fp32(grad_kv_in, 2, (const size_t[]){seq_len, embed_dim},
                      "blt_cross_attention_backward: grad_kv_in must be [seq_len, embed_dim] FP32");
    blt_check_nd_fp32(&grad_weights->grad_weight_q, 2, (const size_t[]){embed_dim, embed_dim},
                      "blt_cross_attention_backward: grad_weight_q must be [embed_dim, embed_dim] FP32");
    blt_check_nd_fp32(&grad_weights->grad_weight_k, 2, (const size_t[]){embed_dim, embed_dim},
                      "blt_cross_attention_backward: grad_weight_k must be [embed_dim, embed_dim] FP32");
    blt_check_nd_fp32(&grad_weights->grad_weight_v, 2, (const size_t[]){embed_dim, embed_dim},
                      "blt_cross_attention_backward: grad_weight_v must be [embed_dim, embed_dim] FP32");
    blt_check_nd_fp32(&grad_weights->grad_weight_proj, 2, (const size_t[]){embed_dim, embed_dim},
                      "blt_cross_attention_backward: grad_weight_proj must be [embed_dim, embed_dim] FP32");

    float scale = 1.0f / sqrtf((float)head_dim);

    // ---- Recompute forward intermediates ----
    attn_bwd_forward_cache cache = {0};
    attn_bwd_recompute_forward(query_in, kv_in, weights, config, arena,
                                num_patches, seq_len, embed_dim, num_heads, head_dim,
                                scale, &cache);

    // ---- Step 1: backward through output projection: output = combined @ weight_proj ----
    blt_tensor combined_tensor;
    blt_tensor_view_2d(&combined_tensor, cache.combined_data, num_patches, embed_dim, query_in->backend);

    float* grad_combined_data = (float*)blt_arena_alloc(arena, num_patches * embed_dim * sizeof(float), sizeof(float));
    BLT_REQUIRE(grad_combined_data, "blt_cross_attention_backward: failed to allocate grad_combined");
    blt_tensor grad_combined_tensor;
    blt_tensor_view_2d(&grad_combined_tensor, grad_combined_data, num_patches, embed_dim, query_in->backend);

    blt_matmul_backward(&combined_tensor, weights->weight_proj, grad_out,
                        &grad_combined_tensor, &grad_weights->grad_weight_proj);

    // ---- Step 2-4: per head, backward through weighted-V sum, softmax, and QK^T ----
    float* grad_q_data = (float*)blt_arena_alloc(arena, num_patches * embed_dim * sizeof(float), sizeof(float));
    float* grad_k_data = (float*)blt_arena_alloc(arena, seq_len * embed_dim * sizeof(float), sizeof(float));
    float* grad_v_data = (float*)blt_arena_alloc(arena, seq_len * embed_dim * sizeof(float), sizeof(float));
    BLT_REQUIRE(grad_q_data && grad_k_data && grad_v_data,
                "blt_cross_attention_backward: failed to allocate grad_q/k/v");
    memset(grad_q_data, 0, num_patches * embed_dim * sizeof(float));
    memset(grad_k_data, 0, seq_len * embed_dim * sizeof(float));
    memset(grad_v_data, 0, seq_len * embed_dim * sizeof(float));

    float* grad_w_buf = (float*)blt_arena_alloc(arena, num_patches * seq_len * sizeof(float), sizeof(float));
    float* grad_scores_buf = (float*)blt_arena_alloc(arena, num_patches * seq_len * sizeof(float), sizeof(float));
    BLT_REQUIRE(grad_w_buf && grad_scores_buf,
                "blt_cross_attention_backward: failed to allocate score grad buffers");

    for (size_t head = 0; head < num_heads; head++) {
        size_t head_offset = head * head_dim;
        const float* W = cache.weights_all + head * num_patches * seq_len;

        // grad_v_j[d] = sum_i W[i][j] * grad_out_head[i][d]
        // grad_W[i][j] = sum_d grad_out_head[i][d] * v_j[d]
        memset(grad_w_buf, 0, num_patches * seq_len * sizeof(float));
        for (size_t i = 0; i < num_patches; i++) {
            const float* go_i = grad_combined_data + i * embed_dim + head_offset;
            const float* w_i = W + i * seq_len;
            float* gw_i = grad_w_buf + i * seq_len;
            for (size_t j = 0; j < seq_len; j++) {
                float wgt = w_i[j];
                const float* v_j = cache.v_data + j * embed_dim + head_offset;
                float dot = 0.0f;
                for (size_t d = 0; d < head_dim; d++) {
                    dot += go_i[d] * v_j[d];
                }
                gw_i[j] = dot;

                if (wgt != 0.0f) {
                    float* gv_j = grad_v_data + j * embed_dim + head_offset;
                    for (size_t d = 0; d < head_dim; d++) {
                        gv_j[d] += wgt * go_i[d];
                    }
                }
            }
        }

        // Softmax backward, per row
        for (size_t i = 0; i < num_patches; i++) {
            const float* w_i = W + i * seq_len;
            const float* gw_i = grad_w_buf + i * seq_len;
            float* gs_i = grad_scores_buf + i * seq_len;
            float dot = 0.0f;
            for (size_t j = 0; j < seq_len; j++) {
                dot += gw_i[j] * w_i[j];
            }
            for (size_t j = 0; j < seq_len; j++) {
                gs_i[j] = w_i[j] * (gw_i[j] - dot);
            }
        }

        // Backward through scores = scale * Q @ K^T
        // grad_q_i[d] += scale * sum_j grad_scores[i][j] * k_j[d]
        // grad_k_j[d] += scale * sum_i grad_scores[i][j] * q_i[d]
        for (size_t i = 0; i < num_patches; i++) {
            const float* gs_i = grad_scores_buf + i * seq_len;
            float* gq_i = grad_q_data + i * embed_dim + head_offset;
            const float* q_i = cache.q_data + i * embed_dim + head_offset;
            for (size_t j = 0; j < seq_len; j++) {
                float gscore = gs_i[j];
                if (gscore == 0.0f) {
                    continue;
                }
                const float* k_j = cache.k_data + j * embed_dim + head_offset;
                float* gk_j = grad_k_data + j * embed_dim + head_offset;
                for (size_t d = 0; d < head_dim; d++) {
                    gq_i[d] += scale * gscore * k_j[d];
                    gk_j[d] += scale * gscore * q_i[d];
                }
            }
        }
    }

    // ---- Step 5: backward through the input projections ----
    // Q = query_in @ W_q  =>  grad_query_in, grad_weight_q
    blt_tensor grad_q_tensor;
    blt_tensor_view_2d(&grad_q_tensor, grad_q_data, num_patches, embed_dim, query_in->backend);
    blt_matmul_backward(query_in, weights->weight_q, &grad_q_tensor,
                        grad_query_in, &grad_weights->grad_weight_q);

    // K = kv_in @ W_k and V = kv_in @ W_v share the same input, so grad_kv_in
    // is the sum of both matmul input gradients
    blt_tensor grad_k_tensor, grad_v_tensor;
    blt_tensor_view_2d(&grad_k_tensor, grad_k_data, seq_len, embed_dim, kv_in->backend);
    blt_tensor_view_2d(&grad_v_tensor, grad_v_data, seq_len, embed_dim, kv_in->backend);

    float* grad_kv_k_data = (float*)blt_arena_alloc(arena, seq_len * embed_dim * sizeof(float), sizeof(float));
    float* grad_kv_v_data = (float*)blt_arena_alloc(arena, seq_len * embed_dim * sizeof(float), sizeof(float));
    BLT_REQUIRE(grad_kv_k_data && grad_kv_v_data,
                "blt_cross_attention_backward: failed to allocate grad_kv temporaries");
    blt_tensor grad_kv_k_tensor, grad_kv_v_tensor;
    blt_tensor_view_2d(&grad_kv_k_tensor, grad_kv_k_data, seq_len, embed_dim, kv_in->backend);
    blt_tensor_view_2d(&grad_kv_v_tensor, grad_kv_v_data, seq_len, embed_dim, kv_in->backend);

    blt_matmul_backward(kv_in, weights->weight_k, &grad_k_tensor,
                        &grad_kv_k_tensor, &grad_weights->grad_weight_k);
    blt_matmul_backward(kv_in, weights->weight_v, &grad_v_tensor,
                        &grad_kv_v_tensor, &grad_weights->grad_weight_v);

    blt_add(&grad_kv_k_tensor, &grad_kv_v_tensor, grad_kv_in);

    // Arena cleanup is callers responsibility
}