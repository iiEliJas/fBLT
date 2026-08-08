#include "blt/models/attention.h"
#include "blt/core/backend.h"
#include "blt/core/tensor.h"
#include "blt/core/allocator.h"
#include "blt/ops/matmul.h"
#include "blt/ops/vecmath.h"
#include "blt/ops/elementwise.h"
#include "blt/ops/rope.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


//----------------------------------------------------------------
// Helper: Validate inputs and config for multihead attention

static void validate_attention_call(
    const blt_tensor* input, const blt_tensor* weight_qkv, const blt_tensor* weight_proj,
    const blt_tensor* output, const blt_attention_config* config, const blt_arena* arena,
    size_t* out_seq_len, size_t* out_head_dim
) {
    BLT_REQUIRE(config != NULL && arena != NULL, "blt_multihead_attention: config and arena cannot be NULL");
 
    size_t embed_dim = config->embed_dim;
    size_t num_heads = config->num_heads;
    BLT_REQUIRE(num_heads != 0 && embed_dim % num_heads == 0,
                "blt_multihead_attention: embed_dim must be divisible by num_heads");
 
    blt_check_nd_fp32(input, 2, (const size_t[]){0, embed_dim}, "blt_multihead_attention: input must be [seq_len, embed_dim] FP32");
    size_t seq_len = input->shape[0];
 
    blt_check_nd_fp32(weight_qkv, 2, (const size_t[]){embed_dim, 3 * embed_dim},
                       "blt_multihead_attention: weight_qkv must be [embed_dim, 3*embed_dim] FP32");
    blt_check_nd_fp32(weight_proj, 2, (const size_t[]){embed_dim, embed_dim},
                       "blt_multihead_attention: weight_proj must be [embed_dim, embed_dim] FP32");
    blt_check_nd_fp32(output, 2, (const size_t[]){seq_len, embed_dim},
                       "blt_multihead_attention: output must be [seq_len, embed_dim] FP32");
 
    size_t head_dim = (config->head_dim != 0) ? config->head_dim : (embed_dim / num_heads);
    BLT_REQUIRE(head_dim * num_heads == embed_dim,
                "blt_multihead_attention: head_dim * num_heads must equal embed_dim");
 
    if (config->use_rope) {
        BLT_REQUIRE(head_dim % 2 == 0, "blt_multihead_attention: RoPE requires an even head_dim");
        BLT_REQUIRE(config->rope_theta > 0.0f,
                    "blt_multihead_attention: rope_theta must be positive when use_rope is true");
    }
 
    *out_seq_len = seq_len;
    *out_head_dim = head_dim;
}



//----------------------------------------------------------------
// Helper: Compute softmax of a row in-place, with optional causal masking

static void causal_softmax_row_inplace(float* row, size_t seq_len, size_t row_idx, bool is_causal, float scale) {
    float max_val = -INFINITY;
 
    for (size_t col = 0; col < seq_len; ++col) {
        if (is_causal && col > row_idx) {
            row[col] = -INFINITY;
        } else {
            row[col] *= scale;
            if (isfinite(row[col]) && row[col] > max_val) {
                max_val = row[col];
            }
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
// Helper: Apply RoPE to all heads in the QKV tensor

static void apply_rope_to_all_heads(
    float* qkv_data, size_t seq_len, size_t embed_dim, size_t num_heads, size_t head_dim,
    const blt_tensor* rope_cos, const blt_tensor* rope_sin,
    float* q_head_buf, float* q_rot_buf, float* k_head_buf, float* k_rot_buf,
    blt_backend backend) {
    size_t qkv_stride = 3 * embed_dim;
 
    for (size_t head = 0; head < num_heads; ++head) {
        size_t q_offset = head * head_dim;
        size_t k_offset = embed_dim + head * head_dim;
 
        for (size_t i = 0; i < seq_len; ++i) {
            memcpy(q_head_buf + i * head_dim, qkv_data + i * qkv_stride + q_offset, head_dim * sizeof(float));
            memcpy(k_head_buf + i * head_dim, qkv_data + i * qkv_stride + k_offset, head_dim * sizeof(float));
        }
 
        blt_tensor q_head_t, q_rot_t, k_head_t, k_rot_t;
        blt_tensor_view_3d(&q_head_t, q_head_buf, seq_len, 1, head_dim, backend);
        blt_tensor_view_3d(&q_rot_t, q_rot_buf, seq_len, 1, head_dim, backend);
        blt_tensor_view_3d(&k_head_t, k_head_buf, seq_len, 1, head_dim, backend);
        blt_tensor_view_3d(&k_rot_t, k_rot_buf, seq_len, 1, head_dim, backend);
 
        blt_rope_apply(&q_head_t, rope_cos, rope_sin, &q_rot_t);
        blt_rope_apply(&k_head_t, rope_cos, rope_sin, &k_rot_t);
 
        for (size_t i = 0; i < seq_len; ++i) {
            memcpy(qkv_data + i * qkv_stride + q_offset, q_rot_buf + i * head_dim, head_dim * sizeof(float));
            memcpy(qkv_data + i * qkv_stride + k_offset, k_rot_buf + i * head_dim, head_dim * sizeof(float));
        }
    }
}



//----------------------------------------------------------------
// Helper: Process a single attention head

static void attention_head(const float* qkv_data, size_t head_idx, size_t seq_len, size_t embed_dim,
                            size_t head_dim, bool is_causal, float scale, float* scores_buf,
                            float* combined_out) {
    size_t qkv_stride = 3 * embed_dim;
    size_t q_offset = head_idx * head_dim;
    size_t k_offset = embed_dim + head_idx * head_dim;
    size_t v_offset = 2 * embed_dim + head_idx * head_dim;
 
    for (size_t i = 0; i < seq_len; ++i) {
        const float* q_i = qkv_data + i * qkv_stride + q_offset;
        float* scores_i = scores_buf + i * seq_len;
 
        for (size_t j = 0; j < seq_len; ++j) {
            const float* k_j = qkv_data + j * qkv_stride + k_offset;
            scores_i[j] = blt_vec_dot(q_i, k_j, head_dim);
        }
 
        causal_softmax_row_inplace(scores_i, seq_len, i, is_causal, scale);
    }
 
    for (size_t i = 0; i < seq_len; ++i) {
        float* out_i = combined_out + i * embed_dim + q_offset;
        const float* scores_i = scores_buf + i * seq_len;
 
        for (size_t d = 0; d < head_dim; ++d) {
            out_i[d] = 0.0f;
        }
 
        for (size_t j = 0; j < seq_len; ++j) {
            float weight = scores_i[j];
            if (weight == 0.0f) {
                continue;
            }
            const float* v_j = qkv_data + j * qkv_stride + v_offset;
            for (size_t d = 0; d < head_dim; ++d) {
                out_i[d] += weight * v_j[d];
            }
        }
    }
}



//----------------------------------------------------------------
// Main: Multihead attention function

void blt_multihead_attention(const blt_tensor* input, const blt_tensor* weight_qkv,
                              const blt_tensor* weight_proj, blt_tensor* output,
                              const blt_attention_config* config, blt_arena* arena) {
    size_t seq_len, head_dim;
    validate_attention_call(input, weight_qkv, weight_proj, output, config, arena, &seq_len, &head_dim);
 
    size_t embed_dim = config->embed_dim;
    size_t num_heads = config->num_heads;
    float scale = 1.0f / sqrtf((float)head_dim);
 
    // ---- Arena allocations ----
    size_t qkv_numel = seq_len * 3 * embed_dim;
    size_t combined_numel = seq_len * embed_dim;
    size_t scores_numel = seq_len * seq_len;
    size_t head_numel = seq_len * head_dim;
 
    float* qkv_data = (float*)blt_arena_alloc(arena, qkv_numel * sizeof(float), sizeof(float));
    float* combined_data = (float*)blt_arena_alloc(arena, combined_numel * sizeof(float), sizeof(float));
    float* scores_buf = (float*)blt_arena_alloc(arena, scores_numel * sizeof(float), sizeof(float));
    BLT_REQUIRE(qkv_data && combined_data && scores_buf,
                "blt_multihead_attention: failed to allocate temporary buffers from arena");
 
    memset(combined_data, 0, combined_numel * sizeof(float));
 
    // ---- Step 1: QKV projection ----
    blt_tensor qkv_tensor;
    blt_tensor_view_2d(&qkv_tensor, qkv_data, seq_len, 3 * embed_dim, input->backend);
    blt_matmul(input, weight_qkv, &qkv_tensor);
 
    // ---- Step 2: RoPE
    if (config->use_rope) {
        size_t half = head_dim / 2;
 
        const blt_tensor* rope_cos_t;
        const blt_tensor* rope_sin_t;
        blt_tensor computed_cos_t, computed_sin_t;   // only used in the fallback path
 
        bool have_cache = (config->rope_cos_cache != NULL && config->rope_sin_cache != NULL);
 
        if (have_cache) {
            blt_check_nd_fp32(config->rope_cos_cache, 2, (const size_t[]){seq_len, half},
                               "blt_multihead_attention: rope_cos_cache must be [seq_len, head_dim/2] FP32");
            blt_check_nd_fp32(config->rope_sin_cache, 2, (const size_t[]){seq_len, half},
                               "blt_multihead_attention: rope_sin_cache must be [seq_len, head_dim/2] FP32");
            rope_cos_t = config->rope_cos_cache;
            rope_sin_t = config->rope_sin_cache;
        } else {
            float* rope_cos_data = (float*)blt_arena_alloc(arena, seq_len * half * sizeof(float), sizeof(float));
            float* rope_sin_data = (float*)blt_arena_alloc(arena, seq_len * half * sizeof(float), sizeof(float));
            BLT_REQUIRE(rope_cos_data && rope_sin_data,
                        "blt_multihead_attention: failed to allocate RoPE table from arena");
 
            blt_tensor_view_2d(&computed_cos_t, rope_cos_data, seq_len, half, input->backend);
            blt_tensor_view_2d(&computed_sin_t, rope_sin_data, seq_len, half, input->backend);
 
            blt_rope_config rope_config = { .theta = config->rope_theta, .head_dim = head_dim };
            blt_rope_precompute(seq_len, &rope_config, &computed_cos_t, &computed_sin_t);
 
            rope_cos_t = &computed_cos_t;
            rope_sin_t = &computed_sin_t;
        }
 
        float* q_head_buf = (float*)blt_arena_alloc(arena, head_numel * sizeof(float), sizeof(float));
        float* k_head_buf = (float*)blt_arena_alloc(arena, head_numel * sizeof(float), sizeof(float));
        float* q_rot_buf = (float*)blt_arena_alloc(arena, head_numel * sizeof(float), sizeof(float));
        float* k_rot_buf = (float*)blt_arena_alloc(arena, head_numel * sizeof(float), sizeof(float));
        BLT_REQUIRE(q_head_buf && k_head_buf && q_rot_buf && k_rot_buf,
                    "blt_multihead_attention: failed to allocate RoPE temporary buffers from arena");
 
        apply_rope_to_all_heads(qkv_data, seq_len, embed_dim, num_heads, head_dim,
                                 rope_cos_t, rope_sin_t,
                                 q_head_buf, q_rot_buf, k_head_buf, k_rot_buf, input->backend);
    }
 
    // ---- Step 3: per-head attention ----
    for (size_t head = 0; head < num_heads; ++head) {
        attention_head(qkv_data, head, seq_len, embed_dim, head_dim,
                        config->is_causal, scale, scores_buf, combined_data);
    }
 
    // ---- Step 4: output projection ----
    blt_tensor combined_tensor;
    blt_tensor_view_2d(&combined_tensor, combined_data, seq_len, embed_dim, input->backend);
    blt_matmul(&combined_tensor, weight_proj, output);
 
    // Arena cleanup remains the caller's responsibility (blt_arena_reset).
}