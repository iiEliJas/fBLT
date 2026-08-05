#include "blt/models/attention.h"
#include "blt/core/backend.h"
#include "blt/core/tensor.h"
#include "blt/core/allocator.h"
#include "blt/ops/matmul.h"
#include "blt/ops/vecmath.h"
#include "blt/ops/elementwise.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>


/*
 * Helper: Initialize a 2D tensor struct with row-major strides.
 * Sets up all fields and computes strides for efficient memory access.
 */
static void init_tensor_2d(blt_tensor* t, void* data, size_t rows, size_t cols,
                            size_t numel, blt_backend backend) {
    t->data = data;
    t->shape[0] = rows;
    t->shape[1] = cols;
    t->ndim = 2;
    t->numel = numel;
    t->dtype = BLT_DTYPE_FP32;
    t->backend = backend;
    t->is_view = false;
    blt_tensor_compute_row_major_strides(t->shape, 2, t->strides);
}

/*
 * Helper: Compute softmax of a row in-place, with optional causal masking.
 * If is_causal is true, elements at positions > row_idx are set to -inf before softmax.
 * Row data is assumed to be contiguous.
 */
static void softmax_row(float* row, size_t seq_len, size_t row_idx, bool is_causal, float scale) {
    float max_val = -INFINITY;

    /* Apply causal mask and find max for numerical stability */
    for (size_t col = 0; col < seq_len; ++col) {
        if (is_causal && col > row_idx) {
            row[col] = -INFINITY;
        } else {
            row[col] *= scale;  /* Apply scale factor 1/sqrt(d_k) here */
            if (isfinite(row[col]) && row[col] > max_val) {
                max_val = row[col];
            }
        }
    }

    /* Compute exp and sum for numerical stability */
    float sum = 0.0f;
    for (size_t col = 0; col < seq_len; ++col) {
        if (isfinite(row[col])) {
            row[col] = expf(row[col] - max_val);
            sum += row[col];
        } else {
            row[col] = 0.0f;
        }
    }

    /* Normalize */
    if (sum > 0.0f) {
        for (size_t col = 0; col < seq_len; ++col) {
            row[col] /= sum;
        }
    }
}

/*
 * Process a single attention head.
 * Computes attention weights and context for head `head_idx` and writes
 * the output directly into the appropriate slice of `combined_out`.
 */
static void attention_head(const float* qkv_data, size_t head_idx, size_t seq_len, size_t embed_dim,
                            size_t head_dim, bool is_causal, float scale, float* scores_buf,
                            float* combined_out) {
    size_t qkv_stride = 3 * embed_dim;
    size_t q_offset = head_idx * head_dim;
    size_t k_offset = embed_dim + head_idx * head_dim;
    size_t v_offset = 2 * embed_dim + head_idx * head_dim;

    /* Compute QK^T and apply softmax */
    for (size_t i = 0; i < seq_len; ++i) {
        const float* q_i = qkv_data + i * qkv_stride + q_offset;
        float* scores_i = scores_buf + i * seq_len;

        /* Compute Q[i] @ K^T */
        for (size_t j = 0; j < seq_len; ++j) {
            const float* k_j = qkv_data + j * qkv_stride + k_offset;
            scores_i[j] = blt_vec_dot(q_i, k_j, head_dim);
        }

        /* Apply softmax with optional causal mask */
        softmax_row(scores_i, seq_len, i, is_causal, scale);
    }

    /* Compute attention output: scores @ V */
    for (size_t i = 0; i < seq_len; ++i) {
        float* out_i = combined_out + i * embed_dim + q_offset;
        const float* scores_i = scores_buf + i * seq_len;

        /* Initialize output for this head */
        for (size_t d = 0; d < head_dim; ++d) {
            out_i[d] = 0.0f;
        }

        /* Weighted sum over value vectors */
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

void blt_multihead_attention(const blt_tensor* input, const blt_tensor* weight_qkv,
                              const blt_tensor* weight_proj, blt_tensor* output,
                              const blt_attention_config* config, blt_arena* arena) {
    /* Validation */
    if (input == NULL || weight_qkv == NULL || weight_proj == NULL || output == NULL || 
        config == NULL || arena == NULL) {
        BLT_FATAL("blt_multihead_attention: null pointer argument");
    }

    if (input->dtype != BLT_DTYPE_FP32 || weight_qkv->dtype != BLT_DTYPE_FP32 ||
        weight_proj->dtype != BLT_DTYPE_FP32 || output->dtype != BLT_DTYPE_FP32) {
        BLT_FATAL("blt_multihead_attention: all tensors must be FP32");
    }

    if (input->ndim != 2 || weight_qkv->ndim != 2 || weight_proj->ndim != 2 || output->ndim != 2) {
        BLT_FATAL("blt_multihead_attention: all tensors must be 2D");
    }

    size_t seq_len = input->shape[0];
    size_t embed_dim = config->embed_dim;
    size_t num_heads = config->num_heads;

    /* Validate dimensions */
    if (input->shape[1] != embed_dim) {
        BLT_FATAL("blt_multihead_attention: input width must equal embed_dim");
    }
    if (weight_qkv->shape[0] != embed_dim || weight_qkv->shape[1] != 3 * embed_dim) {
        BLT_FATAL("blt_multihead_attention: weight_qkv shape must be [embed_dim, 3*embed_dim]");
    }
    if (weight_proj->shape[0] != embed_dim || weight_proj->shape[1] != embed_dim) {
        BLT_FATAL("blt_multihead_attention: weight_proj shape must be [embed_dim, embed_dim]");
    }
    if (output->shape[0] != seq_len || output->shape[1] != embed_dim) {
        BLT_FATAL("blt_multihead_attention: output shape must be [seq_len, embed_dim]");
    }

    /* Validate config */
    if (num_heads == 0 || embed_dim % num_heads != 0) {
        BLT_FATAL("blt_multihead_attention: embed_dim must be divisible by num_heads");
    }

    size_t head_dim = embed_dim / num_heads;
    float scale = 1.0f / sqrtf((float)head_dim);

    /* Allocate temporary buffers from arena */
    size_t qkv_numel = seq_len * 3 * embed_dim;
    size_t combined_numel = seq_len * embed_dim;
    size_t scores_numel = seq_len * seq_len;

    float* qkv_data = (float*)blt_arena_alloc(arena, qkv_numel * sizeof(float), sizeof(float));
    float* combined_data = (float*)blt_arena_alloc(arena, combined_numel * sizeof(float), sizeof(float));
    float* scores_buf = (float*)blt_arena_alloc(arena, scores_numel * sizeof(float), sizeof(float));

    if (!qkv_data || !combined_data || !scores_buf) {
        BLT_FATAL("blt_multihead_attention: failed to allocate temporary buffers from arena");
    }

    memset(combined_data, 0, combined_numel * sizeof(float));

    /* Step 1: QKV projection: input @ weight_qkv -> [seq_len, 3*embed_dim] */
    blt_tensor qkv_tensor;
    init_tensor_2d(&qkv_tensor, (void*)qkv_data, seq_len, 3 * embed_dim,
                    qkv_numel, input->backend);

    blt_matmul(input, weight_qkv, &qkv_tensor);

    /* Step 2: Process each attention head */
    for (size_t head = 0; head < num_heads; ++head) {
        attention_head(qkv_data, head, seq_len, embed_dim, head_dim,
                       config->is_causal, scale, scores_buf, combined_data);
    }

    /* Step 3: Final projection: combined @ weight_proj -> output */
    blt_tensor combined_tensor;
    init_tensor_2d(&combined_tensor, (void*)combined_data, seq_len, embed_dim,
                    combined_numel, input->backend);

    blt_matmul(&combined_tensor, weight_proj, output);
    
    /* Arena cleanup is handled by the caller via blt_arena_reset */
}