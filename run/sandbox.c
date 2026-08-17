#include <math.h>
#include <stdio.h>
#include <string.h>

#include "blt/core/allocator.h"
#include "blt/core/tensor.h"
#include "blt/ops/matmul.h"
#include "blt/ops/vecmath.h"
#include "blt/ops/mask_builder.h"

// reproduces the Q/K projection + masked softmax steps of cross-attention
// prints the results [num_patches x seq_len] attention-weight matrix for every head
//
// Setup: 2 patches, 4 kv (byte) positions, split into two patch-groups of 2 bytes each with a block-diagonal mask
// W_q = W_k = identity, so Q = query_in, K = kv_in.
//
// Expected: for every head:
// weights[0][2] == weights[0][3] == 0
// weights[1][0] == weights[1][1] == 0

static void fill_identity(float* data, size_t dim) {
    for (size_t r = 0; r < dim; ++r) {
        for (size_t c = 0; c < dim; ++c) {
            data[r * dim + c] = (r == c) ? 1.0f : 0.0f;
        }
    }
}



static void print_weights(const char* label, const float* w, size_t num_patches, size_t seq_len) {
    printf("%s:\n", label);
    for (size_t i = 0; i < num_patches; ++i) {
        printf("  patch %zu: [", i);
        for (size_t j = 0; j < seq_len; ++j) {
            printf("%8.5f%s", w[i * seq_len + j], (j + 1 < seq_len) ? ", " : "");
        }
        printf("]\n");
    }
}



int main(void) {
    blt_arena* arena = blt_arena_create(65536, BLT_BACKEND_CPU);
    if (!arena) {
        fprintf(stderr, "arena creation failed\n");
        return 1;
    }

    const size_t num_patches = 2;
    const size_t seq_len = 4;
    const size_t embed_dim = 4;
    const size_t num_heads = 2;
    const size_t head_dim = embed_dim / num_heads;
    const float scale = 1.0f / sqrtf((float)head_dim);

    size_t query_shape[2] = {num_patches, embed_dim};
    size_t kv_shape[2] = {seq_len, embed_dim};
    size_t weight_shape[2] = {embed_dim, embed_dim};

    blt_tensor query_in = blt_tensor_create(arena, query_shape, 2, BLT_DTYPE_FP32);
    blt_tensor kv_in = blt_tensor_create(arena, kv_shape, 2, BLT_DTYPE_FP32);
    blt_tensor weight_q = blt_tensor_create(arena, weight_shape, 2, BLT_DTYPE_FP32);
    blt_tensor weight_k = blt_tensor_create(arena, weight_shape, 2, BLT_DTYPE_FP32);
    blt_tensor q_tensor = blt_tensor_create(arena, query_shape, 2, BLT_DTYPE_FP32);
    blt_tensor k_tensor = blt_tensor_create(arena, kv_shape, 2, BLT_DTYPE_FP32);

    float* query_data = (float*)query_in.data;
    query_data[0] = 1.0f; query_data[1] = 0.0f; query_data[2] = 1.0f; query_data[3] = 0.0f;
    query_data[4] = 0.0f; query_data[5] = 1.0f; query_data[6] = 0.0f; query_data[7] = 1.0f;

    float* kv_data = (float*)kv_in.data;
    kv_data[0]  = 1.0f; kv_data[1]  = 0.0f; kv_data[2]  = 1.0f; kv_data[3]  = 0.0f;
    kv_data[4]  = 0.0f; kv_data[5]  = 1.0f; kv_data[6]  = 0.0f; kv_data[7]  = 1.0f;
    kv_data[8]  = 1.0f; kv_data[9]  = 0.0f; kv_data[10] = 1.0f; kv_data[11] = 0.0f;
    kv_data[12] = 0.0f; kv_data[13] = 1.0f; kv_data[14] = 0.0f; kv_data[15] = 1.0f;

    fill_identity((float*)weight_q.data, embed_dim);
    fill_identity((float*)weight_k.data, embed_dim);

    blt_matmul(&query_in, &weight_q, &q_tensor);
    blt_matmul(&kv_in, &weight_k, &k_tensor);

    // Block-diagonal patch mask
    // query i only attends to kv positions in its group
    size_t query_group_ids[2] = {0, 1};
    size_t kv_group_ids[4] = {0, 0, 1, 1};

    blt_mask_config mask_cfg = {0};
    mask_cfg.seq_len_q = num_patches;
    mask_cfg.seq_len_kv = seq_len;
    mask_cfg.sliding_window = 0;
    mask_cfg.doc_boundaries = NULL;
    mask_cfg.num_docs = 0;
    mask_cfg.query_group_ids = query_group_ids;
    mask_cfg.kv_group_ids = kv_group_ids;
    mask_cfg.bidirectional_within_group = true;
    mask_cfg.is_causal = false;

    blt_tensor mask_tensor = {0};
    blt_build_attention_mask(&mask_cfg, &mask_tensor, arena);
    const float* mask_data = (const float*)mask_tensor.data;

    printf("Mask (0 = allowed, -inf = masked):\n");
    print_weights("mask", mask_data, num_patches, seq_len);
    printf("\n");

    const float* q_data = (const float*)q_tensor.data;
    const float* k_data = (const float*)k_tensor.data;
    float* weights_buf = (float*)blt_arena_alloc(arena, num_patches * seq_len * sizeof(float), sizeof(float));

    for (size_t head = 0; head < num_heads; ++head) {
        size_t head_offset = head * head_dim;

        for (size_t i = 0; i < num_patches; ++i) {
            const float* q_i = q_data + i * embed_dim + head_offset;
            const float* mask_row = mask_data + i * seq_len;
            float* w_i = weights_buf + i * seq_len;

            float max_val = -INFINITY;
            for (size_t j = 0; j < seq_len; ++j) {
                const float* k_j = k_data + j * embed_dim + head_offset;
                float score = blt_vec_dot(q_i, k_j, head_dim) * scale + mask_row[j];
                w_i[j] = score;
                if (isfinite(score) && score > max_val) {
                    max_val = score;
                }
            }

            float sum = 0.0f;
            for (size_t j = 0; j < seq_len; ++j) {
                if (isfinite(w_i[j])) {
                    w_i[j] = expf(w_i[j] - max_val);
                    sum += w_i[j];
                } else {
                    w_i[j] = 0.0f;
                }
            }
            if (sum > 0.0f) {
                for (size_t j = 0; j < seq_len; ++j) {
                    w_i[j] /= sum;
                }
            }
        }

        char label[64];
        snprintf(label, sizeof(label), "Head %zu attention weights", head);
        print_weights(label, weights_buf, num_patches, seq_len);
        printf("\n");
    }

    blt_arena_destroy(arena);
    return 0;
}