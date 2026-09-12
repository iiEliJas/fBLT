#ifndef BLT_OPS_ATTN_CORE_H
#define BLT_OPS_ATTN_CORE_H

#include <stdbool.h>
#include <stddef.h>
#include "blt/core/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

// Single-head scaled dot-product attention over raw buffers, so packed QKV
// layouts and cache slices both fit without copies. All pointers live in
// `backend` space; buffer access happens entirely inside the backend
// implementation.
//
// Forward, per query row i:
//   scores_ij = dot(q_i, k_j)                      (head_dim terms)
//   w_ij      = softmax(scale * scores_ij [+ mask_ij])
//   combined_i[col_offset + d] = sum_j w_ij * v_j[d]
//
// Masking: additive `mask` ([nq * nk], 0 / -INFINITY) wins over is_causal;
// with mask == NULL and is_causal, columns j > i are masked. Rows whose
// entries are all non-finite produce zeros (matches dense paths).
typedef struct {
    const float *q; // [nq, q_stride]
    size_t q_stride;
    const float *k; // [nk, k_stride]
    size_t k_stride;
    const float *v; // [nk, v_stride]
    size_t v_stride;
    float *combined; // [nq, combined_stride]; only head_dim cols at col_offset are written
    size_t combined_stride;
    size_t combined_col_offset;
    float *weights_out;    // optional [nq * nk]; receives post-softmax weights (backward cache)
    float *scores_scratch; // required when weights_out == NULL ([nq * nk] workspace)
    const float *mask;     // optional [nq * nk] additive mask
    size_t nq;
    size_t nk;
    size_t head_dim;
    bool is_causal;
    float scale;
} blt_attention_head_args;

void blt_attention_head_core(blt_backend backend, const blt_attention_head_args *args);

// Backward pass. Given saved post-softmax weights and grad_combined, accumulates:
//   grad_q   (scale * sum_j gs_ij * k_j)        gs = w * (gw - sum_j gw*w)
//   grad_k   (scale * sum_i gs_ij * q_i)
//   grad_v   (sum_i w_ij * grad_combined_i)
// Any of grad_q / grad_k / grad_v may be NULL to skip. Accumulates INTO the
// provided buffers (caller zeroes once per layer, matching multi-head fan-in).
typedef struct {
    const float *q; // [nq, q_stride]
    size_t q_stride;
    const float *k; // [nk, k_stride]
    size_t k_stride;
    const float *v; // [nk, v_stride]
    size_t v_stride;
    const float *weights;       // [nq * nk] post-softmax weights from forward
    const float *grad_combined; // [nq, gc_stride], slice at gc_col_offset
    size_t gc_stride;
    size_t gc_col_offset;
    float *scores_scratch; // [nq * nk] workspace; required when grad_q or grad_k is used
    float *grad_q;         // optional [nq, gq_stride]
    size_t gq_stride;
    float *grad_k; // optional [nk, gk_stride]
    size_t gk_stride;
    float *grad_v; // optional [nk, gv_stride]
    size_t gv_stride;
    size_t nq;
    size_t nk;
    size_t head_dim;
    float scale;
} blt_attention_head_bwd_args;

void blt_attention_head_core_backward(blt_backend backend, const blt_attention_head_bwd_args *args);

#ifdef __cplusplus
}
#endif

#endif // BLT_OPS_ATTN_CORE_H
