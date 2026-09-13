#include "ops/attn_core.h"
#include "core/backend.h"
#include "ops/vecmath.h"

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

// CPU reference for the attention core. Loop structure mirrors the original
// dense implementations in src/models/attention.c (bit-identical numerics):
// per-row dot products, masked stable softmax, zero-weight-skipping V
// accumulation; backward computes gw via go.v dots, softmax jacobian, and
// strided gq/gk/gv accumulation in the same order.

void blt_attention_head_core_cpu(blt_backend backend, const blt_attention_head_args *a) {
    BLT_REQUIRE(backend == BLT_BACKEND_CPU, "blt_attention_head_core: CPU implementation called with non-CPU backend");
    BLT_REQUIRE(a != NULL && a->q != NULL && a->k != NULL && a->v != NULL && a->combined != NULL,
                "blt_attention_head_core: buffers must not be NULL");
    BLT_REQUIRE(a->weights_out != NULL || a->scores_scratch != NULL,
                "blt_attention_head_core: need weights_out or scores_scratch");

    float *scores = a->weights_out ? a->weights_out : a->scores_scratch;

    for (size_t i = 0; i < a->nq; ++i) {
        const float *q_i = a->q + i * a->q_stride;
        float *scores_i = scores + i * a->nk;

        for (size_t j = 0; j < a->nk; ++j) {
            const float *k_j = a->k + j * a->k_stride;
            scores_i[j] = blt_vec_dot(BLT_BACKEND_CPU, q_i, k_j, a->head_dim);
        }

        const float *mask_row = a->mask ? (a->mask + i * a->nk) : NULL;
        blt_softmax_masked_row_inplace(BLT_BACKEND_CPU, scores_i, a->nk, i, a->is_causal, mask_row, a->scale);
    }

    for (size_t i = 0; i < a->nq; ++i) {
        float *out_i = a->combined + i * a->combined_stride + a->combined_col_offset;
        const float *scores_i = scores + i * a->nk;

        for (size_t d = 0; d < a->head_dim; ++d) {
            out_i[d] = 0.0f;
        }

        for (size_t j = 0; j < a->nk; ++j) {
            const float weight = scores_i[j];
            if (weight == 0.0f) {
                continue;
            }
            const float *v_j = a->v + j * a->v_stride;
            for (size_t d = 0; d < a->head_dim; ++d) {
                out_i[d] += weight * v_j[d];
            }
        }
    }
}

void blt_attention_head_core_backward_cpu(blt_backend backend, const blt_attention_head_bwd_args *a) {
    BLT_REQUIRE(backend == BLT_BACKEND_CPU,
                "blt_attention_head_core_backward: CPU implementation called with non-CPU backend");
    BLT_REQUIRE(a != NULL && a->q != NULL && a->k != NULL && a->v != NULL && a->weights != NULL &&
                    a->grad_combined != NULL,
                "blt_attention_head_core_backward: buffers must not be NULL");

    const size_t nq = a->nq;
    const size_t nk = a->nk;
    const size_t hd = a->head_dim;

    // grad_v_j += w_ij * go_i ; grad_w_ij = dot(go_i, v_j)
    if (a->grad_v != NULL) {
        for (size_t i = 0; i < nq; i++) {
            const float *go_i = a->grad_combined + i * a->gc_stride + a->gc_col_offset;
            const float *w_i = a->weights + i * nk;
            for (size_t j = 0; j < nk; j++) {
                const float wgt = w_i[j];
                if (wgt == 0.0f) {
                    continue;
                }
                float *gv_j = a->grad_v + j * a->gv_stride;
                for (size_t d = 0; d < hd; d++) {
                    gv_j[d] += wgt * go_i[d];
                }
            }
        }
    }

    // Softmax backward, per row: gs_ij = w_ij * (gw_ij - dot_i).
    // grad_w_ij = dot(go_i, v_j)
    if (a->grad_q != NULL || a->grad_k != NULL) {
        for (size_t i = 0; i < nq; i++) {
            const float *go_i = a->grad_combined + i * a->gc_stride + a->gc_col_offset;
            const float *w_i = a->weights + i * nk;

            float dot = 0.0f;
            for (size_t j = 0; j < nk; j++) {
                const float *v_j = a->v + j * a->v_stride;
                float gw = 0.0f;
                for (size_t d = 0; d < hd; d++) {
                    gw += go_i[d] * v_j[d];
                }
                dot += gw * w_i[j];
            }

            const float *q_i = a->q + i * a->q_stride;
            float *gq_i = a->grad_q ? a->grad_q + i * a->gq_stride : NULL;

            for (size_t j = 0; j < nk; j++) {
                const float *v_j = a->v + j * a->v_stride;
                float gw = 0.0f;
                for (size_t d = 0; d < hd; d++) {
                    gw += go_i[d] * v_j[d];
                }
                const float gs = w_i[j] * (gw - dot);

                if (gs == 0.0f) {
                    continue;
                }
                if (gq_i != NULL) {
                    const float *k_j = a->k + j * a->k_stride;
                    for (size_t d = 0; d < hd; d++) {
                        gq_i[d] += a->scale * gs * k_j[d];
                    }
                }
                if (a->grad_k != NULL) {
                    float *gk_j = a->grad_k + j * a->gk_stride;
                    for (size_t d = 0; d < hd; d++) {
                        gk_j[d] += a->scale * gs * q_i[d];
                    }
                }
            }
        }
    }
}
