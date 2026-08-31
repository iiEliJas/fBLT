#include "blt/ops/attn_core.h"
#include "blt/core/backend.h"

#include <cuda_runtime.h>
#include <math.h>

// Attention core kernels:
//   forward  - one block per query row; per-thread serial dot products in
//              CPU order, thread-0 softmax, and d-parallel AV accumulation
//              that keeps the j-ascending addition order per output element;
//   backward - grad_q kernel is one block per row (rows are independent);
//              grad_k/grad_v use a single block that walks query rows
//              sequentially so every grad_k_j / grad_v_j cell is written by
//              exactly one thread at a time, in CPU accumulation order.

#define ATTN_BLOCK 128

static int blt_cuda_launch_check(const char* what) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        BLT_FATAL("%s failed: %s", what, cudaGetErrorString(err));
    }
    return 1;
}

__global__ void blt_attn_core_fwd_kernel(const float* q, size_t q_stride,
                                         const float* k, size_t k_stride,
                                         const float* v, size_t v_stride,
                                         float* combined, size_t c_stride, size_t c_offset,
                                         float* scores, const float* mask,
                                         size_t nk, size_t head_dim,
                                         int is_causal, float scale) {
    const size_t i = blockIdx.x;
    const size_t tid = threadIdx.x;
    const float* q_i = q + i * q_stride;
    float* sc_i = scores + i * nk;

    for (size_t j = tid; j < nk; j += ATTN_BLOCK) {
        const float* k_j = k + j * k_stride;
        float sum = 0.0f;
        for (size_t d = 0; d < head_dim; ++d) {
            sum += q_i[d] * k_j[d];
        }
        sc_i[j] = sum;
    }
    __syncthreads();

    if (tid == 0) {
        float max_val = -INFINITY;
        for (size_t col = 0; col < nk; ++col) {
            float val = sc_i[col] * scale;
            if (mask != NULL) {
                val += mask[i * nk + col];
            } else if (is_causal && col > i) {
                val = -INFINITY;
            }
            sc_i[col] = val;
            if (isfinite(val) && val > max_val) {
                max_val = val;
            }
        }
        float sum = 0.0f;
        for (size_t col = 0; col < nk; ++col) {
            if (isfinite(sc_i[col])) {
                sc_i[col] = expf(sc_i[col] - max_val);
                sum += sc_i[col];
            } else {
                sc_i[col] = 0.0f;
            }
        }
        if (sum > 0.0f) {
            for (size_t col = 0; col < nk; ++col) {
                sc_i[col] /= sum;
            }
        }
    }
    __syncthreads();

    // d-parallel V accumulation; j runs ascending inside each lane
    // so addition order matches the CPU reference element-for-element.
    for (size_t d = tid; d < head_dim; d += ATTN_BLOCK) {
        float acc = 0.0f;
        for (size_t j = 0; j < nk; ++j) {
            const float w = sc_i[j];
            if (w != 0.0f) {
                acc += w * v[j * v_stride + d];
            }
        }
        combined[i * c_stride + c_offset + d] = acc;
    }
}

extern "C" void blt_attention_head_core_cuda(blt_backend backend, const blt_attention_head_args* a) {
    (void)backend;
    BLT_REQUIRE(backend == BLT_BACKEND_CUDA, "blt_attention_head_core: CUDA implementation called with non-CUDA backend");
    BLT_REQUIRE(a != NULL && a->q != NULL && a->k != NULL && a->v != NULL && a->combined != NULL,
                "blt_attention_head_core: buffers must not be NULL");
    BLT_REQUIRE(a->weights_out != NULL || a->scores_scratch != NULL,
                "blt_attention_head_core: need weights_out or scores_scratch");
    float* scores = a->weights_out ? a->weights_out : a->scores_scratch;

    blt_attn_core_fwd_kernel<<<(unsigned)a->nq, ATTN_BLOCK>>>(
        a->q, a->q_stride, a->k, a->k_stride, a->v, a->v_stride,
        a->combined, a->combined_stride, a->combined_col_offset,
        scores, a->mask, a->nk, a->head_dim,
        a->is_causal ? 1 : 0, a->scale);
    blt_cuda_launch_check("blt_attention_head_core");
}

// One block per query row: softmax jacobian gs_ij into scratch
// then j-sum for grad_q. gw is recomputed instead of staged.
__global__ void blt_attn_core_bwd_row_kernel(
    const float* q, size_t q_stride, const float* v, size_t v_stride,
    const float* weights, const float* grad_combined, size_t gc_stride, size_t gc_offset,
    float* scores_scratch, float* grad_q, size_t gq_stride,
    size_t nk, size_t head_dim, float scale) {
    const size_t i = blockIdx.x;
    const size_t tid = threadIdx.x;
    const float* go_i = grad_combined + i * gc_stride + gc_offset;
    const float* w_i = weights + i * nk;
    float* gs_i = scores_scratch + i * nk;

    __shared__ float partial[ATTN_BLOCK];

    // Pass 1: partial dots-of-dots for dot_i = sum_j gw_ij * w_ij.
    float local = 0.0f;
    for (size_t j = tid; j < nk; j += ATTN_BLOCK) {
        const float* v_j = v + j * v_stride;
        float gw = 0.0f;
        for (size_t d = 0; d < head_dim; ++d) {
            gw += go_i[d] * v_j[d];
        }
        local += gw * w_i[j];
    }
    partial[tid] = local;
    __syncthreads();
    if (tid == 0) {
        float dot = 0.0f;
        for (size_t t = 0; t < ATTN_BLOCK; ++t) {
            dot += partial[t];
        }
        partial[0] = dot;
    }
    __syncthreads();
    const float dot_i = partial[0];

    // Pass 2: store scale-folded gs_ij so downstream accumulations match
    // the CPU reference term-for-term, then grad_q via d-lanes.
    for (size_t j = tid; j < nk; j += ATTN_BLOCK) {
        const float* v_j = v + j * v_stride;
        float gw = 0.0f;
        for (size_t d = 0; d < head_dim; ++d) {
            gw += go_i[d] * v_j[d];
        }
        gs_i[j] = scale * (w_i[j] * (gw - dot_i));
    }
    __syncthreads();

    if (grad_q != NULL) {
        float* gq_i = grad_q + i * gq_stride;
        for (size_t d = tid; d < head_dim; d += ATTN_BLOCK) {
            float acc = 0.0f;
            for (size_t j = 0; j < nk; ++j) {
                const float gs = gs_i[j];
                if (gs != 0.0f) {
                    acc += gs * q[j * q_stride + d];
                }
            }
            gq_i[d] += acc;
        }
    }
}

// One thread per (j, d) output cell, walking query rows in ascending order
// so each cell's accumulation sequence is identical to the CPU reference
// term-for-term; partial sums live in registers and are flushed once.
// Generic fallback for head_dim > ATTN_BLOCK (memory pattern is strided).
__global__ void blt_attn_core_bwd_kv_cells_kernel(
    const float* q, size_t q_stride,
    const float* weights, const float* grad_combined, size_t gc_stride, size_t gc_offset,
    const float* scores_scratch,
    float* grad_k, size_t gk_stride,
    float* grad_v, size_t gv_stride,
    size_t nq, size_t nk, size_t head_dim) {
    const size_t total = nk * head_dim;
    for (size_t cell = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         cell < total;
         cell += (size_t)gridDim.x * blockDim.x) {
        const size_t j = cell / head_dim;
        const size_t d = cell % head_dim;

        float acc_k = 0.0f;
        float acc_v = 0.0f;
        for (size_t i = 0; i < nq; ++i) {
            if (grad_v != NULL) {
                const float w = weights[i * nk + j];
                if (w != 0.0f) {
                    acc_v += w * grad_combined[i * gc_stride + gc_offset + d];
                }
            }
            if (grad_k != NULL) {
                const float gs = scores_scratch[i * nk + j];
                if (gs != 0.0f) {
                    acc_k += gs * q[i * q_stride + d];
                }
            }
        }

        if (grad_k != NULL) grad_k[j * gk_stride + d] += acc_k;
        if (grad_v != NULL) grad_v[j * gv_stride + d] += acc_v;
    }
}

// Fast path: one BLOCK per kv row j, threads mapped to head-dim lanes so
// q/go reads are coalesced and the w_ij / gs_ij scalars broadcast to the
// whole block. Threads hold register accumulators for their lanes (requires
// head_dim <= blockDim, i.e. at most one lane per thread); query rows are
// walked in ascending order, preserving the CPU reference's per-cell term
// order.
__global__ void blt_attn_core_bwd_kv_kernel(
    const float* q, size_t q_stride,
    const float* weights, const float* grad_combined, size_t gc_stride, size_t gc_offset,
    const float* scores_scratch,
    float* grad_k, size_t gk_stride,
    float* grad_v, size_t gv_stride,
    size_t nq, size_t nk, size_t head_dim) {
    const size_t j = blockIdx.x;
    const size_t tid = threadIdx.x;

    float acc_k = 0.0f;
    float acc_v = 0.0f;
    const bool k_lane = (grad_k != NULL && tid < head_dim);
    const bool v_lane = (grad_v != NULL && tid < head_dim);
    const size_t d = tid;

    for (size_t i = 0; i < nq; ++i) {
        if (v_lane) {
            const float w = weights[i * nk + j];
            if (w != 0.0f) {
                acc_v += w * grad_combined[i * gc_stride + gc_offset + d];
            }
        }
        if (k_lane) {
            const float gs = scores_scratch[i * nk + j];
            if (gs != 0.0f) {
                acc_k += gs * q[i * q_stride + d];
            }
        }
    }

    if (k_lane) grad_k[j * gk_stride + d] += acc_k;
    if (v_lane) grad_v[j * gv_stride + d] += acc_v;
}

extern "C" void blt_attention_head_core_backward_cuda(blt_backend backend,
                                                      const blt_attention_head_bwd_args* a) {
    (void)backend;
    BLT_REQUIRE(backend == BLT_BACKEND_CUDA, "blt_attention_head_core_backward: CUDA implementation called with non-CUDA backend");
    BLT_REQUIRE(a != NULL && a->q != NULL && a->k != NULL && a->v != NULL &&
                a->weights != NULL && a->grad_combined != NULL,
                "blt_attention_head_core_backward: buffers must not be NULL");

    const bool need_gs = (a->grad_q != NULL || a->grad_k != NULL);
    BLT_REQUIRE(!need_gs || a->scores_scratch != NULL,
                "blt_attention_head_core_backward: scores_scratch required for grad_q/grad_k");

    if (need_gs) {
        blt_attn_core_bwd_row_kernel<<<(unsigned)a->nq, ATTN_BLOCK>>>(
            a->q, a->q_stride, a->v, a->v_stride,
            a->weights, a->grad_combined, a->gc_stride, a->gc_col_offset,
            a->scores_scratch, a->grad_q, a->gq_stride,
            a->nk, a->head_dim, a->scale);
        blt_cuda_launch_check("blt_attention_head_core_backward (row pass)");
    }

    if (a->grad_k != NULL || a->grad_v != NULL) {
        if (a->head_dim <= ATTN_BLOCK) {
            // Fast path: one block per kv row.
            blt_attn_core_bwd_kv_kernel<<<(unsigned)a->nk, ATTN_BLOCK>>>(
                a->q, a->q_stride,
                a->weights, a->grad_combined, a->gc_stride, a->gc_col_offset,
                a->scores_scratch,
                a->grad_k, a->gk_stride,
                a->grad_v, a->gv_stride,
                a->nq, a->nk, a->head_dim);
        } else {
            const size_t cells = a->nk * a->head_dim;
            unsigned blocks = (unsigned)((cells + ATTN_BLOCK - 1) / ATTN_BLOCK);
            if (blocks > 4096) blocks = 4096;
            if (blocks == 0) blocks = 1;
            blt_attn_core_bwd_kv_cells_kernel<<<blocks, ATTN_BLOCK>>>(
                a->q, a->q_stride,
                a->weights, a->grad_combined, a->gc_stride, a->gc_col_offset,
                a->scores_scratch,
                a->grad_k, a->gk_stride,
                a->grad_v, a->gv_stride,
                a->nq, a->nk, a->head_dim);
        }
        blt_cuda_launch_check("blt_attention_head_core_backward (kv pass)");
    }
}
