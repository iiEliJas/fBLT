#include "blt/core/backend.h"
#include "blt/core/cuda_shim.h"
#include "blt/core/allocator.h"
#include "blt/ops/softmax.h"
#include "blt/ops/cross_entropy.h"
#include "blt/ops/rope.h"
#include "blt/ops/rmsnorm.h"
#include "blt/ops/layernorm.h"

#include <cuda_runtime.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

// Row-wise reductions run one CUDA thread per row with the CPU loop order,
// which keeps results within a few ulps of the CPU reference (and often
// bit-identical). Cross-row accumulations (cross-entropy mean, rmsnorm
// grad_weight) reduce in fixed sequential order instead of atomics so runs
// stay reproducible.

#define BLT_RMSNORM_EPS 1e-6f

static int blt_cuda_launch_check(const char* what) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        BLT_FATAL("%s failed: %s", what, cudaGetErrorString(err));
    }
    return 1;
}

static unsigned blt_cuda_grid(size_t n) {
    const unsigned block = 256;
    size_t blocks = (n + block - 1) / block;
    if (blocks > 4096) blocks = 4096;
    return (unsigned)(blocks > 0 ? blocks : 1);
}

// Softmax

__global__ void blt_softmax_kernel(const float* in, float* out, size_t rows, size_t last_dim) {
    for (size_t row = (size_t)blockIdx.x * blockDim.x + threadIdx.x; row < rows;
         row += (size_t)gridDim.x * blockDim.x) {
        const size_t base = row * last_dim;
        float max_val = in[base];
        for (size_t i = 1; i < last_dim; ++i) {
            float v = in[base + i];
            if (v > max_val) {
                max_val = v;
            }
        }

        float sum = 0.0f;
        for (size_t i = 0; i < last_dim; ++i) {
            float exp_val = expf(in[base + i] - max_val);
            out[base + i] = exp_val;
            sum += exp_val;
        }

        for (size_t i = 0; i < last_dim; ++i) {
            out[base + i] /= sum;
        }
    }
}

extern "C" void blt_softmax_cuda(const blt_tensor* in, blt_tensor* out) {
    BLT_REQUIRE(in->ndim != 0 && in->shape[in->ndim - 1] != 0, "blt_softmax: empty last dimension");
    const size_t last_dim = in->shape[in->ndim - 1];
    const size_t rows = in->numel / last_dim;
    blt_softmax_kernel<<<blt_cuda_grid(rows), 256>>>(
        (const float*)in->data, (float*)out->data, rows, last_dim);
    blt_cuda_launch_check("blt_softmax");
}

// dx_i = y_i * (dy_i - sum_j(dy_j * y_j)), applied per row over the last dim.
__global__ void blt_softmax_backward_kernel(const float* dy, const float* y, float* dx,
                                            size_t rows, size_t last_dim) {
    for (size_t row = (size_t)blockIdx.x * blockDim.x + threadIdx.x; row < rows;
         row += (size_t)gridDim.x * blockDim.x) {
        const size_t base = row * last_dim;
        float dot = 0.0f;
        for (size_t j = 0; j < last_dim; j++) {
            dot += dy[base + j] * y[base + j];
        }
        for (size_t j = 0; j < last_dim; j++) {
            dx[base + j] = y[base + j] * (dy[base + j] - dot);
        }
    }
}

extern "C" void blt_softmax_backward_cuda(const blt_tensor* grad_out, const blt_tensor* softmax_out,
                                          blt_tensor* grad_in) {
    const size_t last_dim = softmax_out->shape[softmax_out->ndim - 1];
    BLT_REQUIRE(last_dim != 0, "blt_softmax_backward: softmax_out must have a non-zero last dimension");
    const size_t rows = softmax_out->numel / last_dim;
    blt_softmax_backward_kernel<<<blt_cuda_grid(rows), 256>>>(
        (const float*)grad_out->data, (const float*)softmax_out->data, (float*)grad_in->data,
        rows, last_dim);
    blt_cuda_launch_check("blt_softmax_backward");
}

// Cross Entropy

// Per-row: loss_i = log(sum_v exp(logits[v] - max)) + max - logits[target].
__global__ void blt_ce_row_loss_kernel(const float* logits, const unsigned char* targets,
                                       float* row_loss, size_t seq_len, size_t vocab_size) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < seq_len;
         i += (size_t)gridDim.x * blockDim.x) {
        const float* row = logits + i * vocab_size;
        const size_t target = targets[i];

        float max_val = row[0];
        for (size_t v = 1; v < vocab_size; v++) {
            if (row[v] > max_val) {
                max_val = row[v];
            }
        }

        float sum_exp = 0.0f;
        for (size_t v = 0; v < vocab_size; v++) {
            sum_exp += expf(row[v] - max_val);
        }

        row_loss[i] = logf(sum_exp) + max_val - row[target];
    }
}

// Sequential mean so accumulation order matches the CPU reference exactly.
__global__ void blt_ce_mean_kernel(const float* row_loss, float* loss_out, size_t n) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float total = 0.0f;
        for (size_t i = 0; i < n; i++) {
            total += row_loss[i];
        }
        loss_out[0] = total / (float)n;
    }
}

static void blt_ce_validate_targets_device(const blt_tensor* targets, size_t seq_len, size_t vocab_size,
                                           const char* op_name) {
    BLT_REQUIRE(targets->dtype == BLT_DTYPE_UINT8, "%s: targets must be UINT8", op_name);
    BLT_REQUIRE(targets->backend == BLT_BACKEND_CUDA, "%s: targets must live on the CUDA backend", op_name);
    unsigned char* host_targets = (unsigned char*)malloc(seq_len);
    BLT_REQUIRE(host_targets != NULL, "%s: failed to allocate target validation buffer", op_name);
    blt_cuda_memcpy_d2h(host_targets, targets->data, seq_len);
    for (size_t i = 0; i < seq_len; i++) {
        if ((size_t)host_targets[i] >= vocab_size) {
            free(host_targets);
            BLT_FATAL("%s: target index out of range for vocab_size", op_name);
        }
    }
    free(host_targets);
}

extern "C" void blt_cross_entropy_forward_cuda(const blt_tensor* logits, const blt_tensor* targets,
                                               blt_tensor* loss_out) {
    BLT_REQUIRE(loss_out->dtype == BLT_DTYPE_FP32 && loss_out->numel == 1,
                "blt_cross_entropy_forward: loss_out must be a 1-element FP32 tensor");
    const size_t seq_len = logits->shape[0];
    const size_t vocab_size = logits->shape[1];
    blt_ce_validate_targets_device(targets, seq_len, vocab_size, "blt_cross_entropy_forward");

    float* row_loss = (float*)blt_arena_alloc(blt_cuda_get_scratch_arena(),
                                              seq_len * sizeof(float), 16);
    blt_ce_row_loss_kernel<<<blt_cuda_grid(seq_len), 256>>>(
        (const float*)logits->data, (const unsigned char*)targets->data,
        row_loss, seq_len, vocab_size);
    cudaError_t err = cudaGetLastError();
    if (err == cudaSuccess) {
        blt_ce_mean_kernel<<<1, 1>>>(row_loss, (float*)loss_out->data, seq_len);
        err = cudaGetLastError();
    }
    blt_cuda_launch_check("blt_cross_entropy_forward");
}

// dL/dlogits = (softmax(logits) - one_hot(targets)) / seq_len, one thread per row.
__global__ void blt_cross_entropy_backward_kernel(const float* logits, const unsigned char* targets,
                                                  float* grad, size_t seq_len, size_t vocab_size,
                                                  float inv_seq_len) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < seq_len;
         i += (size_t)gridDim.x * blockDim.x) {
        const float* row = logits + i * vocab_size;
        float* grad_row = grad + i * vocab_size;
        const size_t target = targets[i];

        float max_val = row[0];
        for (size_t v = 1; v < vocab_size; v++) {
            if (row[v] > max_val) {
                max_val = row[v];
            }
        }

        float sum_exp = 0.0f;
        for (size_t v = 0; v < vocab_size; v++) {
            grad_row[v] = expf(row[v] - max_val);
            sum_exp += grad_row[v];
        }

        for (size_t v = 0; v < vocab_size; v++) {
            float softmax_v = grad_row[v] / sum_exp;
            float one_hot_v = (v == target) ? 1.0f : 0.0f;
            grad_row[v] = (softmax_v - one_hot_v) * inv_seq_len;
        }
    }
}

extern "C" void blt_cross_entropy_backward_cuda(const blt_tensor* logits, const blt_tensor* targets,
                                                blt_tensor* grad_logits) {
    const size_t seq_len = logits->shape[0];
    const size_t vocab_size = logits->shape[1];
    blt_ce_validate_targets_device(targets, seq_len, vocab_size, "blt_cross_entropy_backward");

    blt_cross_entropy_backward_kernel<<<blt_cuda_grid(seq_len), 256>>>(
        (const float*)logits->data, (const unsigned char*)targets->data,
        (float*)grad_logits->data, seq_len, vocab_size, 1.0f / (float)seq_len);
    blt_cuda_launch_check("blt_cross_entropy_backward");
}

// RoPE

__global__ void blt_rope_precompute_kernel(float* cos_out, float* sin_out, size_t max_seq_len,
                                           size_t half, float theta, size_t head_dim) {
    const size_t count = max_seq_len * half;
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x; idx < count;
         idx += (size_t)gridDim.x * blockDim.x) {
        const size_t pos = idx / half;
        const size_t i = idx % half;
        float freq = 1.0f / powf(theta, (float)(2 * i) / (float)head_dim);
        float angle = (float)pos * freq;
        cos_out[idx] = cosf(angle);
        sin_out[idx] = sinf(angle);
    }
}

extern "C" void blt_rope_precompute_cuda(size_t max_seq_len, const blt_rope_config* config,
                                         blt_tensor* cos_out, blt_tensor* sin_out) {
    const size_t half = config->head_dim / 2;
    const size_t count = max_seq_len * half;
    blt_rope_precompute_kernel<<<blt_cuda_grid(count), 256>>>(
        (float*)cos_out->data, (float*)sin_out->data, max_seq_len, half,
        config->theta, config->head_dim);
    blt_cuda_launch_check("blt_rope_precompute");
}

__global__ void blt_rope_apply_kernel(const float* x, const float* cos, const float* sin, float* out,
                                      size_t seq_len, size_t num_heads, size_t head_dim, size_t half) {
    const size_t rows = seq_len * num_heads;
    for (size_t r = (size_t)blockIdx.x * blockDim.x + threadIdx.x; r < rows;
         r += (size_t)gridDim.x * blockDim.x) {
        const size_t t = r / num_heads;
        const float* xv = x + r * head_dim;
        float* ov = out + r * head_dim;
        const float* c = cos + t * half;
        const float* s = sin + t * half;
        for (size_t i = 0; i < half; i++) {
            float x0 = xv[2 * i];
            float x1 = xv[2 * i + 1];
            ov[2 * i]     = x0 * c[i] - x1 * s[i];
            ov[2 * i + 1] = x1 * c[i] + x0 * s[i];
        }
    }
}

extern "C" void blt_rope_apply_cuda(const blt_tensor* x, const blt_tensor* cos, const blt_tensor* sin,
                                    blt_tensor* out) {
    const size_t seq_len = x->shape[0];
    const size_t num_heads = x->shape[1];
    const size_t head_dim = x->shape[2];
    const size_t half = head_dim / 2;
    blt_rope_apply_kernel<<<blt_cuda_grid(seq_len * num_heads), 256>>>(
        (const float*)x->data, (const float*)cos->data, (const float*)sin->data, (float*)out->data,
        seq_len, num_heads, head_dim, half);
    blt_cuda_launch_check("blt_rope_apply");
}

// Backward rotation: forward applied with sin negated.
__global__ void blt_rope_apply_backward_kernel(const float* grad_out, const float* cos, const float* sin,
                                               float* grad_in, size_t seq_len, size_t num_heads,
                                               size_t head_dim, size_t half) {
    const size_t rows = seq_len * num_heads;
    for (size_t r = (size_t)blockIdx.x * blockDim.x + threadIdx.x; r < rows;
         r += (size_t)gridDim.x * blockDim.x) {
        const size_t t = r / num_heads;
        const float* gov = grad_out + r * head_dim;
        float* giv = grad_in + r * head_dim;
        const float* c = cos + t * half;
        const float* s = sin + t * half;
        for (size_t i = 0; i < half; i++) {
            float g0 = gov[2 * i];
            float g1 = gov[2 * i + 1];
            giv[2 * i]     = g0 * c[i] + g1 * s[i];
            giv[2 * i + 1] = g1 * c[i] - g0 * s[i];
        }
    }
}

extern "C" void blt_rope_apply_backward_cuda(const blt_tensor* grad_out, const blt_tensor* cos,
                                             const blt_tensor* sin, blt_tensor* grad_in) {
    const size_t seq_len = grad_out->shape[0];
    const size_t num_heads = grad_out->shape[1];
    const size_t head_dim = grad_out->shape[2];
    const size_t half = head_dim / 2;
    blt_rope_apply_backward_kernel<<<blt_cuda_grid(seq_len * num_heads), 256>>>(
        (const float*)grad_out->data, (const float*)cos->data, (const float*)sin->data,
        (float*)grad_in->data, seq_len, num_heads, head_dim, half);
    blt_cuda_launch_check("blt_rope_apply_backward");
}

// Fused strided_copy + RoPE + strided_copy: reads from packed QKV layout,
// applies RoPE in-place, writes back to packed layout.
// Eliminates 3 kernel launches per head (strided_copy -> rope -> strided_copy).
__global__ void blt_rope_apply_packed_kernel(float* qkv_data, size_t qkv_stride,
                                             size_t head_offset, size_t seq_len,
                                              size_t head_dim, const float* cos, const float* sin) {
    const size_t half = head_dim / 2;
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = seq_len * half;

    if (tid >= total) return;

    const size_t t = tid / half;
    const size_t i = tid % half;

    const size_t base = t * qkv_stride + head_offset;
    float x0 = qkv_data[base + 2 * i];
    float x1 = qkv_data[base + 2 * i + 1];
    float c = cos[t * half + i];
    float s = sin[t * half + i];

    qkv_data[base + 2 * i]     = x0 * c - x1 * s;
    qkv_data[base + 2 * i + 1] = x1 * c + x0 * s;
}

extern "C" void blt_rope_apply_packed_cuda(float* qkv_data, size_t qkv_stride,
                                           size_t head_offset, size_t seq_len,
                                           size_t head_dim, const float* cos, const float* sin) {
    const size_t half = head_dim / 2;
    const size_t total = seq_len * half;
    const unsigned block = 256;
    unsigned blocks = (unsigned)((total + block - 1) / block);
    if (blocks > 4096) blocks = 4096;
    if (blocks == 0) blocks = 1;

    blt_rope_apply_packed_kernel<<<blocks, block>>>(qkv_data, qkv_stride,
                                                    head_offset, seq_len,
                                                    head_dim, cos, sin);
    blt_cuda_launch_check("blt_rope_apply_packed");
}

__global__ void blt_rope_apply_packed_backward_kernel(float* qkv_data, size_t qkv_stride,
                                                      size_t head_offset, size_t seq_len,
                                                      size_t head_dim, const float* cos, const float* sin) {
    const size_t half = head_dim / 2;
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = seq_len * half;

    if (tid >= total) return;

    const size_t t = tid / half;
    const size_t i = tid % half;

    const size_t base = t * qkv_stride + head_offset;
    float g0 = qkv_data[base + 2 * i];
    float g1 = qkv_data[base + 2 * i + 1];
    float c = cos[t * half + i];
    float s = sin[t * half + i];

    qkv_data[base + 2 * i]     = g0 * c + g1 * s;
    qkv_data[base + 2 * i + 1] = g1 * c - g0 * s;
}

extern "C" void blt_rope_apply_packed_backward_cuda(float* qkv_data, size_t qkv_stride,
                                                    size_t head_offset, size_t seq_len,
                                                    size_t head_dim, const float* cos, const float* sin) {
    const size_t half = head_dim / 2;
    const size_t total = seq_len * half;
    const unsigned block = 256;
    unsigned blocks = (unsigned)((total + block - 1) / block);
    if (blocks > 4096) blocks = 4096;
    if (blocks == 0) blocks = 1;
    
    blt_rope_apply_packed_backward_kernel<<<blocks, block>>>(qkv_data, qkv_stride,
                                                             head_offset, seq_len,
                                                             head_dim, cos, sin);
    blt_cuda_launch_check("blt_rope_apply_packed_backward");
}

// RMSNorm

__global__ void blt_rmsnorm_forward_kernel(const float* x, const float* w, float* out,
                                           size_t seq_len, size_t embed_dim) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < seq_len;
         i += (size_t)gridDim.x * blockDim.x) {
        const float* row = x + i * embed_dim;
        float* out_row = out + i * embed_dim;

        float sumsq = 0.0f;
        for (size_t j = 0; j < embed_dim; j++) sumsq += row[j] * row[j];
        float mean_sq = sumsq / (float)embed_dim;
        float inv_rms = 1.0f / sqrtf(mean_sq + BLT_RMSNORM_EPS);

        for (size_t j = 0; j < embed_dim; j++) {
            out_row[j] = row[j] * inv_rms * w[j];
        }
    }
}

extern "C" void blt_rmsnorm_forward_cuda(const blt_tensor* x, const blt_tensor* weight, blt_tensor* out) {
    const size_t seq_len = x->shape[0];
    const size_t embed_dim = x->shape[1];
    blt_rmsnorm_forward_kernel<<<blt_cuda_grid(seq_len), 256>>>(
        (const float*)x->data, (const float*)weight->data, (float*)out->data, seq_len, embed_dim);
    blt_cuda_launch_check("blt_rmsnorm_forward");
}

// Fused RMSNorm backward: computes inv_rms in shared memory, then grad_x and grad_weight.
// One block per RMSNorm call. Uses shared memory for inv_rms table (seq_len * 4 bytes).
__global__ void blt_rmsnorm_backward_fused_kernel(const float* grad_out, const float* x, const float* w,
                                                  float* grad_x, float* grad_weight,
                                                  size_t seq_len, size_t embed_dim) {
    extern __shared__ float shmem[];
    float* inv_rms = shmem;
    
    const size_t tid = threadIdx.x;
    const size_t num_threads = blockDim.x;
    
    for (size_t i = tid; i < seq_len; i += num_threads) {
        const float* row = x + i * embed_dim;
        float sumsq = 0.0f;
        for (size_t j = 0; j < embed_dim; j++) sumsq += row[j] * row[j];
        float mean_sq = sumsq / (float)embed_dim;
        inv_rms[i] = 1.0f / sqrtf(mean_sq + BLT_RMSNORM_EPS);
    }
    __syncthreads();
    
    for (size_t i = tid; i < seq_len; i += num_threads) {
        const float* row = x + i * embed_dim;
        const float* go_row = grad_out + i * embed_dim;
        float* gx_row = grad_x + i * embed_dim;
        
        float dot = 0.0f;
        for (size_t j = 0; j < embed_dim; j++) {
            dot += go_row[j] * w[j] * row[j];
        }
        
        float inv_rms3_over_n = (inv_rms[i] * inv_rms[i] * inv_rms[i]) / (float)embed_dim;
        for (size_t j = 0; j < embed_dim; j++) {
            gx_row[j] = go_row[j] * w[j] * inv_rms[i] - row[j] * inv_rms3_over_n * dot;
        }
    }
    __syncthreads();
    
    for (size_t j = tid; j < embed_dim; j += num_threads) {
        float acc = grad_weight[j];
        for (size_t i = 0; i < seq_len; i++) {
            acc += grad_out[i * embed_dim + j] * x[i * embed_dim + j] * inv_rms[i];
        }
        grad_weight[j] = acc;
    }
}

extern "C" void blt_rmsnorm_backward_cuda(const blt_tensor* grad_out, const blt_tensor* x,
                                          const blt_tensor* weight, blt_tensor* grad_x,
                                          blt_tensor* grad_weight) {
    const size_t seq_len = x->shape[0];
    const size_t embed_dim = x->shape[1];

    size_t shared_mem = seq_len * sizeof(float);
    size_t threads = (seq_len > embed_dim ? seq_len : embed_dim);
    if (threads > 256) threads = 256;
    blt_rmsnorm_backward_fused_kernel<<<1, threads, shared_mem>>>(
        (const float*)grad_out->data, (const float*)x->data, (const float*)weight->data,
        (float*)grad_x->data, (float*)grad_weight->data, seq_len, embed_dim);
    blt_cuda_launch_check("blt_rmsnorm_backward");
}

// LayerNorm (forward only; backward is unimplemented on CPU as well)

__global__ void blt_layernorm_forward_kernel(const float* x, const float* w, const float* b, float* out,
                                             size_t seq_len, size_t embed_dim, float eps) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < seq_len;
         i += (size_t)gridDim.x * blockDim.x) {
        const float* row = x + i * embed_dim;
        float* out_row = out + i * embed_dim;

        float mean = 0.0f;
        for (size_t j = 0; j < embed_dim; j++) mean += row[j];
        mean /= (float)embed_dim;

        float variance = 0.0f;
        for (size_t j = 0; j < embed_dim; j++) {
            float diff = row[j] - mean;
            variance += diff * diff;
        }
        variance /= (float)embed_dim;

        float inv_std = 1.0f / sqrtf(variance + eps);
        for (size_t j = 0; j < embed_dim; j++) {
            out_row[j] = (row[j] - mean) * inv_std * w[j] + b[j];
        }
    }
}

extern "C" void blt_layernorm_forward_cuda(const blt_tensor* x, const blt_tensor* weight,
                                           const blt_tensor* bias, blt_tensor* out, float eps) {
    const size_t seq_len = x->shape[0];
    const size_t embed_dim = x->shape[1];
    blt_layernorm_forward_kernel<<<blt_cuda_grid(seq_len), 256>>>(
        (const float*)x->data, (const float*)weight->data, (const float*)bias->data,
        (float*)out->data, seq_len, embed_dim, eps);
    blt_cuda_launch_check("blt_layernorm_forward");
}
