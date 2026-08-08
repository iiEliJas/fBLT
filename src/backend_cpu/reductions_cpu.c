#include "blt/core/backend.h"
#include "blt/ops/softmax.h"
#include "blt/ops/layernorm.h"
#include "blt/ops/rmsnorm.h"
#include "blt/ops/rope.h"

#include <math.h>



//----------------------------------------------------------------
// Softmax

void blt_softmax_cpu(const blt_tensor* in, blt_tensor* out) {
    BLT_REQUIRE(in->dtype == BLT_DTYPE_FP32 && out->dtype == BLT_DTYPE_FP32, "softmax only supports FP32 tensors");
    BLT_REQUIRE(in->numel == out->numel, "softmax input/output numel mismatch");

    const float* in_data = (const float*)in->data;
    float* out_data = (float*)out->data;

    size_t last_dim = in->ndim == 0 ? 0 : in->shape[in->ndim - 1];
    size_t rows = in->numel / last_dim;

    for (size_t row = 0; row < rows; ++row) {
        size_t base = row * last_dim;
        float max_val = in_data[base];
        for (size_t i = 1; i < last_dim; ++i) {
            float v = in_data[base + i];
            if (v > max_val) {
                max_val = v;
            }
        }

        float sum = 0.0f;
        for (size_t i = 0; i < last_dim; ++i) {
            float exp_val = expf(in_data[base + i] - max_val);
            out_data[base + i] = exp_val;
            sum += exp_val;
        }

        for (size_t i = 0; i < last_dim; ++i) {
            out_data[base + i] /= sum;
        }
    }
}



//----------------------------------------------------------------
// RoPE (Rotary Position Embedding)

void blt_rope_precompute_cpu(size_t max_seq_len, const blt_rope_config* config,
                          blt_tensor* cos_out, blt_tensor* sin_out) {
    BLT_REQUIRE(config->head_dim % 2 == 0, "rope: head_dim must be even");
    size_t half = config->head_dim / 2;
    blt_check_nd_fp32(cos_out, 2, (const size_t[]){max_seq_len, half}, "rope cos_out");
    blt_check_nd_fp32(sin_out, 2, (const size_t[]){max_seq_len, half}, "rope sin_out");

    float* cos_data = (float*)cos_out->data;
    float* sin_data = (float*)sin_out->data;

    for (size_t pos = 0; pos < max_seq_len; pos++) {
        for (size_t i = 0; i < half; i++) {
            float freq = 1.0f / powf(config->theta, (float)(2 * i) / (float)config->head_dim);
            float angle = (float)pos * freq;
            cos_data[pos * half + i] = cosf(angle);
            sin_data[pos * half + i] = sinf(angle);
        }
    }
}

void blt_rope_apply_cpu(const blt_tensor* x, const blt_tensor* cos, const blt_tensor* sin, blt_tensor* out) {
    blt_check_nd_fp32(x, 3, (const size_t[]){0, 0, 0}, "rope: x must be 3D FP32");
    blt_check_nd_fp32(cos, 2, (const size_t[]){x->shape[0], x->shape[2] / 2}, "rope: cos shape mismatch");
    blt_check_nd_fp32(sin, 2, (const size_t[]){x->shape[0], x->shape[2] / 2}, "rope: sin shape mismatch");
    blt_check_nd_fp32(out, 3, (const size_t[]){x->shape[0], x->shape[1], x->shape[2]}, "rope: out shape mismatch");

    size_t seq_len = x->shape[0];
    size_t num_heads = x->shape[1];
    size_t head_dim = x->shape[2];
    size_t half = head_dim / 2;

    const float* xd = (const float*)x->data;
    const float* cd = (const float*)cos->data;
    const float* sd = (const float*)sin->data;
    float* od = (float*)out->data;

    for (size_t t = 0; t < seq_len; t++) {
        for (size_t h = 0; h < num_heads; h++) {
            const float* xv = xd + (t * num_heads + h) * head_dim;
            float* ov = od + (t * num_heads + h) * head_dim;
            const float* c = cd + t * half;
            const float* s = sd + t * half;
            for (size_t i = 0; i < half; i++) {
                float x0 = xv[2 * i];
                float x1 = xv[2 * i + 1];
                ov[2 * i]     = x0 * c[i] - x1 * s[i];
                ov[2 * i + 1] = x1 * c[i] + x0 * s[i];
            }
        }
    }
}



//----------------------------------------------------------------
// LayerNorm

void blt_layernorm_forward_cpu(const blt_tensor* x, const blt_tensor* weight, const blt_tensor* bias,
                                blt_tensor* out, float eps) {
    blt_check_nd_fp32(x, 2, (const size_t[]){0, 0}, "LayerNorm: input must be 2D FP32");
    size_t seq_len = x->shape[0];
    size_t embed_dim = x->shape[1];
 
    blt_check_nd_fp32(weight, 1, (const size_t[]){embed_dim}, "LayerNorm: weight must be [embed_dim] FP32");
    blt_check_nd_fp32(bias, 1, (const size_t[]){embed_dim}, "LayerNorm: bias must be [embed_dim] FP32");
    blt_check_nd_fp32(out, 2, (const size_t[]){seq_len, embed_dim}, "LayerNorm: output shape/dtype mismatch");
    BLT_REQUIRE(eps >= 1e-12f, "LayerNorm: eps is too small");
 
    const float* in = (const float*)x->data;
    const float* w = (const float*)weight->data;
    const float* b = (const float*)bias->data;
    float* o = (float*)out->data;
 
    for (size_t i = 0; i < seq_len; i++) {
        const float* row = in + i * embed_dim;
        float* out_row = o + i * embed_dim;
 
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
 
void blt_layernorm_backward_cpu(const blt_tensor* grad_out, const blt_tensor* x,
                                 const blt_tensor* weight, blt_tensor* grad_x,
                                 blt_tensor* grad_weight, blt_tensor* grad_bias, float eps) {
    (void)grad_out; (void)x; (void)weight; (void)grad_x; (void)grad_weight; (void)grad_bias; (void)eps;
    BLT_FATAL("LayerNorm backward not yet implemented");
}



//----------------------------------------------------------------
// RMSNorm
 
#define BLT_RMSNORM_EPS 1e-6f
 
void blt_rmsnorm_forward_cpu(const blt_tensor* x, const blt_tensor* weight, blt_tensor* out) {
    blt_check_nd_fp32(x, 2, (const size_t[]){0, 0}, "RMSNorm: input must be 2D FP32");
    size_t seq_len = x->shape[0];
    size_t embed_dim = x->shape[1];
 
    blt_check_nd_fp32(weight, 1, (const size_t[]){embed_dim}, "RMSNorm: weight must be [embed_dim] FP32");
    blt_check_nd_fp32(out, 2, (const size_t[]){seq_len, embed_dim}, "RMSNorm: output shape/dtype mismatch");
 
    const float* in = (const float*)x->data;
    const float* w = (const float*)weight->data;
    float* o = (float*)out->data;
 
    for (size_t i = 0; i < seq_len; i++) {
        const float* row = in + i * embed_dim;
        float* out_row = o + i * embed_dim;
 
        float sumsq = 0.0f;
        for (size_t j = 0; j < embed_dim; j++) sumsq += row[j] * row[j];
        float mean_sq = sumsq / (float)embed_dim;
        float inv_rms = 1.0f / sqrtf(mean_sq + BLT_RMSNORM_EPS);
 
        for (size_t j = 0; j < embed_dim; j++) {
            out_row[j] = row[j] * inv_rms * w[j];
        }
    }
}
 
void blt_rmsnorm_backward_cpu(const blt_tensor* grad_out, const blt_tensor* x,
                               const blt_tensor* weight, blt_tensor* grad_x, blt_tensor* grad_weight) {
    (void)grad_out; (void)x; (void)weight; (void)grad_x; (void)grad_weight;
    BLT_FATAL("RMSNorm backward not yet implemented");
}
