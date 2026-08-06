#include "blt/core/backend.h"
#include "blt/ops/softmax.h"
#include "blt/ops/layernorm.h"
#include "blt/ops/rmsnorm.h"

#include <math.h>



//----------------------------------------------------------------
// Softmax

void blt_softmax_cpu(const blt_tensor* in, blt_tensor* out) {
    if (in->dtype != BLT_DTYPE_FP32 || out->dtype != BLT_DTYPE_FP32) {
        BLT_FATAL("softmax only supports FP32 tensors");
    }
    if (in->numel != out->numel) {
        BLT_FATAL("softmax input/output numel mismatch");
    }

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
// LayerNorm

void blt_layernorm_forward_cpu(const blt_tensor* x, const blt_tensor* weight, const blt_tensor* bias,
                                blt_tensor* out, float eps) {
    blt_check_2d_fp32(x, 0, 0, "LayerNorm: input must be 2D FP32");
    size_t seq_len = x->shape[0];
    size_t embed_dim = x->shape[1];
 
    blt_check_1d_fp32(weight, embed_dim, "LayerNorm: weight must be [embed_dim] FP32");
    blt_check_1d_fp32(bias, embed_dim, "LayerNorm: bias must be [embed_dim] FP32");
    blt_check_2d_fp32(out, seq_len, embed_dim, "LayerNorm: output shape/dtype mismatch");
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
    blt_check_2d_fp32(x, 0, 0, "RMSNorm: input must be 2D FP32");
    size_t seq_len = x->shape[0];
    size_t embed_dim = x->shape[1];
 
    blt_check_1d_fp32(weight, embed_dim, "RMSNorm: weight must be [embed_dim] FP32");
    blt_check_2d_fp32(out, seq_len, embed_dim, "RMSNorm: output shape/dtype mismatch");
 
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
