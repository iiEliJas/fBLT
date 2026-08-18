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


// dx_i = y_i * (dy_i - sum_j(dy_j * y_j)), applied per row over the last dim.
void blt_softmax_backward_cpu(const blt_tensor* grad_out, const blt_tensor* softmax_out, blt_tensor* grad_in) {
    BLT_REQUIRE(grad_out != NULL && softmax_out != NULL && grad_in != NULL,
                "blt_softmax_backward: grad_out, softmax_out, grad_in must not be NULL");
    BLT_REQUIRE(grad_out->dtype == BLT_DTYPE_FP32 && softmax_out->dtype == BLT_DTYPE_FP32 &&
                grad_in->dtype == BLT_DTYPE_FP32, "blt_softmax_backward: all tensors must be FP32");
    BLT_REQUIRE(grad_out->numel == softmax_out->numel && grad_out->numel == grad_in->numel,
                "blt_softmax_backward: element counts must match");
 
    const float* dy = (const float*)grad_out->data;
    const float* y = (const float*)softmax_out->data;
    float* dx = (float*)grad_in->data;
 
    size_t last_dim = softmax_out->ndim == 0 ? 0 : softmax_out->shape[softmax_out->ndim - 1];
    BLT_REQUIRE(last_dim != 0, "blt_softmax_backward: softmax_out must have a non-zero last dimension");
    size_t rows = softmax_out->numel / last_dim;
 
    for (size_t row = 0; row < rows; row++) {
        size_t base = row * last_dim;
        float dot = 0.0f;
        for (size_t j = 0; j < last_dim; j++) {
            dot += dy[base + j] * y[base + j];
        }
        for (size_t j = 0; j < last_dim; j++) {
            dx[base + j] = y[base + j] * (dy[base + j] - dot);
        }
    }
}



//----------------------------------------------------------------
// Cross Entropy

// loss = mean_i [-log(softmax(logits_i)[target_i])]
//      = mean_i [log(sum_v exp(logits_i[v] - max_i)) - (logits_i[target_i] - max_i)]
 
void blt_cross_entropy_forward_cpu(const blt_tensor* logits, const blt_tensor* targets, blt_tensor* loss_out) {
    BLT_REQUIRE(logits != NULL && targets != NULL && loss_out != NULL,
                "blt_cross_entropy_forward: logits, targets, loss_out must not be NULL");
    blt_check_nd_fp32(logits, 2, (const size_t[]){0, 0}, "blt_cross_entropy_forward: logits must be 2D FP32 [seq_len, vocab_size]");
 
    size_t seq_len = logits->shape[0];
    size_t vocab_size = logits->shape[1];
 
    BLT_REQUIRE(targets->dtype == BLT_DTYPE_UINT8, "blt_cross_entropy_forward: targets must be UINT8");
    BLT_REQUIRE(targets->ndim == 1 && targets->shape[0] == seq_len,
                "blt_cross_entropy_forward: targets must be [seq_len]");
 
    BLT_REQUIRE(loss_out->dtype == BLT_DTYPE_FP32 && loss_out->numel == 1,
                "blt_cross_entropy_forward: loss_out must be a 1-element FP32 tensor");
 
    const float* logits_data = (const float*)logits->data;
    const uint8_t* td = (const uint8_t*)targets->data;      // adjust if INT32
    float* loss_data = (float*)loss_out->data;
 
    float total_loss = 0.0f;
 
    for (size_t i = 0; i < seq_len; i++) {
        const float* row = logits_data + i * vocab_size;
        uint8_t target = td[i];
        BLT_REQUIRE((size_t)target < vocab_size,
                    "blt_cross_entropy_forward: target index out of range for vocab_size");
 
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
 
        float log_sum_exp = logf(sum_exp) + max_val;
        total_loss += (log_sum_exp - row[target]);
    }
 
    loss_data[0] = total_loss / (float)seq_len;
}
 
 

// dL/dlogits = (softmax(logits) - one_hot(targets)) / seq_len
void blt_cross_entropy_backward_cpu(const blt_tensor* logits, const blt_tensor* targets, blt_tensor* grad_logits) {
    BLT_REQUIRE(logits != NULL && targets != NULL && grad_logits != NULL,
                "blt_cross_entropy_backward: logits, targets, grad_logits must not be NULL");
    blt_check_nd_fp32(logits, 2, (const size_t[]){0, 0}, "blt_cross_entropy_backward: logits must be 2D FP32 [seq_len, vocab_size]");
 
    size_t seq_len = logits->shape[0];
    size_t vocab_size = logits->shape[1];
 
    BLT_REQUIRE(targets->dtype == BLT_DTYPE_UINT8, "blt_cross_entropy_backward: targets must be UINT8");
    BLT_REQUIRE(targets->ndim == 1 && targets->shape[0] == seq_len,
                "blt_cross_entropy_backward: targets must be [seq_len]");
 
    blt_check_nd_fp32(grad_logits, 2, (const size_t[]){seq_len, vocab_size},
                       "blt_cross_entropy_backward: grad_logits must be [seq_len, vocab_size] FP32");
 
    const float* logits_data = (const float*)logits->data;
    const uint8_t* td = (const uint8_t*)targets->data;      // adjust if INT32
    float* grad_data = (float*)grad_logits->data;
 
    float inv_seq_len = 1.0f / (float)seq_len;
 
    for (size_t i = 0; i < seq_len; i++) {
        const float* row = logits_data + i * vocab_size;
        float* grad_row = grad_data + i * vocab_size;
        uint8_t target = td[i];
        BLT_REQUIRE((size_t)target < vocab_size,
                    "blt_cross_entropy_backward: target index out of range for vocab_size");
 
        float max_val = row[0];
        for (size_t v = 1; v < vocab_size; v++) {
            if (row[v] > max_val) {
                max_val = row[v];
            }
        }
 
        float sum_exp = 0.0f;
        for (size_t v = 0; v < vocab_size; v++) {
            grad_row[v] = expf(row[v] - max_val);   // store exp(logit - max)
            sum_exp += grad_row[v];
        }
 
        for (size_t v = 0; v < vocab_size; v++) {
            float softmax_v = grad_row[v] / sum_exp;
            float one_hot_v = (v == target) ? 1.0f : 0.0f;
            grad_row[v] = (softmax_v - one_hot_v) * inv_seq_len;
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



// RoPE is a per pair rotation:
// out0 =  x0*cos - x1*sin
// out1 =  x1*cos + x0*sin
// the jacobian is an orthogonal 2x2 rotation matrix, so the backward pass is just the inverse rotation
// so just flip the sign of sin (sign of sin is a funny saying):
//   grad_x0 =  grad_out0*cos + grad_out1*sin
//   grad_x1 = -grad_out0*sin + grad_out1*cos
// which is exactly forward rope applied with sin negated
 
void blt_rope_apply_backward_cpu(const blt_tensor* grad_out, const blt_tensor* cos, const blt_tensor* sin,
                              blt_tensor* grad_in) {
    blt_check_nd_fp32(grad_out, 3, (const size_t[]){0, 0, 0}, "blt_rope_apply_backward: grad_out must be 3D FP32");
    blt_check_nd_fp32(cos, 2, (const size_t[]){grad_out->shape[0], grad_out->shape[2] / 2},
                       "blt_rope_apply_backward: cos shape mismatch");
    blt_check_nd_fp32(sin, 2, (const size_t[]){grad_out->shape[0], grad_out->shape[2] / 2},
                       "blt_rope_apply_backward: sin shape mismatch");
    blt_check_nd_fp32(grad_in, 3, (const size_t[]){grad_out->shape[0], grad_out->shape[1], grad_out->shape[2]},
                       "blt_rope_apply_backward: grad_in shape mismatch");
 
    size_t seq_len = grad_out->shape[0];
    size_t num_heads = grad_out->shape[1];
    size_t head_dim = grad_out->shape[2];
    size_t half = head_dim / 2;
 
    const float* god = (const float*)grad_out->data;
    const float* cd = (const float*)cos->data;
    const float* sd = (const float*)sin->data;
    float* gid = (float*)grad_in->data;
 
    for (size_t t = 0; t < seq_len; t++) {
        for (size_t h = 0; h < num_heads; h++) {
            const float* gov = god + (t * num_heads + h) * head_dim;
            float* giv = gid + (t * num_heads + h) * head_dim;
            const float* c = cd + t * half;
            const float* s = sd + t * half;
            for (size_t i = 0; i < half; i++) {
                float g0 = gov[2 * i];
                float g1 = gov[2 * i + 1];
                // inverse rotation is forward rotation with sin negated
                giv[2 * i]     = g0 * c[i] + g1 * s[i];
                giv[2 * i + 1] = g1 * c[i] - g0 * s[i];
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
 
    for(size_t i = 0; i < seq_len; i++) {
        const float* row = in + i * embed_dim;
        float* out_row = o + i * embed_dim;
 
        float sumsq = 0.0f;
        for (size_t j = 0; j < embed_dim; j++) sumsq += row[j] * row[j];
        float mean_sq = sumsq / (float)embed_dim;
        float inv_rms = 1.0f / sqrtf(mean_sq + BLT_RMSNORM_EPS);
 
        for(size_t j = 0; j < embed_dim; j++) {
            out_row[j] = row[j] * inv_rms * w[j];
        }
    }
}


void blt_rmsnorm_backward_cpu(const blt_tensor* grad_out, const blt_tensor* x,
                            const blt_tensor* weight, blt_tensor* grad_x, blt_tensor* grad_weight) {
    blt_check_nd_fp32(x, 2, (const size_t[]){0, 0}, "RMSNorm backward: x must be 2D FP32");
    size_t seq_len = x->shape[0];
    size_t embed_dim = x->shape[1];
 
    blt_check_nd_fp32(weight, 1, (const size_t[]){embed_dim}, "RMSNorm backward: weight must be [embed_dim] FP32");
    blt_check_nd_fp32(grad_out, 2, (const size_t[]){seq_len, embed_dim}, "RMSNorm backward: grad_out shape/dtype mismatch");
    blt_check_nd_fp32(grad_x, 2, (const size_t[]){seq_len, embed_dim}, "RMSNorm backward: grad_x shape/dtype mismatch");
    blt_check_nd_fp32(grad_weight, 1, (const size_t[]){embed_dim}, "RMSNorm backward: grad_weight must be [embed_dim] FP32");
 
    const float* xd = (const float*)x->data;
    const float* w = (const float*)weight->data;
    const float* god = (const float*)grad_out->data;
    float* gxd = (float*)grad_x->data;
    float* gwd = (float*)grad_weight->data;
    
    // out_j = x_j * inv_rms * w_j,  inv_rms = 1/sqrt(mean(x^2) + eps)
    // grad_x_i = grad_out_i * w_i * inv_rms - (x_i * inv_rms^3 / n) * sum_j(grad_out_j * w_j * x_j)
    // grad_weight_j += sum_over_rows( grad_out[row,j] * x[row,j] * inv_rms[row] )
    // (grad_weight must be zeroed by caller)
    for(size_t i = 0; i < seq_len; i++) {
        const float* row = xd + i * embed_dim;
        const float* go_row = god + i * embed_dim;
        float* gx_row = gxd + i * embed_dim;
 
        float sumsq = 0.0f;
        for (size_t j = 0; j < embed_dim; j++) sumsq += row[j] * row[j];
        float mean_sq = sumsq / (float)embed_dim;
        float inv_rms = 1.0f / sqrtf(mean_sq + BLT_RMSNORM_EPS);
 
        float dot = 0.0f;
        for(size_t j = 0; j < embed_dim; j++) {
            dot += go_row[j] * w[j] * row[j];
        }
 
        float inv_rms3_over_n = (inv_rms * inv_rms * inv_rms) / (float)embed_dim;
        for (size_t j = 0; j < embed_dim; j++) {
            gx_row[j] = go_row[j] * w[j] * inv_rms - row[j] * inv_rms3_over_n * dot;
            gwd[j] += go_row[j] * row[j] * inv_rms;
        }
    }
}
