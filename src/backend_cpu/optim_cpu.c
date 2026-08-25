#include "blt/core/backend.h"
#include "blt/core/tensor.h"
#include "blt/ops/optim.h"

#include <math.h>


//------------------------------------------------------------
// Simple SGD step

// param -= lr * grad elementwise.
void blt_sgd_step_cpu(blt_tensor* param, const blt_tensor* grad, float lr) {
    blt_check_elementwise_fp32(param, grad,
        "blt_sgd_step: param and grad must be FP32 and elementwise-compatible");

    float* p = (float*)param->data;
    const float* g = (const float*)grad->data;

    for (size_t i = 0; i < param->numel; ++i) {
        p[i] -= lr * g[i];
    }
}


//------------------------------------------------------------
// AdamW (single tensor, decoupled weight decay)

void blt_adamw_step_cpu(blt_tensor* param, const blt_tensor* grad,
                        blt_tensor* exp_avg, blt_tensor* exp_avg_sq,
                        const blt_adamw_config* config) {
    BLT_REQUIRE(config != NULL, "blt_adamw_step: config must not be NULL");
    BLT_REQUIRE(config->step >= 1, "blt_adamw_step: step must be >= 1");
    BLT_REQUIRE(config->beta1 > 0.0f && config->beta1 < 1.0f &&
                config->beta2 > 0.0f && config->beta2 < 1.0f,
                "blt_adamw_step: betas must be in (0, 1)");
    blt_check_elementwise_fp32(param, grad,
        "blt_adamw_step: param and grad must be FP32 and elementwise-compatible");
    blt_check_elementwise_fp32(param, exp_avg,
        "blt_adamw_step: param and exp_avg must be FP32 and elementwise-compatible");
    blt_check_elementwise_fp32(param, exp_avg_sq,
        "blt_adamw_step: param and exp_avg_sq must be FP32 and elementwise-compatible");

    const float lr = config->lr;
    const float b1 = config->beta1;
    const float b2 = config->beta2;
    const float eps = config->eps;
    const float wd = config->weight_decay;
    const float bc1 = 1.0f - powf(b1, (float)config->step);
    const float bc2 = 1.0f - powf(b2, (float)config->step);

    float* p = (float*)param->data;
    const float* g = (const float*)grad->data;
    float* m = (float*)exp_avg->data;
    float* v = (float*)exp_avg_sq->data;

    for (size_t i = 0; i < param->numel; ++i) {
        const float gi = g[i];
        m[i] = b1 * m[i] + (1.0f - b1) * gi;
        v[i] = b2 * v[i] + (1.0f - b2) * gi * gi;
        const float mhat = m[i] / bc1;
        const float vhat = v[i] / bc2;
        p[i] -= lr * (mhat / (sqrtf(vhat) + eps) + wd * p[i]);
    }
}
