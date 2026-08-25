#ifndef BLT_OPS_OPTIM_H
#define BLT_OPS_OPTIM_H

#ifdef __cplusplus
extern "C" {
#endif

#include "blt/core/tensor.h"

// In-place SGD parameter update: param -= lr * grad.
void blt_sgd_step(blt_tensor* param, const blt_tensor* grad, float lr);

// Single-tensor AdamW (decoupled weight decay), PyTorch-compatible math.
// exp_avg / exp_avg_sq are persistent per-parameter state tensors of the
// same shape/dtype as param; they must be zero-initialized before the very
// first step. `step` is 1-based and drives the bias correction.
typedef struct {
    float lr;
    float beta1;
    float beta2;
    float eps;
    float weight_decay;   // decoupled L2 applied as p -= lr * wd * p
    size_t step;
} blt_adamw_config;

void blt_adamw_step(blt_tensor* param, const blt_tensor* grad,
                    blt_tensor* exp_avg, blt_tensor* exp_avg_sq,
                    const blt_adamw_config* config);

#ifdef __cplusplus
}
#endif

#endif // BLT_OPS_OPTIM_H
