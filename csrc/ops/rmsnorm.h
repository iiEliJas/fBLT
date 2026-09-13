#ifndef BLT_OPS_RMSNORM_H
#define BLT_OPS_RMSNORM_H

#ifdef __cplusplus
extern "C" {
#endif
#include "core/tensor.h"

// out = x / rms(x) * weight, where rms(x) = sqrt(mean(x^2) + eps).
// No bias, no mean subtraction (that's RMSNorm vs LayerNorm).
// eps is a fixed internal constant, not an ablation option (BLT §7).
void blt_rmsnorm_forward(const blt_tensor *x, const blt_tensor *weight, blt_tensor *out);
void blt_rmsnorm_backward(const blt_tensor *grad_out, const blt_tensor *x, const blt_tensor *weight, blt_tensor *grad_x,
                          blt_tensor *grad_weight);
#ifdef __cplusplus
}
#endif
#endif
