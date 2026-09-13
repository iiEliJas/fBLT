#ifndef BLT_OPS_LAYERNORM_H
#define BLT_OPS_LAYERNORM_H

#ifdef __cplusplus
extern "C" {
#endif
#include "core/tensor.h"

// Row-wise LayerNorm: zero mean / unit variance per row, then learned scale+bias.
// Use blt_rmsnorm_forward when the block config selects BLT_NORM_RMSNORM.
void blt_layernorm_forward(const blt_tensor *x, const blt_tensor *weight, const blt_tensor *bias, blt_tensor *out,
                           float eps);
void blt_layernorm_backward(const blt_tensor *grad_out, const blt_tensor *x, const blt_tensor *weight,
                            blt_tensor *grad_x, blt_tensor *grad_weight, blt_tensor *grad_bias, float eps);

#ifdef __cplusplus
}
#endif

#endif
