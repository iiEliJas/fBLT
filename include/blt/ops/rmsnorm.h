#ifndef BLT_OPS_RMSNORM_H
#define BLT_OPS_RMSNORM_H

#ifdef __cplusplus
extern "C" {
#endif
#include "blt/core/tensor.h"

// Row-wise RMSNorm over the last dimension of a 2D [seq_len, embed_dim]
// tensor: out = x / rms(x) * weight, where rms(x) = sqrt(mean(x^2) + eps).
// No bias term and no mean subtraction (that's what distinguishes it from
// LayerNorm) — eps is a small fixed internal constant, not exposed here,
// since it isn't an ablation option per BLT §7.
void blt_rmsnorm_forward(const blt_tensor* x, const blt_tensor* weight, blt_tensor* out);
void blt_rmsnorm_backward(const blt_tensor* grad_out, const blt_tensor* x,
                           const blt_tensor* weight, blt_tensor* grad_x, blt_tensor* grad_weight);
#ifdef __cplusplus
}
#endif
#endif
