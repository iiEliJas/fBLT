#ifndef BLT_OPS_LAYERNORM_H
#define BLT_OPS_LAYERNORM_H
#include "blt/core/tensor.h"

// Row-wise LayerNorm over the last dimension of a 2D [seq_len, embed_dim]
// tensor: normalizes each row to zero mean / unit variance, then applies a
// learned per-channel weight and bias. Use blt_rmsnorm_forward instead when
// the block config selects BLT_NORM_RMSNORM (no bias term, no mean subtraction).
void blt_layernorm_forward(const blt_tensor* x, const blt_tensor* weight, const blt_tensor* bias,
                            blt_tensor* out, float eps);
void blt_layernorm_backward(const blt_tensor* grad_out, const blt_tensor* x,
                             const blt_tensor* weight, blt_tensor* grad_x,
                             blt_tensor* grad_weight, blt_tensor* grad_bias, float eps);

#endif // BLT_OPS_LAYERNORM_H