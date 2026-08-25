#ifndef BLT_OPS_GELU_H
#define BLT_OPS_GELU_H

#ifdef __cplusplus
extern "C" {
#endif
#include "blt/core/tensor.h"

// Elementwise GELU (tanh approximation):
//   out = 0.5*x*(1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
// x and out just need matching element count (any rank).
void blt_gelu_forward(const blt_tensor* x, blt_tensor* out);
void blt_gelu_backward(const blt_tensor* grad_out, const blt_tensor* x, blt_tensor* grad_x);

#ifdef __cplusplus
}
#endif

#endif // BLT_OPS_GELU_H