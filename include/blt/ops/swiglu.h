#ifndef BLT_OPS_SWIGLU_H
#define BLT_OPS_SWIGLU_H
#include "blt/core/tensor.h"

// Elementwise SwiGLU gate: out = silu(gate) * up, where silu(z) = z * sigmoid(z).
// gate, up, and out must all have the same element count (typically
// [seq_len, hidden_dim], already produced by separate gate/up projections).
void blt_swiglu_forward(const blt_tensor* gate, const blt_tensor* up, blt_tensor* out);
void blt_swiglu_backward(const blt_tensor* grad_out, const blt_tensor* gate, const blt_tensor* up,
                          blt_tensor* grad_gate, blt_tensor* grad_up);

#endif // BLT_OPS_SWIGLU_H