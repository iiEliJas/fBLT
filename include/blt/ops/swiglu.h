#ifndef BLT_OPS_SWIGLU_H
#define BLT_OPS_SWIGLU_H

#ifdef __cplusplus
extern "C" {
#endif
#include "blt/core/tensor.h"

// out = silu(gate) * up, where silu(z) = z * sigmoid(z).
// gate, up, and out must have the same element count.
void blt_swiglu_forward(const blt_tensor *gate, const blt_tensor *up, blt_tensor *out);
void blt_swiglu_backward(const blt_tensor *grad_out, const blt_tensor *gate, const blt_tensor *up,
                         blt_tensor *grad_gate, blt_tensor *grad_up);

#ifdef __cplusplus
}
#endif

#endif // BLT_OPS_SWIGLU_H