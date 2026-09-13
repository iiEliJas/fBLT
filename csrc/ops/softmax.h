#ifndef BLT_OPS_SOFTMAX_H
#define BLT_OPS_SOFTMAX_H

#ifdef __cplusplus
extern "C" {
#endif
#include "core/tensor.h"
void blt_softmax(const blt_tensor *in, blt_tensor *out);
void blt_softmax_backward(const blt_tensor *grad_out, const blt_tensor *softmax_out, blt_tensor *grad_in);
#ifdef __cplusplus
}
#endif
#endif
