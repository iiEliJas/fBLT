#ifndef BLT_OPS_SOFTMAX_H
#define BLT_OPS_SOFTMAX_H
#include "blt/core/tensor.h"
void blt_softmax(const blt_tensor* in, blt_tensor* out);
void blt_softmax_backward(const blt_tensor* grad_out, const blt_tensor* softmax_out,
                        blt_tensor* grad_in);
#endif
