#ifndef BLT_OPS_MATMUL_H
#define BLT_OPS_MATMUL_H

#ifdef __cplusplus
extern "C" {
#endif
#include "core/tensor.h"
void blt_matmul(const blt_tensor *a, const blt_tensor *b, blt_tensor *out);
void blt_matmul_backward(const blt_tensor *a, const blt_tensor *b, const blt_tensor *grad_out, blt_tensor *grad_a,
                         blt_tensor *grad_b);
#ifdef __cplusplus
}
#endif
#endif
