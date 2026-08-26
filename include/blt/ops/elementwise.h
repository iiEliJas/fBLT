#ifndef BLT_OPS_ELEMENTWISE_H
#define BLT_OPS_ELEMENTWISE_H

#ifdef __cplusplus
extern "C" {
#endif
#include "blt/core/tensor.h"
void blt_add(const blt_tensor* a, const blt_tensor* b, blt_tensor* out);
void blt_mul(const blt_tensor* a, const blt_tensor* b, blt_tensor* out);
void blt_scale(blt_tensor* t, float scalar);
void blt_scaled_copy(blt_tensor* dst, const blt_tensor* src, float scalar);
#ifdef __cplusplus
}
#endif
#endif
