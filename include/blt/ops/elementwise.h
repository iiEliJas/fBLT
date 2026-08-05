#ifndef BLT_OPS_ELEMENTWISE_H
#define BLT_OPS_ELEMENTWISE_H
#include "blt/core/tensor.h"
void blt_add(const blt_tensor* a, const blt_tensor* b, blt_tensor* out);
void blt_mul(const blt_tensor* a, const blt_tensor* b, blt_tensor* out);
void blt_scale(blt_tensor* t, float scalar);
#endif
