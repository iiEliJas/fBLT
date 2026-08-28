#ifndef BLT_OPS_CAST_H
#define BLT_OPS_CAST_H

#ifdef __cplusplus
extern "C" {
#endif

#include "blt/core/tensor.h"

// Elementwise dtype conversion. in and out must have the same shape and
// numel. Supported conversions: FP32 <-> BF16.
void blt_cast(const blt_tensor* in, blt_tensor* out);

#ifdef __cplusplus
}
#endif
#endif
