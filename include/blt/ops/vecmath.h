#ifndef BLT_OPS_VECMATH_H
#define BLT_OPS_VECMATH_H

#include <math.h>
#include <stddef.h>
#include <stdbool.h>
#include "blt/core/tensor.h"

// Raw-pointer utilities dispatched by backend enum rather than tensor handle:
// callers pass the backend of the buffers they hand over (CPU pointers or
// CUDA device pointers).

// blt_vec_dot: dot product of two contiguous FP32 vectors.
// Input:  two pointers to arrays of n fp32 values
// Output: the scalar dot product sum(a[i] * b[i]) for i in [0, n)
float blt_vec_dot(blt_backend backend, const float* a, const float* b, size_t n);

// blt_softmax_masked_row_inplace: numerically stable softmax over a single
// contiguous FP32 row of length row_len, in place.
//
// For each column col:
//   - if mask_row != NULL:  row[col] = row[col]*scale + mask_row[col]
//     (mask_row is expected to hold additive mask values, 0 or -INFINITY)
//   - else if is_causal and col > row_idx: row[col] = -INFINITY
//   - else:                 row[col] *= scale
//
// Non-finite entries after masking are zeroed so they contribute nothing to
// the sum; if every entry is non-finite the row is left all-zero.
void blt_softmax_masked_row_inplace(
    blt_backend backend, float* row, size_t row_len, size_t row_idx,
    bool is_causal, const float* mask_row, float scale);

// blt_strided_copy: copies a rows x cols element tile between strided fp32
// buffers (row strides in elements). Handles the QKV head-slice gather and
// cache-row scatter patterns that used to be memcpy loops in model code.
void blt_strided_copy(blt_backend backend, float* dst, size_t dst_stride,
                      const float* src, size_t src_stride,
                      size_t rows, size_t cols);

#endif // BLT_OPS_VECMATH_H
