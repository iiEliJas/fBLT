#ifndef BLT_OPS_VECMATH_H
#define BLT_OPS_VECMATH_H

#include <math.h>
#include <stddef.h>
#include <stdbool.h>
#include "blt/core/tensor.h"

// sum(a[i] * b[i]) for i in [0, n).
float blt_vec_dot(blt_backend backend, const float *a, const float *b, size_t n);

// Numerically stable softmax over a single FP32 row, in place.
//
// For each column col:
//   - if mask_row != NULL:  row[col] = row[col]*scale + mask_row[col]
//     (mask_row holds additive mask values, 0 or -INFINITY)
//   - else if is_causal and col > row_idx: row[col] = -INFINITY
//   - else:                 row[col] *= scale
//
// Non-finite entries after masking are zeroed; if every entry is non-finite
// the row is left all-zero.
void blt_softmax_masked_row_inplace(blt_backend backend, float *row, size_t row_len, size_t row_idx, bool is_causal,
                                    const float *mask_row, float scale);

// Copies a rows x cols tile between strided fp32 buffers (row strides in
// elements). Used for QKV head-slice gather and cache-row scatter.
void blt_strided_copy(blt_backend backend, float *dst, size_t dst_stride, const float *src, size_t src_stride,
                      size_t rows, size_t cols);

void blt_fill_uniform(blt_backend backend, float *data, size_t n, uint64_t *rng_state);
void blt_fill_constant(blt_backend backend, float *data, size_t n, float v);

#endif // BLT_OPS_VECMATH_H
