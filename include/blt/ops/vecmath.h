#ifndef BLT_OPS_VECMATH_H
#define BLT_OPS_VECMATH_H

#include <math.h>
#include <stddef.h>
#include <stdbool.h>
 
/*
 * blt_vec_dot: dot product of two contiguous FP32 vectors
 *
 * Input:  two pointers to arrays of n fp32 values
 * Output: the scalar dot product sum(a[i] * b[i]) for i in [0, n)
 *
 * This is a raw-pointer utility, not a blt_tensor op
 */
static inline float blt_vec_dot(const float* a, const float* b, size_t n) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

/*
 * blt_softmax_masked_row_inplace: numerically stable softmax over a single
 * contiguous FP32 row of length row_len.
 *
 * For each column col:
 *   - if mask_row != NULL:  row[col] = row[col]*scale + mask_row[col]
 *     (mask_row is expected to hold additive mask values, 0 or -INFINITY)
 *   - else if is_causal and col > row_idx: row[col] = -INFINITY
 *   - else:                 row[col] *= scale
 *
 * Non-finite entries after masking are zeroed so they contribute nothing to
 * the sum; if every entry is non-finite the row is left all-zero.
 *
 * This is a raw-pointer utility, not a blt_tensor op
 */
static inline void blt_softmax_masked_row_inplace(
    float* row, size_t row_len, size_t row_idx,
    bool is_causal, const float* mask_row, float scale) {
    float max_val = -INFINITY;

    for (size_t col = 0; col < row_len; ++col) {
        if (mask_row != NULL) {
            row[col] = row[col] * scale + mask_row[col];
        } else if (is_causal && col > row_idx) {
            row[col] = -INFINITY;
        } else {
            row[col] *= scale;
        }

        if (isfinite(row[col]) && row[col] > max_val) {
            max_val = row[col];
        }
    }

    float sum = 0.0f;
    for (size_t col = 0; col < row_len; ++col) {
        if (isfinite(row[col])) {
            row[col] = expf(row[col] - max_val);
            sum += row[col];
        } else {
            row[col] = 0.0f;
        }
    }

    if (sum > 0.0f) {
        for (size_t col = 0; col < row_len; ++col) {
            row[col] /= sum;
        }
    }
}

#endif // BLT_OPS_VECMATH_H