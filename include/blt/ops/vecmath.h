#ifndef BLT_OPS_VECMATH_H
#define BLT_OPS_VECMATH_H

#include <stddef.h>
 
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

#endif // BLT_OPS_VECMATH_H