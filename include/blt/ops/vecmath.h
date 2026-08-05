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
float blt_vec_dot(const float* a, const float* b, size_t n);

#endif // BLT_OPS_VECMATH_H