#ifndef BLT_CORE_BACKEND_H
#define BLT_CORE_BACKEND_H

#include <stdio.h>
#include <stdlib.h>
#include "blt/core/tensor.h"
#include "blt/core/dtype.h"

#define BLT_FATAL(msg) do { \
    fprintf(stderr, "BLT Fatal Error [%s:%d]: %s\n", __FILE__, __LINE__, msg); \
    exit(EXIT_FAILURE); \
} while (0)

#define BLT_WARN(msg, ...) do { \
    fprintf(stderr, "BLT Warning [%s:%d]: " msg "\n", __FILE__, __LINE__, ##__VA_ARGS__); \
} while (0)

#define BLT_REQUIRE(cond, msg) do { if (!(cond)) { BLT_FATAL(msg); } } while (0)
 


// Validates a 2D FP32 tensor. Pass 0 for dim0/dim1 to skip that dimension's
// check (useful when the size is being *read out* of the tensor rather than
// checked against a known value, e.g. inferring seq_len from input).
static inline void blt_check_2d_fp32(const blt_tensor* t, size_t dim0, size_t dim1, const char* msg) {
    BLT_REQUIRE(t != NULL, msg);
    BLT_REQUIRE(t->dtype == BLT_DTYPE_FP32, msg);
    BLT_REQUIRE(t->ndim == 2, msg);
    if (dim0 != 0) BLT_REQUIRE(t->shape[0] == dim0, msg);
    if (dim1 != 0) BLT_REQUIRE(t->shape[1] == dim1, msg);
}
 
// Validates a 1D FP32 tensor of exact length dim0.
static inline void blt_check_1d_fp32(const blt_tensor* t, size_t dim0, const char* msg) {
    BLT_REQUIRE(t != NULL, msg);
    BLT_REQUIRE(t->dtype == BLT_DTYPE_FP32, msg);
    BLT_REQUIRE(t->ndim == 1, msg);
    BLT_REQUIRE(t->shape[0] == dim0, msg);
}
 
// Validates that two tensors are FP32 and elementwise-compatible (same
// element count), regardless of rank.
static inline void blt_check_elementwise_fp32(const blt_tensor* a, const blt_tensor* b, const char* msg) {
    BLT_REQUIRE(a != NULL && b != NULL, msg);
    BLT_REQUIRE(a->dtype == BLT_DTYPE_FP32 && b->dtype == BLT_DTYPE_FP32, msg);
    BLT_REQUIRE(a->numel == b->numel, msg);
}


#endif // BLT_CORE_BACKEND_H
