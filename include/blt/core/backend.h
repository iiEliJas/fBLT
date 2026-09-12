#ifndef BLT_CORE_BACKEND_H
#define BLT_CORE_BACKEND_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdlib.h>
#include "blt/core/tensor.h"
#include "blt/core/dtype.h"

#define BLT_FATAL(msg, ...)                                                                                            \
    do {                                                                                                               \
        fprintf(stderr, "BLT Fatal Error [%s:%d]: " msg "\n", __FILE__, __LINE__, ##__VA_ARGS__);                      \
        exit(EXIT_FAILURE);                                                                                            \
    } while (0)

#define BLT_WARN(msg, ...)                                                                                             \
    do {                                                                                                               \
        fprintf(stderr, "BLT Warning [%s:%d]: " msg "\n", __FILE__, __LINE__, ##__VA_ARGS__);                          \
    } while (0)

#define BLT_REQUIRE(cond, msg, ...)                                                                                    \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            fprintf(stderr, "BLT Fatal Error [%s:%d]: Requirement '%s' failed: " msg "\n", __FILE__, __LINE__,         \
                    #cond, ##__VA_ARGS__);                                                                             \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)

// msg must be a string literal
// For runtime strings: BLT_REQUIRE(cond, "%s", runtime_str);

// Validates an N-dimensional FP32 tensor. Pass 0 for any dimension in `dims` to skip that dimension's check.
static inline void blt_check_nd_fp32(const blt_tensor *t, size_t ndim, const size_t *dims, const char *msg) {
    BLT_REQUIRE(t != NULL, "%s", msg);
    BLT_REQUIRE(t->data != NULL, "%s", msg);
    BLT_REQUIRE(t->dtype == BLT_DTYPE_FP32, "%s", msg);
    BLT_REQUIRE(t->ndim == ndim, "%s", msg);
    BLT_REQUIRE(ndim <= BLT_MAX_NDIM, "ndim exceeds BLT_MAX_NDIM");
    for (size_t i = 0; i < ndim; i++) {
        if (dims[i] != 0) BLT_REQUIRE(t->shape[i] == dims[i], "%s", msg);
    }
}

// Validates that two tensors are FP32 and elementwise-compatible (same
// element count), regardless of rank.
static inline void blt_check_elementwise_fp32(const blt_tensor *a, const blt_tensor *b, const char *msg) {
    BLT_REQUIRE(a != NULL && b != NULL, "%s", msg);
    BLT_REQUIRE(a->dtype == BLT_DTYPE_FP32 && b->dtype == BLT_DTYPE_FP32, "%s", msg);
    BLT_REQUIRE(a->numel == b->numel, "%s", msg);
}

// Synchronization point at pass boundaries (forward/backward).
// CPU: no-op. CUDA: flushes stream and checks for kernel execution errors.
void blt_backend_pass_sync(blt_backend backend);

#ifdef __cplusplus
}
#endif

#endif // BLT_CORE_BACKEND_H
