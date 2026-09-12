#ifndef BLT_TEST_HELPERS_H
#define BLT_TEST_HELPERS_H

#include <stdio.h>

#include "blt/core/tensor.h"
#include "blt/core/allocator.h"

// Helper
int blt_test_load_binary_tensor(const char *path, blt_arena *arena, blt_tensor *out_tensor);
int blt_test_check_close(const blt_tensor *actual, const blt_tensor *expected, float tol);

#define load_binary_tensor blt_test_load_binary_tensor
#define check_close blt_test_check_close

// Assertion Macros
#define TEST_ASSERT(cond)                                                                                              \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            fprintf(stderr, "  [FAIL] %s:%d: Assertion failed: %s\n", __FILE__, __LINE__, #cond);                      \
            return 0;                                                                                                  \
        }                                                                                                              \
    } while (0)

#define TEST_ASSERT_CLOSE(actual, expected, tol)                                                                       \
    do {                                                                                                               \
        if (!blt_test_check_close((actual), (expected), (tol))) {                                                      \
            fprintf(stderr, "  [FAIL] %s:%d: Tensor values out of tolerance (tol=%.1e)\n", __FILE__, __LINE__,         \
                    (double)(tol));                                                                                    \
            return 0;                                                                                                  \
        }                                                                                                              \
    } while (0)

#endif