#ifndef BLT_TEST_HELPERS_H
#define BLT_TEST_HELPERS_H

#include <stdio.h>

#include "core/tensor.h"
#include "core/allocator.h"

// Configurable base directory for parity test golden files.
// Set in test_main.c; override with a single line change to redirect all
// parity data loading to a different directory tree.
extern const char *g_test_data_dir;

// Build "<g_test_data_dir>/<subpath>" into buf.  Returns buf.
char *blt_test_data_path(char *buf, size_t bufsz, const char *subpath);

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