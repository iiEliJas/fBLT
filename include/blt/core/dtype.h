#ifndef BLT_CORE_DTYPE_H
#define BLT_CORE_DTYPE_H

#include <stddef.h>
#include <stdint.h>

typedef enum {
    BLT_DTYPE_FP32 = 0,
    BLT_DTYPE_BF16 = 1,
    BLT_DTYPE_INT32 = 2,
    BLT_DTYPE_UINT8 = 3
} blt_dtype;

static inline size_t blt_dtype_sizeof(blt_dtype dtype) {
    switch (dtype) {
        case BLT_DTYPE_FP32:  return 4;
        case BLT_DTYPE_BF16:  return 2;
        case BLT_DTYPE_INT32: return 4;
        case BLT_DTYPE_UINT8: return 1;
        default: return 0;
    }
}

#endif // BLT_CORE_DTYPE_H
