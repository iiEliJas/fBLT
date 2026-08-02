#include <stdio.h>
#include "blt/core/tensor.h"

int run_tensor_core_tests(void) {
    size_t shape[3] = {2, 3, 4};
    size_t strides[BLT_MAX_NDIM] = {0};

    size_t numel = blt_tensor_compute_numel(shape, 3);
    if (numel != 24) {
        fprintf(stderr, "[FAIL] tensor numel computation\n");
        return 0;
    }

    blt_tensor_compute_row_major_strides(shape, 3, strides);
    if (strides[0] != 12 || strides[1] != 4 || strides[2] != 1) {
        fprintf(stderr, "[FAIL] tensor stride computation\n");
        return 0;
    }

    blt_tensor tensor = {0};
    tensor.dtype = BLT_DTYPE_FP32;
    tensor.numel = numel;
    if (blt_tensor_bytes(&tensor) != 96) {
        fprintf(stderr, "[FAIL] tensor byte size computation\n");
        return 0;
    }

    return 1;
}
