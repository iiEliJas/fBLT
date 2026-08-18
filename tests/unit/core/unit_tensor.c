#include <stdio.h>
#include "blt/core/tensor.h"

#include "test_helpers.h"
#include "test_suite.h"

int run_tensor_core_tests(void) {
    size_t shape[3] = {2, 3, 4};
    size_t strides[BLT_MAX_NDIM] = {0};

    size_t numel = blt_tensor_compute_numel(shape, 3);
    TEST_ASSERT(numel == 24);

    blt_tensor_compute_row_major_strides(shape, 3, strides);
    TEST_ASSERT(strides[0] == 12);
    TEST_ASSERT(strides[1] == 4);
    TEST_ASSERT(strides[2] == 1);

    blt_tensor tensor = {0};
    tensor.dtype = BLT_DTYPE_FP32;
    tensor.numel = numel;
    TEST_ASSERT(blt_tensor_bytes(&tensor) == 96);

    return 1;
}
