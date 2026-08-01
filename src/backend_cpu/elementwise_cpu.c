#include "blt/core/backend.h"
#include "blt/ops/elementwise.h"

#include <string.h>

static void validate_same_shape_and_dtype(const blt_tensor* a, const blt_tensor* b, const blt_tensor* out) {
    if (a->numel != b->numel || b->numel != out->numel) {
        BLT_FATAL("elementwise tensors must have equal numel");
    }
    if (a->dtype != BLT_DTYPE_FP32 || b->dtype != BLT_DTYPE_FP32 || out->dtype != BLT_DTYPE_FP32) {
        BLT_FATAL("elementwise ops only support FP32 tensors");
    }
}

void blt_add_cpu(const blt_tensor* a, const blt_tensor* b, blt_tensor* out) {
    validate_same_shape_and_dtype(a, b, out);
    const float* a_data = (const float*)a->data;
    const float* b_data = (const float*)b->data;
    float* out_data = (float*)out->data;
    for (size_t i = 0; i < out->numel; ++i) {
        out_data[i] = a_data[i] + b_data[i];
    }
}

void blt_mul_cpu(const blt_tensor* a, const blt_tensor* b, blt_tensor* out) {
    validate_same_shape_and_dtype(a, b, out);
    const float* a_data = (const float*)a->data;
    const float* b_data = (const float*)b->data;
    float* out_data = (float*)out->data;
    for (size_t i = 0; i < out->numel; ++i) {
        out_data[i] = a_data[i] * b_data[i];
    }
}
