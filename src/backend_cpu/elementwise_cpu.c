#include "blt/core/backend.h"
#include "blt/ops/gelu.h"
#include "blt/ops/elementwise.h"

#include <string.h>
#include <math.h>

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


void blt_scale(blt_tensor* t, float scalar) {
    if (t == NULL) {
        BLT_FATAL("blt_scale: tensor must not be null");
    }
    if (t->dtype != BLT_DTYPE_FP32) {
        BLT_FATAL("blt_scale: only supports FP32 tensors");
    }
 
    float* data = (float*)t->data;
    for (size_t i = 0; i < t->numel; ++i) {
        data[i] *= scalar;
    }
}


// -------------------------------------------------------------
// GELU CPU Implementation

void blt_gelu_forward_cpu(const blt_tensor* x, blt_tensor* out) {
    blt_check_elementwise_fp32(x, out, "GELU: input/output must be FP32 with matching element count");

    const float* in = (const float*)x->data;
    float* o = (float*)out->data;

    for (size_t i = 0; i < x->numel; i++) {
        float v = in[i];
        float v3 = v * v * v;
        float inner = 0.7978845608f * (v + 0.044715f * v3);
        o[i] = 0.5f * v * (1.0f + tanhf(inner));
    }
}

void blt_gelu_backward_cpu(const blt_tensor* grad_out, const blt_tensor* x, blt_tensor* grad_x) {
    (void)grad_out;
    (void)x;
    (void)grad_x;
    BLT_FATAL("GELU backward not yet implemented");
}
