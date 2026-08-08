#include "blt/core/backend.h"
#include "blt/ops/gelu.h"
#include "blt/ops/swiglu.h"
#include "blt/ops/elementwise.h"

#include <string.h>
#include <math.h>


void blt_add_cpu(const blt_tensor* a, const blt_tensor* b, blt_tensor* out) {
    blt_check_elementwise_fp32(a, b, "blt_add_cpu: input tensors must be FP32 with matching element count");
    blt_check_elementwise_fp32(a, out, "blt_add_cpu: output tensor must be FP32 with matching element count");
    const float* a_data = (const float*)a->data;
    const float* b_data = (const float*)b->data;
    float* out_data = (float*)out->data;
    for (size_t i = 0; i < out->numel; ++i) {
        out_data[i] = a_data[i] + b_data[i];
    }
}


void blt_mul_cpu(const blt_tensor* a, const blt_tensor* b, blt_tensor* out) {
    blt_check_elementwise_fp32(a, b, "blt_mul_cpu: input tensors must be FP32 with matching element count");
    blt_check_elementwise_fp32(a, out, "blt_mul_cpu: output tensor must be FP32 with matching element count");
    const float* a_data = (const float*)a->data;
    const float* b_data = (const float*)b->data;
    float* out_data = (float*)out->data;
    for (size_t i = 0; i < out->numel; ++i) {
        out_data[i] = a_data[i] * b_data[i];
    }
}


void blt_scale(blt_tensor* t, float scalar) {
    BLT_REQUIRE(t != NULL, "blt_scale: tensor must not be null");
    BLT_REQUIRE(t->dtype == BLT_DTYPE_FP32, "blt_scale: only supports FP32 tensors");
 
    float* data = (float*)t->data;
    for (size_t i = 0; i < t->numel; ++i) {
        data[i] *= scalar;
    }
}



// -------------------------------------------------------------
// GELU

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



// -------------------------------------------------------------
// SwiGLU

void blt_swiglu_forward_cpu(const blt_tensor* gate, const blt_tensor* up, blt_tensor* out) {
    blt_check_elementwise_fp32(gate, up, "SwiGLU: gate/up must be FP32 with matching element count");
    blt_check_elementwise_fp32(gate, out, "SwiGLU: output must be FP32 with matching element count");
 
    const float* g = (const float*)gate->data;
    const float* u = (const float*)up->data;
    float* o = (float*)out->data;
 
    for (size_t i = 0; i < gate->numel; i++) {
        float gv = g[i];
        float silu = gv / (1.0f + expf(-gv));
        o[i] = silu * u[i];
    }
}
 
void blt_swiglu_backward_cpu(const blt_tensor* grad_out, const blt_tensor* gate, const blt_tensor* up,
                              blt_tensor* grad_gate, blt_tensor* grad_up) {
    (void)grad_out; (void)gate; (void)up; (void)grad_gate; (void)grad_up;
    BLT_FATAL("SwiGLU backward not yet implemented");
}
