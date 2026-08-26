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


void blt_scale_cpu(blt_tensor* t, float scalar) {
    BLT_REQUIRE(t != NULL, "blt_scale: tensor must not be null");
    BLT_REQUIRE(t->dtype == BLT_DTYPE_FP32, "blt_scale: only supports FP32 tensors");
    BLT_REQUIRE(t->backend == BLT_BACKEND_CPU, "blt_scale: CPU implementation called with non-CPU backend");

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
    blt_check_elementwise_fp32(grad_out, x, "GELU backward: grad_out/x must be FP32 with matching element count");
    blt_check_elementwise_fp32(grad_out, grad_x, "GELU backward: grad_out/grad_x must be FP32 with matching element count");
 
    const float* go = (const float*)grad_out->data;
    const float* xd = (const float*)x->data;
    float* gx = (float*)grad_x->data;
    size_t n = x->numel;
 
    const float k0 = 0.7978845608f; // sqrt(2/pi) approx.
    const float k1 = 0.044715f;
 
    for (size_t i = 0; i < n; i++) {
        float xi = xd[i];
        float x3 = xi * xi * xi;
        float inner = k0 * (xi + k1 * x3);
        float t = tanhf(inner);
        float sech2 = 1.0f - t * t;
        float dinner_dx = k0 * (1.0f + 3.0f * k1 * xi * xi);
        // d/dx [0.5*x*(1+tanh(inner))] = 0.5*(1+tanh(inner)) + 0.5*x*sech2*dinner_dx
        float dgelu_dx = 0.5f * (1.0f + t) + 0.5f * xi * sech2 * dinner_dx;
        gx[i] = go[i] * dgelu_dx;
    }
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
    blt_check_elementwise_fp32(gate, up, "SwiGLU backward: gate/up must be FP32 with matching element count");
    blt_check_elementwise_fp32(gate, grad_out, "SwiGLU backward: grad_out must be FP32 with matching element count");
    blt_check_elementwise_fp32(gate, grad_gate, "SwiGLU backward: grad_gate must be FP32 with matching element count");
    blt_check_elementwise_fp32(gate, grad_up, "SwiGLU backward: grad_up must be FP32 with matching element count");
 
    const float* go = (const float*)grad_out->data;
    const float* g = (const float*)gate->data;
    const float* u = (const float*)up->data;
    float* gg = (float*)grad_gate->data;
    float* gu = (float*)grad_up->data;
                                
    // out = silu(gate) * up
    // silu(x) = x * sigmoid(x)
    // d(silu)/dx = sigmoid(x) * (1 + x * (1 - sigmoid(x)))

    // grad_gate = grad_out * up * d(silu)/dgate
    // grad_up   = grad_out * silu(gate)
    for (size_t i = 0; i < gate->numel; i++) {
        float gv = g[i];
        float sig = 1.0f / (1.0f + expf(-gv));
        float silu = gv * sig;
        float dsilu_dgate = sig * (1.0f + gv * (1.0f - sig));
 
        gg[i] = go[i] * u[i] * dsilu_dgate;
        gu[i] = go[i] * silu;
    }
}

void blt_scaled_copy_cpu(blt_tensor* dst, const blt_tensor* src, float scalar) {
    BLT_REQUIRE(dst != NULL && src != NULL, "blt_scaled_copy: tensors must not be null");
    BLT_REQUIRE(dst->dtype == BLT_DTYPE_FP32 && src->dtype == BLT_DTYPE_FP32,
                "blt_scaled_copy: only supports FP32 tensors");
    BLT_REQUIRE(dst->numel == src->numel, "blt_scaled_copy: element count mismatch");
    BLT_REQUIRE(dst->backend == BLT_BACKEND_CPU && src->backend == BLT_BACKEND_CPU,
                "blt_scaled_copy: CPU implementation called with non-CPU tensors");

    const float* s_data = (const float*)src->data;
    float* d_data = (float*)dst->data;
    for (size_t i = 0; i < dst->numel; ++i) {
        d_data[i] = scalar * s_data[i];
    }
}
