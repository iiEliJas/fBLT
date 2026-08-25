#include "blt/core/backend.h"
#include "blt/ops/gelu.h"
#include "blt/ops/swiglu.h"
#include "blt/ops/elementwise.h"

#include <cuda_runtime.h>

// Grid-stride elementwise kernels; fp32 math mirrors the CPU reference
// expression-for-expression so results agree to within transcendental ulps.

__global__ void blt_add_kernel(const float* a, const float* b, float* out, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < n; i += stride) {
        out[i] = a[i] + b[i];
    }
}

__global__ void blt_mul_kernel(const float* a, const float* b, float* out, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < n; i += stride) {
        out[i] = a[i] * b[i];
    }
}

__global__ void blt_scale_kernel(float* t, float scalar, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < n; i += stride) {
        t[i] *= scalar;
    }
}

__global__ void blt_gelu_forward_kernel(const float* x, float* out, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < n; i += stride) {
        float v = x[i];
        float v3 = v * v * v;
        float inner = 0.7978845608f * (v + 0.044715f * v3);
        out[i] = 0.5f * v * (1.0f + tanhf(inner));
    }
}

__global__ void blt_gelu_backward_kernel(const float* grad_out, const float* x, float* grad_x, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    const float k0 = 0.7978845608f;
    const float k1 = 0.044715f;
    for (; i < n; i += stride) {
        float xi = x[i];
        float x3 = xi * xi * xi;
        float inner = k0 * (xi + k1 * x3);
        float t = tanhf(inner);
        float sech2 = 1.0f - t * t;
        float dinner_dx = k0 * (1.0f + 3.0f * k1 * xi * xi);
        float dgelu_dx = 0.5f * (1.0f + t) + 0.5f * xi * sech2 * dinner_dx;
        grad_x[i] = grad_out[i] * dgelu_dx;
    }
}

__global__ void blt_swiglu_forward_kernel(const float* gate, const float* up, float* out, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < n; i += stride) {
        float gv = gate[i];
        float silu = gv / (1.0f + expf(-gv));
        out[i] = silu * up[i];
    }
}

__global__ void blt_swiglu_backward_kernel(const float* grad_out, const float* gate, const float* up,
                                           float* grad_gate, float* grad_up, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < n; i += stride) {
        float gv = gate[i];
        float sig = 1.0f / (1.0f + expf(-gv));
        float silu = gv * sig;
        float dsilu_dgate = sig * (1.0f + gv * (1.0f - sig));
        grad_gate[i] = grad_out[i] * up[i] * dsilu_dgate;
        grad_up[i] = grad_out[i] * silu;
    }
}

static int blt_cuda_sync_check(const char* what) {
    cudaError_t err = cudaGetLastError();
    if (err == cudaSuccess) {
        err = cudaDeviceSynchronize();
    }
    if (err != cudaSuccess) {
        BLT_FATAL("%s failed: %s", what, cudaGetErrorString(err));
    }
    return 1;
}

static unsigned blt_cuda_grid(size_t n) {
    const unsigned block = 256;
    size_t blocks = (n + block - 1) / block;
    if (blocks > 4096) blocks = 4096;
    return (unsigned)blocks;
}

extern "C" void blt_add_cuda(const blt_tensor* a, const blt_tensor* b, blt_tensor* out) {
    blt_add_kernel<<<blt_cuda_grid(out->numel), 256>>>(
        (const float*)a->data, (const float*)b->data, (float*)out->data, out->numel);
    blt_cuda_sync_check("blt_add");
}

extern "C" void blt_mul_cuda(const blt_tensor* a, const blt_tensor* b, blt_tensor* out) {
    blt_mul_kernel<<<blt_cuda_grid(out->numel), 256>>>(
        (const float*)a->data, (const float*)b->data, (float*)out->data, out->numel);
    blt_cuda_sync_check("blt_mul");
}

extern "C" void blt_scale_cuda(blt_tensor* t, float scalar) {
    blt_scale_kernel<<<blt_cuda_grid(t->numel), 256>>>((float*)t->data, scalar, t->numel);
    blt_cuda_sync_check("blt_scale");
}

extern "C" void blt_gelu_forward_cuda(const blt_tensor* x, blt_tensor* out) {
    blt_gelu_forward_kernel<<<blt_cuda_grid(x->numel), 256>>>(
        (const float*)x->data, (float*)out->data, x->numel);
    blt_cuda_sync_check("blt_gelu_forward");
}

extern "C" void blt_gelu_backward_cuda(const blt_tensor* grad_out, const blt_tensor* x, blt_tensor* grad_x) {
    blt_gelu_backward_kernel<<<blt_cuda_grid(x->numel), 256>>>(
        (const float*)grad_out->data, (const float*)x->data, (float*)grad_x->data, x->numel);
    blt_cuda_sync_check("blt_gelu_backward");
}

extern "C" void blt_swiglu_forward_cuda(const blt_tensor* gate, const blt_tensor* up, blt_tensor* out) {
    blt_swiglu_forward_kernel<<<blt_cuda_grid(gate->numel), 256>>>(
        (const float*)gate->data, (const float*)up->data, (float*)out->data, gate->numel);
    blt_cuda_sync_check("blt_swiglu_forward");
}

extern "C" void blt_swiglu_backward_cuda(const blt_tensor* grad_out, const blt_tensor* gate,
                                         const blt_tensor* up, blt_tensor* grad_gate, blt_tensor* grad_up) {
    blt_swiglu_backward_kernel<<<blt_cuda_grid(gate->numel), 256>>>(
        (const float*)grad_out->data, (const float*)gate->data, (const float*)up->data,
        (float*)grad_gate->data, (float*)grad_up->data, gate->numel);
    blt_cuda_sync_check("blt_swiglu_backward");
}
