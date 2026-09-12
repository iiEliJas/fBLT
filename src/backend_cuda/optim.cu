#include "blt/core/backend.h"
#include "blt/ops/optim.h"

#include <cuda_runtime.h>
#include <math.h>

// Fused in-place optimizers. One thread per parameter element; bias
// correction terms are precomputed on the host so the kernel does a single
// pass with no cross-element dependencies.

static int blt_cuda_launch_check(const char *what) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        BLT_FATAL("%s failed: %s", what, cudaGetErrorString(err));
    }
    return 1;
}

static unsigned blt_cuda_grid(size_t n) {
    const unsigned block = 256;
    size_t blocks = (n + block - 1) / block;
    if (blocks > 4096) blocks = 4096;
    return (unsigned)(blocks > 0 ? blocks : 1);
}

__global__ void blt_sgd_step_kernel(float *p, const float *g, float lr, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < n; i += stride) {
        p[i] -= lr * g[i];
    }
}

extern "C" void blt_sgd_step_cuda(blt_tensor *param, const blt_tensor *grad, float lr) {
    blt_sgd_step_kernel<<<blt_cuda_grid(param->numel), 256>>>((float *)param->data, (const float *)grad->data, lr,
                                                              param->numel);
    blt_cuda_launch_check("blt_sgd_step");
}

__global__ void blt_adamw_step_kernel(float *p, const float *g, float *m, float *v, size_t n, float b1, float b2,
                                      float bc1, float bc2, float eps, float lr, float wd) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < n; i += stride) {
        const float gi = g[i];
        m[i] = b1 * m[i] + (1.0f - b1) * gi;
        v[i] = b2 * v[i] + (1.0f - b2) * gi * gi;
        const float mhat = m[i] / bc1;
        const float vhat = v[i] / bc2;
        // Same association as the CPU reference so fp32 rounding matches.
        p[i] -= lr * (mhat / (sqrtf(vhat) + eps) + wd * p[i]);
    }
}

extern "C" void blt_adamw_step_cuda(blt_tensor *param, const blt_tensor *grad, blt_tensor *exp_avg,
                                    blt_tensor *exp_avg_sq, const blt_adamw_config *config) {
    BLT_REQUIRE(config != NULL, "blt_adamw_step: config must not be NULL");
    BLT_REQUIRE(config->step >= 1, "blt_adamw_step: step must be >= 1");
    BLT_REQUIRE(config->beta1 > 0.0f && config->beta1 < 1.0f && config->beta2 > 0.0f && config->beta2 < 1.0f,
                "blt_adamw_step: betas must be in (0, 1)");

    const float lr = config->lr;
    const float wd = config->weight_decay;
    const float bc1 = 1.0f - powf(config->beta1, (float)config->step);
    const float bc2 = 1.0f - powf(config->beta2, (float)config->step);

    blt_adamw_step_kernel<<<blt_cuda_grid(param->numel), 256>>>(
        (float *)param->data, (const float *)grad->data, (float *)exp_avg->data, (float *)exp_avg_sq->data,
        param->numel, config->beta1, config->beta2, bc1, bc2, config->eps, lr, wd);
    blt_cuda_launch_check("blt_adamw_step");
}
