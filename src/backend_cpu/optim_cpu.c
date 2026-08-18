#include "blt/core/backend.h"
#include "blt/core/tensor.h"
#include "blt/ops/optim.h"


//------------------------------------------------------------
// Simple SGD step

// param -= lr * grad elementwise. CPU only
void blt_sgd_step(blt_tensor* param, const blt_tensor* grad, float lr) {
    blt_check_elementwise_fp32(param, grad,
        "blt_sgd_step: param and grad must be FP32 and elementwise-compatible");
    BLT_REQUIRE(param->backend == BLT_BACKEND_CPU,
        "blt_sgd_step: only the CPU backend is supported");
 
    float* p = (float*)param->data;
    const float* g = (const float*)grad->data;
 
    for (size_t i = 0; i < param->numel; ++i) {
        p[i] -= lr * g[i];
    }
}