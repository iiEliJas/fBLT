#include "blt/core/backend.h"
#include "blt/ops/cast.h"

#include <cuda_runtime.h>
#include <cuda_bf16.h>

static int blt_cuda_numel(const blt_tensor *t) { return (int)t->numel; }

static __global__ void cast_fp32_to_bf16_kernel(const float *in, __nv_bfloat16 *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2bfloat16(in[i]);
}

static __global__ void cast_bf16_to_fp32_kernel(const __nv_bfloat16 *in, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __bfloat162float(in[i]);
}

extern "C" void blt_cast_cuda(const blt_tensor *in, blt_tensor *out) {
    BLT_REQUIRE(in != NULL && out != NULL, "blt_cast: in and out must not be NULL");
    BLT_REQUIRE(in->numel == out->numel, "blt_cast: in and out must have the same numel");

    int n = blt_cuda_numel(in);
    if (n == 0) return;

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    if (in->dtype == BLT_DTYPE_FP32 && out->dtype == BLT_DTYPE_BF16) {
        cast_fp32_to_bf16_kernel<<<blocks, threads>>>((const float *)in->data, (__nv_bfloat16 *)out->data, n);
        return;
    }

    if (in->dtype == BLT_DTYPE_BF16 && out->dtype == BLT_DTYPE_FP32) {
        cast_bf16_to_fp32_kernel<<<blocks, threads>>>((const __nv_bfloat16 *)in->data, (float *)out->data, n);
        return;
    }

    BLT_FATAL("blt_cast_cuda: unsupported conversion from dtype %d to %d", in->dtype, out->dtype);
}
