#include "core/allocator.h"
#include "core/backend.h"
#include "core/tensor.h"

#include <cuda_runtime.h>
#include <cstdio>

// Device-side sanity kernel used only by bin/cuda_smoke.
__global__ void blt_smoke_scale_kernel(float *x, size_t n, float s) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] *= s;
    }
}

static int blt_cuda_smoke_check(blt_arena *host, blt_arena *dev) {
    const size_t N = 1024;
    const size_t shape[1] = {N};
    float ref[1024];
    for (size_t i = 0; i < N; i++) {
        ref[i] = (float)i;
    }

    blt_tensor t_cpu = blt_tensor_create(host, shape, 1, BLT_DTYPE_FP32);
    memcpy(t_cpu.data, ref, N * sizeof(float));

    // Fresh device tensors must come up zeroed.
    blt_tensor t_zero = blt_tensor_create(dev, shape, 1, BLT_DTYPE_FP32);
    float probe = 1.0f;
    cudaMemcpy(&probe, t_zero.data, sizeof(float), cudaMemcpyDeviceToHost);
    if (probe != 0.0f) {
        fprintf(stderr, "[cuda-smoke] device tensor not zero-initialized\n");
        return 1;
    }

    blt_tensor t_dev = blt_tensor_to_device(&t_cpu, dev);

    const float scale = 2.0f;
    unsigned blocks = (unsigned)((N + 255) / 256);
    blt_smoke_scale_kernel<<<blocks, 256>>>((float *)t_dev.data, N, scale);
    if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
        fprintf(stderr, "[cuda-smoke] kernel launch failed\n");
        return 2;
    }

    blt_tensor t_out = blt_tensor_to_host(&t_dev, host);
    for (size_t i = 0; i < N; i++) {
        if (((const float *)t_out.data)[i] != scale * ref[i]) {
            fprintf(stderr, "[cuda-smoke] mismatch at %zu: got %f want %f\n", i, ((const float *)t_out.data)[i],
                    scale * ref[i]);
            return 3;
        }
    }
    printf("[cuda-smoke] H2D -> kernel -> D2H round trip OK (%zu elements)\n", N);
    return 0;
}

extern "C" int blt_cuda_smoke_run(void) {
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
        fprintf(stderr, "[cuda-smoke] cannot query device properties\n");
        return 2;
    }
    printf("[cuda-smoke] device: %s (sm_%d%d, %.1f MB)\n", prop.name, prop.major, prop.minor,
           prop.totalGlobalMem / (1024.0 * 1024.0));

    blt_arena *host = blt_arena_create(1 << 20, BLT_BACKEND_CPU);
    blt_arena *dev = blt_arena_create(1 << 20, BLT_BACKEND_CUDA);
    printf("[cuda-smoke] arenas created (host %zu B, device %zu B)\n", host->capacity, dev->capacity);

    int rc = blt_cuda_smoke_check(host, dev);

    blt_arena_destroy(dev);
    blt_arena_destroy(host);
    return rc;
}
