#include "blt/core/backend.h"
#include "blt/core/cuda_shim.h"
#include "blt/ops/patch_pool.h"

#include <cuda_runtime.h>
#include <string.h>

static int blt_cuda_launch_check(const char *what) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        BLT_FATAL("%s failed: %s", what, cudaGetErrorString(err));
    }
    return 1;
}

// Upload host patch table to device using scratch arena.
static void *blt_cuda_upload_patches(const blt_patch_info *patches, size_t num_patches) {
    if (num_patches == 0) return NULL;
    size_t bytes = num_patches * sizeof(blt_patch_info);
    void *dev_ptr = blt_arena_alloc(blt_cuda_get_scratch_arena(), bytes, 16);
    blt_cuda_memcpy_h2d(dev_ptr, patches, bytes);
    return dev_ptr;
}

__global__ void blt_patch_pool_forward_kernel(const float *h, const blt_patch_info *patches, float *out,
                                              size_t num_patches, size_t embed_dim, int pool_max) {
    for (size_t j = (size_t)blockIdx.x * blockDim.x + threadIdx.x; j < num_patches;
         j += (size_t)gridDim.x * blockDim.x) {
        const size_t start = patches[j].start_idx;
        const size_t len = patches[j].length;
        float *dst = out + j * embed_dim;
        const float *src0 = h + start * embed_dim;

        if (!pool_max) {
            for (size_t d = 0; d < embed_dim; d++) dst[d] = 0.0f;
            for (size_t i = 0; i < len; i++) {
                const float *src = src0 + i * embed_dim;
                for (size_t d = 0; d < embed_dim; d++) dst[d] += src[d];
            }
            const float inv_len = 1.0f / (float)len;
            for (size_t d = 0; d < embed_dim; d++) dst[d] *= inv_len;
        } else {
            memcpy(dst, src0, embed_dim * sizeof(float));
            for (size_t i = 1; i < len; i++) {
                const float *src = src0 + i * embed_dim;
                for (size_t d = 0; d < embed_dim; d++)
                    if (src[d] > dst[d]) dst[d] = src[d];
            }
        }
    }
}

extern "C" void blt_patch_pool_forward_cuda(const blt_tensor *byte_hidden, const blt_patch_info *patches,
                                            size_t num_patches, blt_patch_pool_type pool_type, blt_tensor *out) {
    const size_t seq_len = byte_hidden->shape[0];
    const size_t embed_dim = byte_hidden->shape[1];
    BLT_REQUIRE(pool_type == BLT_POOL_MEAN || pool_type == BLT_POOL_MAX, "patch_pool: unknown pool_type");

    for (size_t j = 0; j < num_patches; j++) {
        BLT_REQUIRE(patches[j].length >= 1, "patch_pool: empty patch");
        BLT_REQUIRE(patches[j].start_idx + patches[j].length <= seq_len, "patch_pool: patch out of range");
    }

    blt_patch_info *d_patches = (blt_patch_info *)blt_cuda_upload_patches(patches, num_patches);
    blt_patch_pool_forward_kernel<<<(unsigned)(num_patches > 4096 ? 4096 : (num_patches > 0 ? num_patches : 1)), 256>>>(
        (const float *)byte_hidden->data, d_patches, (float *)out->data, num_patches, embed_dim,
        pool_type == BLT_POOL_MAX ? 1 : 0);
    blt_cuda_launch_check("patch_pool forward");
}

__global__ void blt_patch_pool_backward_kernel(const float *g, const float *h, const blt_patch_info *patches, float *gh,
                                               size_t num_patches, size_t seq_len, size_t embed_dim, int pool_max) {
    if (!pool_max) {
        // Mean: one thread per patch; rows are disjoint across patches.
        for (size_t j = (size_t)blockIdx.x * blockDim.x + threadIdx.x; j < num_patches;
             j += (size_t)gridDim.x * blockDim.x) {
            const size_t start = patches[j].start_idx;
            const size_t len = patches[j].length;
            const float *gj = g + j * embed_dim;
            const float inv_len = 1.0f / (float)len;
            for (size_t i = 0; i < len; i++) {
                float *dst = gh + (start + i) * embed_dim;
                for (size_t d = 0; d < embed_dim; d++) dst[d] += gj[d] * inv_len;
            }
        }
    } else {
        // Max: one thread per (patch, channel); recompute argmax exactly
        // like the CPU reference.
        const size_t work = num_patches * embed_dim;
        for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x; idx < work;
             idx += (size_t)gridDim.x * blockDim.x) {
            const size_t j = idx / embed_dim;
            const size_t d = idx % embed_dim;
            const size_t start = patches[j].start_idx;
            const size_t len = patches[j].length;
            size_t argmax = start;
            float best = h[start * embed_dim + d];
            for (size_t i = 1; i < len; i++) {
                const float v = h[(start + i) * embed_dim + d];
                if (v > best) {
                    best = v;
                    argmax = start + i;
                }
            }
            gh[argmax * embed_dim + d] += g[j * embed_dim + d];
        }
    }
}

extern "C" void blt_patch_pool_backward_cuda(const blt_tensor *grad_out, const blt_tensor *byte_hidden,
                                             const blt_patch_info *patches, size_t num_patches,
                                             blt_patch_pool_type pool_type, blt_tensor *grad_byte_hidden) {
    const size_t seq_len = byte_hidden->shape[0];
    const size_t embed_dim = byte_hidden->shape[1];
    BLT_REQUIRE(pool_type == BLT_POOL_MEAN || pool_type == BLT_POOL_MAX, "patch_pool_bw: unknown pool_type");

    for (size_t j = 0; j < num_patches; j++) {
        BLT_REQUIRE(patches[j].length >= 1, "patch_pool_bw: empty patch");
        BLT_REQUIRE(patches[j].start_idx + patches[j].length <= seq_len, "patch_pool_bw: patch out of range");
    }

    blt_patch_info *d_patches = (blt_patch_info *)blt_cuda_upload_patches(patches, num_patches);
    blt_patch_pool_backward_kernel<<<(unsigned)(seq_len > 4096 ? 4096 : (seq_len > 0 ? seq_len : 1)), 256>>>(
        (const float *)grad_out->data, (const float *)byte_hidden->data, d_patches, (float *)grad_byte_hidden->data,
        num_patches, seq_len, embed_dim, pool_type == BLT_POOL_MAX ? 1 : 0);
    blt_cuda_launch_check("patch_pool backward");
}
