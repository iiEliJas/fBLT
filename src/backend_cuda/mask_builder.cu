#include "blt/core/backend.h"
#include "blt/core/cuda_shim.h"
#include "blt/core/allocator.h"
#include "blt/ops/mask_builder.h"

#include <cuda_runtime.h>
#include <math.h>
#include <string.h>

// Mask construction on device: config arrays arrive as host pointers, so
// they are mirrored into temporary device buffers; one thread writes each
// (query, key) cell. Row-validity is reduced to a per-row flag buffer that
// is checked host-side after sync, mirroring the CPU BLT_REQUIRE.

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

static void* blt_cuda_upload_temp(const void* host_data, size_t bytes) {
    void* dev_ptr = NULL;
    cudaError_t err = cudaMalloc(&dev_ptr, bytes > 0 ? bytes : 1);
    if (err != cudaSuccess) {
        BLT_FATAL("mask_builder: temp upload malloc failed: %s", cudaGetErrorString(err));
    }
    if (bytes > 0) {
        blt_cuda_memcpy_h2d(dev_ptr, host_data, bytes);
    }
    return dev_ptr;
}

__global__ void blt_attention_mask_kernel(float* mask, size_t seq_q, size_t seq_kv,
                                          int is_causal, size_t causal_offset,
                                          size_t sliding_window,
                                          const size_t* doc_bounds, size_t num_docs,
                                          const size_t* q_groups, const size_t* kv_groups,
                                          int bidirectional_group) {
    const size_t count = seq_q * seq_kv;
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x; idx < count;
         idx += (size_t)gridDim.x * blockDim.x) {
        const size_t i = idx / seq_kv;
        const size_t j = idx % seq_kv;

        bool allowed = true;
        if (is_causal && j > i + causal_offset) {
            allowed = false;
        }
        if (allowed && sliding_window > 0) {
            if (is_causal) {
                const size_t diag = i + causal_offset;
                size_t back = diag >= j ? diag - j : j - diag;
                if (back >= sliding_window) {
                    allowed = false;
                }
            } else {
                size_t diff = i > j ? i - j : j - i;
                if (diff >= sliding_window) {
                    allowed = false;
                }
            }
        }
        if (allowed && doc_bounds != NULL) {
            // Doc id = number of boundaries at or below the position.
            size_t doc_q = 0;
            for (size_t b = 0; b < num_docs - 1; b++) {
                if (i >= doc_bounds[b]) doc_q++;
                else break;
            }
            size_t doc_k = 0;
            for (size_t b = 0; b < num_docs - 1; b++) {
                if (j >= doc_bounds[b]) doc_k++;
                else break;
            }
            if (doc_q != doc_k) {
                allowed = false;
            }
        }
        if (allowed && q_groups != NULL) {
            if (q_groups[i] != kv_groups[j]) {
                allowed = false;
            } else if (!bidirectional_group && j > i) {
                allowed = false;
            }
        }

        mask[idx] = allowed ? 0.0f : -INFINITY;
    }
}

// A row is valid iff it contains at least one finite (allowed) entry.
__global__ void blt_mask_row_check_kernel(const float* mask, size_t seq_q, size_t seq_kv,
                                          unsigned char* row_ok) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < seq_q;
         i += (size_t)gridDim.x * blockDim.x) {
        const float* row = mask + i * seq_kv;
        bool any = false;
        for (size_t j = 0; j < seq_kv; j++) {
            if (isfinite(row[j])) {
                any = true;
                break;
            }
        }
        row_ok[i] = any ? 1u : 0u;
    }
}

extern "C" void blt_build_attention_mask_cuda(const blt_mask_config* config, blt_tensor* out_mask,
                                              blt_arena* arena) {
    const size_t seq_q = config->seq_len_q;
    const size_t seq_kv = config->seq_len_kv;
    const bool has_docs = (config->doc_boundaries != NULL && config->num_docs > 0);
    const bool has_groups = (config->query_group_ids != NULL && config->kv_group_ids != NULL);

    size_t shape[2] = {seq_q, seq_kv};
    *out_mask = blt_tensor_create(arena, shape, 2, BLT_DTYPE_FP32);

    size_t* d_docs = NULL;
    size_t* d_qg = NULL;
    size_t* d_kvg = NULL;
    unsigned char* d_row_ok = NULL;
    if (has_docs) {
        d_docs = (size_t*)blt_cuda_upload_temp(config->doc_boundaries, config->num_docs * sizeof(size_t));
    }
    if (has_groups) {
        d_qg = (size_t*)blt_cuda_upload_temp(config->query_group_ids, seq_q * sizeof(size_t));
        d_kvg = (size_t*)blt_cuda_upload_temp(config->kv_group_ids, seq_kv * sizeof(size_t));
    }

    cudaError_t err = cudaMalloc(&d_row_ok, seq_q);
    if (err != cudaSuccess) {
        BLT_FATAL("blt_build_attention_mask: row flag malloc failed: %s", cudaGetErrorString(err));
    }

    const size_t count = seq_q * seq_kv;
    const unsigned block = 256;
    size_t blocks = (count + block - 1) / block;
    if (blocks > 4096) blocks = 4096;
    if (blocks == 0) blocks = 1;

    // First pass fills the matrix; second pass flags fully-masked rows.
    blt_attention_mask_kernel<<<(unsigned)blocks, block>>>(
        (float*)out_mask->data, seq_q, seq_kv,
        config->is_causal ? 1 : 0, config->causal_offset, config->sliding_window,
        d_docs, has_docs ? config->num_docs : 0,
        d_qg, d_kvg,
        config->bidirectional_within_group ? 1 : 0);
    err = cudaGetLastError();
    if (err == cudaSuccess) {
        blt_mask_row_check_kernel<<<(unsigned)(seq_q > 4096 ? 4096 : (seq_q > 0 ? seq_q : 1)), block>>>(
            (const float*)out_mask->data, seq_q, seq_kv, d_row_ok);
        err = cudaGetLastError();
    }
    if (err == cudaSuccess) {
        err = cudaDeviceSynchronize();
    }
    if (err != cudaSuccess) {
        BLT_FATAL("blt_build_attention_mask: kernel failed: %s", cudaGetErrorString(err));
    }

    unsigned char* row_ok = (unsigned char*)malloc(seq_q);
    BLT_REQUIRE(row_ok != NULL, "blt_build_attention_mask: failed to allocate row check buffer");
    blt_cuda_memcpy_d2h(row_ok, d_row_ok, seq_q);
    for (size_t i = 0; i < seq_q; i++) {
        // Column 0 is always valid under every supported mask configuration
        // (causal keeps j=0; window keeps |i-j|<w with j=i; docs/groups include self).
        if (!row_ok[i]) {
            free(row_ok);
            cudaFree(d_row_ok);
            if (d_docs) cudaFree(d_docs);
            if (d_qg) cudaFree(d_qg);
            if (d_kvg) cudaFree(d_kvg);
            BLT_FATAL("blt_build_attention_mask: query position has no valid key");
        }
    }
    free(row_ok);

    cudaFree(d_row_ok);
    if (d_docs) cudaFree(d_docs);
    if (d_qg) cudaFree(d_qg);
    if (d_kvg) cudaFree(d_kvg);
}

__global__ void blt_block_diffusion_mask_kernel(float* m, size_t S, size_t N, int infer_mode) {
    const size_t count = S * S;
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x; idx < count;
         idx += (size_t)gridDim.x * blockDim.x) {
        const size_t i = idx / S;
        const size_t j = idx % S;

        bool allowed;
        if (i < N) {
            // Clean prefix: causal among clean rows only; never see blocks.
            allowed = (j < N) && (j <= i);
        } else if (infer_mode) {
            // Single live block: all clean + whole block section.
            allowed = true;
        } else {
            // TRAIN: plain causal over [clean ; blocks].
            allowed = (j <= i);
        }
        m[idx] = allowed ? 0.0f : -INFINITY;
    }
}

extern "C" void blt_build_block_diffusion_mask_cuda(const blt_block_diffusion_config* config,
                                                    blt_tensor* out_mask, blt_arena* arena) {
    const size_t S = config->seq_len;
    const size_t N = config->num_clean;

    size_t shape[2] = {S, S};
    *out_mask = blt_tensor_create(arena, shape, 2, BLT_DTYPE_FP32);

    const size_t count = S * S;
    const unsigned block = 256;
    size_t blocks = (count + block - 1) / block;
    if (blocks > 4096) blocks = 4096;
    if (blocks == 0) blocks = 1;

    blt_block_diffusion_mask_kernel<<<(unsigned)blocks, block>>>(
        (float*)out_mask->data, S, N, config->mode == BLT_BDM_INFER ? 1 : 0);
    blt_cuda_sync_check("blt_build_block_diffusion_mask");

    // Every supported configuration gives each row a valid key (row i sees
    // itself), so no extra validity scan is needed here.
}
