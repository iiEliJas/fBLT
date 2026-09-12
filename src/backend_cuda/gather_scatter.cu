#include "blt/ops/gather_scatter.h"
#include "blt/core/backend.h"
#include "blt/core/cuda_shim.h"
#include "blt/core/allocator.h"
#include "blt/core/tensor.h"

#include <cuda_runtime.h>
#include <stdlib.h>
#include <string.h>

// Host-side id/index arrays are mirrored into device scratch buffers per call.

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

// Deterministic single-threaded scatter-add, selected when g_blt_deterministic is set.

__global__ void blt_embedding_scatter_add_det_kernel(float *grad_table, const unsigned char *ids, const float *grad_out,
                                                     size_t seq_len, size_t embed_dim) {
    for (size_t i = 0; i < seq_len; i++) {
        const size_t row = (size_t)ids[i];
        float *dst = grad_table + row * embed_dim;
        const float *src = grad_out + i * embed_dim;
        for (size_t d = 0; d < embed_dim; d++) dst[d] += src[d];
    }
}

__global__ void blt_indexed_row_scatter_add_det_kernel(float *grad_table, const unsigned int *idx,
                                                       const float *grad_out, size_t rows, size_t embed_dim,
                                                       float scale) {
    for (size_t i = 0; i < rows; i++) {
        const unsigned int id = idx[i];
        if (id == BLT_IDX_SENTINEL) continue;
        float *dst = grad_table + (size_t)id * embed_dim;
        const float *src = grad_out + i * embed_dim;
        for (size_t d = 0; d < embed_dim; d++) dst[d] += scale * src[d];
    }
}

__global__ void blt_indexed_row_scatter_add_normalized_count_det_kernel(unsigned int *counts, const unsigned int *idx,
                                                                        size_t rows) {
    for (size_t i = 0; i < rows; i++) {
        const unsigned int id = idx[i];
        if (id == BLT_IDX_SENTINEL) continue;
        counts[id]++;
    }
}

__global__ void blt_indexed_row_scatter_add_normalized_det_kernel(float *grad_table, const unsigned int *idx,
                                                                  const unsigned int *counts, const float *grad_out,
                                                                  size_t rows, size_t embed_dim, float scale) {
    for (size_t i = 0; i < rows; i++) {
        const unsigned int id = idx[i];
        if (id == BLT_IDX_SENTINEL) continue;
        const unsigned int count = counts[id];
        if (count == 0) continue;
        const float inv_count = 1.0f / (float)count;
        float *dst = grad_table + (size_t)id * embed_dim;
        const float *src = grad_out + i * embed_dim;
        for (size_t d = 0; d < embed_dim; d++) dst[d] += scale * inv_count * src[d];
    }
}

__global__ void blt_embedding_lookup_kernel(const float *table, size_t table_rows, const unsigned char *ids, float *out,
                                            size_t seq_len, size_t embed_dim) {
    const size_t work = seq_len * embed_dim;
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x; idx < work;
         idx += (size_t)gridDim.x * blockDim.x) {
        const size_t i = idx / embed_dim;
        const unsigned char id = ids[i];
        out[idx] = table[(size_t)id * embed_dim + (idx % embed_dim)];
    }
}

extern "C" void blt_embedding_lookup_cuda(const blt_tensor *table, const uint8_t *ids_host, blt_tensor *out) {
    const size_t seq_len = out->shape[0];
    const size_t embed_dim = out->shape[1];

    for (size_t i = 0; i < seq_len; i++) {
        BLT_REQUIRE(ids_host[i] < table->shape[0], "blt_embedding_lookup: id out of range");
    }

    unsigned char *d_ids = (unsigned char *)blt_arena_alloc(blt_cuda_get_scratch_arena(), seq_len > 0 ? seq_len : 1, 1);
    blt_cuda_memcpy_h2d(d_ids, ids_host, seq_len);
    blt_embedding_lookup_kernel<<<blt_cuda_grid(seq_len * embed_dim), 256>>>(
        (const float *)table->data, table->shape[0], d_ids, (float *)out->data, seq_len, embed_dim);
    blt_cuda_launch_check("blt_embedding_lookup");
}

__global__ void blt_embedding_scatter_add_kernel(float *grad_table, size_t table_rows, const unsigned char *ids,
                                                 const float *grad_out, size_t seq_len, size_t embed_dim) {
    const size_t work = seq_len * embed_dim;
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x; idx < work;
         idx += (size_t)gridDim.x * blockDim.x) {
        const size_t i = idx / embed_dim;
        const size_t row = (size_t)ids[i];
        atomicAdd(&grad_table[row * embed_dim + (idx % embed_dim)], grad_out[idx]);
    }
}

extern "C" void blt_embedding_scatter_add_cuda(const blt_tensor *grad_table, const uint8_t *ids_host,
                                               const blt_tensor *grad_out) {
    const size_t seq_len = grad_out->shape[0];
    const size_t embed_dim = grad_out->shape[1];

    for (size_t i = 0; i < seq_len; i++) {
        BLT_REQUIRE(ids_host[i] < grad_table->shape[0], "blt_embedding_scatter_add: id out of range");
    }

    unsigned char *d_ids = (unsigned char *)blt_arena_alloc(blt_cuda_get_scratch_arena(), seq_len > 0 ? seq_len : 1, 1);
    blt_cuda_memcpy_h2d(d_ids, ids_host, seq_len);
    if (g_blt_deterministic) {
        blt_embedding_scatter_add_det_kernel<<<1, 1>>>((float *)grad_table->data, d_ids, (const float *)grad_out->data,
                                                       seq_len, embed_dim);
    } else {
        blt_embedding_scatter_add_kernel<<<blt_cuda_grid(seq_len * embed_dim), 256>>>(
            (float *)grad_table->data, grad_table->shape[0], d_ids, (const float *)grad_out->data, seq_len, embed_dim);
    }
    blt_cuda_launch_check("blt_embedding_scatter_add");
}

__global__ void blt_indexed_row_accumulate_kernel(const float *table, size_t table_rows, const unsigned int *idx,
                                                  float *io, size_t rows, size_t embed_dim) {
    const size_t work = rows * embed_dim;
    for (size_t pos = (size_t)blockIdx.x * blockDim.x + threadIdx.x; pos < work;
         pos += (size_t)gridDim.x * blockDim.x) {
        const size_t i = pos / embed_dim;
        const unsigned int id = idx[i];
        if (id == BLT_IDX_SENTINEL) continue;
        io[pos] += table[(size_t)id * embed_dim + (pos % embed_dim)];
    }
}

extern "C" void blt_indexed_row_accumulate_cuda(const blt_tensor *table, const uint32_t *idx_host, blt_tensor *io) {
    const size_t rows = io->shape[0];
    const size_t embed_dim = io->shape[1];

    for (size_t i = 0; i < rows; i++) {
        BLT_REQUIRE(idx_host[i] == BLT_IDX_SENTINEL || idx_host[i] < table->shape[0],
                    "blt_indexed_row_accumulate: index out of range");
    }

    unsigned int *d_idx =
        (unsigned int *)blt_arena_alloc(blt_cuda_get_scratch_arena(), rows > 0 ? rows * sizeof(unsigned int) : 1, 4);
    blt_cuda_memcpy_h2d(d_idx, idx_host, rows * sizeof(unsigned int));
    blt_indexed_row_accumulate_kernel<<<blt_cuda_grid(rows * embed_dim), 256>>>(
        (const float *)table->data, table->shape[0], d_idx, (float *)io->data, rows, embed_dim);
    blt_cuda_launch_check("blt_indexed_row_accumulate");
}

__global__ void blt_indexed_row_scatter_add_kernel(float *grad_table, size_t table_rows, const unsigned int *idx,
                                                   const float *grad_out, size_t rows, size_t embed_dim, float scale) {
    const size_t work = rows * embed_dim;
    for (size_t pos = (size_t)blockIdx.x * blockDim.x + threadIdx.x; pos < work;
         pos += (size_t)gridDim.x * blockDim.x) {
        const size_t i = pos / embed_dim;
        const unsigned int id = idx[i];
        if (id == BLT_IDX_SENTINEL) continue;
        atomicAdd(&grad_table[(size_t)id * embed_dim + (pos % embed_dim)], scale * grad_out[pos]);
    }
}

extern "C" void blt_indexed_row_scatter_add_cuda(const blt_tensor *grad_table, const uint32_t *idx_host,
                                                 const blt_tensor *grad_out, float scale) {
    const size_t rows = grad_out->shape[0];
    const size_t embed_dim = grad_out->shape[1];

    for (size_t i = 0; i < rows; i++) {
        BLT_REQUIRE(idx_host[i] == BLT_IDX_SENTINEL || idx_host[i] < grad_table->shape[0],
                    "blt_indexed_row_scatter_add: index out of range");
    }

    unsigned int *d_idx =
        (unsigned int *)blt_arena_alloc(blt_cuda_get_scratch_arena(), rows > 0 ? rows * sizeof(unsigned int) : 1, 4);
    blt_cuda_memcpy_h2d(d_idx, idx_host, rows * sizeof(unsigned int));
    if (g_blt_deterministic) {
        blt_indexed_row_scatter_add_det_kernel<<<1, 1>>>((float *)grad_table->data, d_idx,
                                                         (const float *)grad_out->data, rows, embed_dim, scale);
    } else {
        blt_indexed_row_scatter_add_kernel<<<blt_cuda_grid(rows * embed_dim), 256>>>(
            (float *)grad_table->data, grad_table->shape[0], d_idx, (const float *)grad_out->data, rows, embed_dim,
            scale);
    }
    blt_cuda_launch_check("blt_indexed_row_scatter_add");
}

__global__ void blt_indexed_row_scatter_add_normalized_count_kernel(unsigned int *counts, const unsigned int *idx,
                                                                    size_t rows) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < rows; i += (size_t)gridDim.x * blockDim.x) {
        const unsigned int id = idx[i];
        if (id == BLT_IDX_SENTINEL) continue;
        atomicAdd(&counts[id], 1u);
    }
}

__global__ void blt_indexed_row_scatter_add_normalized_kernel(float *grad_table, const unsigned int *idx,
                                                              const unsigned int *counts, const float *grad_out,
                                                              size_t rows, size_t embed_dim, float scale) {
    const size_t work = rows * embed_dim;
    for (size_t pos = (size_t)blockIdx.x * blockDim.x + threadIdx.x; pos < work;
         pos += (size_t)gridDim.x * blockDim.x) {
        const size_t i = pos / embed_dim;
        const unsigned int id = idx[i];
        if (id == BLT_IDX_SENTINEL) continue;
        const unsigned int count = counts[id];
        if (count == 0) continue;
        const float inv_count = 1.0f / (float)count;
        atomicAdd(&grad_table[(size_t)id * embed_dim + (pos % embed_dim)], scale * inv_count * grad_out[pos]);
    }
}

extern "C" void blt_indexed_row_scatter_add_normalized_cuda(const blt_tensor *grad_table, const uint32_t *idx_host,
                                                            const blt_tensor *grad_out, float scale) {
    const size_t rows = grad_out->shape[0];
    const size_t embed_dim = grad_out->shape[1];
    const size_t table_rows = grad_table->shape[0];

    for (size_t i = 0; i < rows; i++) {
        BLT_REQUIRE(idx_host[i] == BLT_IDX_SENTINEL || idx_host[i] < table_rows,
                    "blt_indexed_row_scatter_add_normalized: index out of range");
    }

    unsigned int *d_idx =
        (unsigned int *)blt_arena_alloc(blt_cuda_get_scratch_arena(), rows > 0 ? rows * sizeof(unsigned int) : 1, 4);
    blt_cuda_memcpy_h2d(d_idx, idx_host, rows * sizeof(unsigned int));

    unsigned int *d_counts = (unsigned int *)blt_arena_alloc(blt_cuda_get_scratch_arena(),
                                                             table_rows > 0 ? table_rows * sizeof(unsigned int) : 1, 4);
    cudaMemset(d_counts, 0, table_rows * sizeof(unsigned int));
    blt_cuda_launch_check("blt_indexed_row_scatter_add_normalized memset");

    if (g_blt_deterministic) {
        blt_indexed_row_scatter_add_normalized_count_det_kernel<<<1, 1>>>(d_counts, d_idx, rows);
        blt_cuda_launch_check("blt_indexed_row_scatter_add_normalized count (det)");

        blt_indexed_row_scatter_add_normalized_det_kernel<<<1, 1>>>(
            (float *)grad_table->data, d_idx, d_counts, (const float *)grad_out->data, rows, embed_dim, scale);
        blt_cuda_launch_check("blt_indexed_row_scatter_add_normalized scatter (det)");
    } else {
        blt_indexed_row_scatter_add_normalized_count_kernel<<<blt_cuda_grid(rows), 256>>>(d_counts, d_idx, rows);
        blt_cuda_launch_check("blt_indexed_row_scatter_add_normalized count");

        blt_indexed_row_scatter_add_normalized_kernel<<<blt_cuda_grid(rows * embed_dim), 256>>>(
            (float *)grad_table->data, d_idx, d_counts, (const float *)grad_out->data, rows, embed_dim, scale);
        blt_cuda_launch_check("blt_indexed_row_scatter_add_normalized scatter");
    }
}

__global__ void blt_rows_gather_kernel(const float *src, const size_t *pos, float *dst, size_t rows, size_t row_len) {
    const size_t work = rows * row_len;
    for (size_t off = (size_t)blockIdx.x * blockDim.x + threadIdx.x; off < work;
         off += (size_t)gridDim.x * blockDim.x) {
        const size_t i = off / row_len;
        dst[off] = src[pos[i] * row_len + (off % row_len)];
    }
}

extern "C" void blt_rows_gather_cuda(const blt_tensor *src, const size_t *pos_host, blt_tensor *dst) {
    const size_t rows = dst->shape[0];
    const size_t row_len = dst->numel / rows;

    for (size_t i = 0; i < rows; i++) {
        BLT_REQUIRE(pos_host[i] * row_len <= src->numel - row_len, "blt_rows_gather: position out of range");
    }

    size_t *d_pos = (size_t *)blt_arena_alloc(blt_cuda_get_scratch_arena(), rows > 0 ? rows * sizeof(size_t) : 1, 8);
    cudaError_t err = cudaSuccess;
    blt_cuda_memcpy_h2d(d_pos, pos_host, rows * sizeof(size_t));
    blt_rows_gather_kernel<<<blt_cuda_grid(rows * row_len), 256>>>((const float *)src->data, d_pos, (float *)dst->data,
                                                                   rows, row_len);
    err = cudaGetLastError();
    blt_cuda_launch_check("blt_rows_gather");
}
