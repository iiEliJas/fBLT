#include "ops/gather_scatter.h"
#include "core/backend.h"

#include <stdlib.h>
#include <string.h>

void blt_embedding_lookup_cpu(const blt_tensor *table, const uint32_t *ids_host, blt_tensor *out) {
    BLT_REQUIRE(table->backend == BLT_BACKEND_CPU && out->backend == BLT_BACKEND_CPU,
                "blt_embedding_lookup: CPU implementation called with non-CPU tensors");
    const size_t seq_len = out->shape[0];
    const size_t embed_dim = out->shape[1];
    BLT_REQUIRE(table->shape[1] == embed_dim, "blt_embedding_lookup: embed_dim mismatch");

    const float *t = (const float *)table->data;
    float *o = (float *)out->data;
    for (size_t i = 0; i < seq_len; i++) {
        BLT_REQUIRE(ids_host[i] < table->shape[0], "blt_embedding_lookup: id out of range");
        memcpy(o + i * embed_dim, t + (size_t)ids_host[i] * embed_dim, embed_dim * sizeof(float));
    }
}

void blt_embedding_scatter_add_cpu(const blt_tensor *grad_table, const uint32_t *ids_host, const blt_tensor *grad_out) {
    BLT_REQUIRE(grad_table->backend == BLT_BACKEND_CPU && grad_out->backend == BLT_BACKEND_CPU,
                "blt_embedding_scatter_add: CPU implementation called with non-CPU tensors");
    const size_t seq_len = grad_out->shape[0];
    const size_t embed_dim = grad_out->shape[1];
    BLT_REQUIRE(grad_table->shape[1] == embed_dim, "blt_embedding_scatter_add: embed_dim mismatch");

    float *g = (float *)grad_table->data;
    const float *go = (const float *)grad_out->data;
    for (size_t i = 0; i < seq_len; i++) {
        BLT_REQUIRE(ids_host[i] < grad_table->shape[0], "blt_embedding_scatter_add: id out of range");
        float *row = g + (size_t)ids_host[i] * embed_dim;
        const float *src = go + i * embed_dim;
        for (size_t d = 0; d < embed_dim; d++) {
            row[d] += src[d];
        }
    }
}

void blt_indexed_row_accumulate_cpu(const blt_tensor *table, const uint32_t *idx_host, blt_tensor *io) {
    BLT_REQUIRE(table->backend == BLT_BACKEND_CPU && io->backend == BLT_BACKEND_CPU,
                "blt_indexed_row_accumulate: CPU implementation called with non-CPU tensors");
    const size_t rows = io->shape[0];
    const size_t embed_dim = io->shape[1];
    BLT_REQUIRE(table->shape[1] == embed_dim, "blt_indexed_row_accumulate: embed_dim mismatch");

    const float *t = (const float *)table->data;
    float *o = (float *)io->data;
    for (size_t i = 0; i < rows; i++) {
        if (idx_host[i] == BLT_IDX_SENTINEL) {
            continue;
        }
        BLT_REQUIRE(idx_host[i] < table->shape[0], "blt_indexed_row_accumulate: index out of range");
        const float *row = t + (size_t)idx_host[i] * embed_dim;
        float *dst = o + i * embed_dim;
        for (size_t d = 0; d < embed_dim; d++) {
            dst[d] += row[d];
        }
    }
}

void blt_indexed_row_scatter_add_cpu(const blt_tensor *grad_table, const uint32_t *idx_host, const blt_tensor *grad_out,
                                     float scale) {
    BLT_REQUIRE(grad_table->backend == BLT_BACKEND_CPU && grad_out->backend == BLT_BACKEND_CPU,
                "blt_indexed_row_scatter_add: CPU implementation called with non-CPU tensors");
    const size_t rows = grad_out->shape[0];
    const size_t embed_dim = grad_out->shape[1];
    BLT_REQUIRE(grad_table->shape[1] == embed_dim, "blt_indexed_row_scatter_add: embed_dim mismatch");

    float *g = (float *)grad_table->data;
    const float *go = (const float *)grad_out->data;
    for (size_t i = 0; i < rows; i++) {
        if (idx_host[i] == BLT_IDX_SENTINEL) {
            continue;
        }
        BLT_REQUIRE(idx_host[i] < grad_table->shape[0], "blt_indexed_row_scatter_add: index out of range");
        float *row = g + (size_t)idx_host[i] * embed_dim;
        const float *src = go + i * embed_dim;
        for (size_t d = 0; d < embed_dim; d++) {
            row[d] += scale * src[d];
        }
    }
}

void blt_indexed_row_scatter_add_normalized_cpu(const blt_tensor *grad_table, const uint32_t *idx_host,
                                                const blt_tensor *grad_out, float scale) {
    BLT_REQUIRE(grad_table->backend == BLT_BACKEND_CPU && grad_out->backend == BLT_BACKEND_CPU,
                "blt_indexed_row_scatter_add_normalized: CPU implementation called with non-CPU tensors");
    const size_t rows = grad_out->shape[0];
    const size_t embed_dim = grad_out->shape[1];
    BLT_REQUIRE(grad_table->shape[1] == embed_dim, "blt_indexed_row_scatter_add_normalized: embed_dim mismatch");

    float *g = (float *)grad_table->data;
    const float *go = (const float *)grad_out->data;

    const size_t table_rows = grad_table->shape[0];
    size_t *counts = (size_t *)calloc(table_rows, sizeof(size_t));
    BLT_REQUIRE(counts != NULL, "blt_indexed_row_scatter_add_normalized: allocation failed");

    for (size_t i = 0; i < rows; i++) {
        if (idx_host[i] == BLT_IDX_SENTINEL) continue;
        BLT_REQUIRE(idx_host[i] < table_rows, "blt_indexed_row_scatter_add_normalized: index out of range");
        counts[idx_host[i]]++;
    }

    for (size_t i = 0; i < rows; i++) {
        if (idx_host[i] == BLT_IDX_SENTINEL) continue;
        const size_t idx = idx_host[i];
        const float inv_count = 1.0f / (float)counts[idx];
        float *row = g + idx * embed_dim;
        const float *src = go + i * embed_dim;
        for (size_t d = 0; d < embed_dim; d++) {
            row[d] += scale * inv_count * src[d];
        }
    }

    free(counts);
}

void blt_rows_gather_cpu(const blt_tensor *src, const size_t *pos_host, blt_tensor *dst) {
    BLT_REQUIRE(src->backend == BLT_BACKEND_CPU && dst->backend == BLT_BACKEND_CPU,
                "blt_rows_gather: CPU implementation called with non-CPU tensors");
    const size_t rows = dst->shape[0];
    const size_t row_len = dst->numel > 0 ? dst->numel / rows : 0;
    BLT_REQUIRE(src->ndim >= 1 && src->numel % src->shape[src->ndim - 1] == 0,
                "blt_rows_gather: src must be row-structured");

    const float *s = (const float *)src->data;
    float *d = (float *)dst->data;
    for (size_t i = 0; i < rows; i++) {
        BLT_REQUIRE(pos_host[i] * row_len <= src->numel - row_len, "blt_rows_gather: position out of range");
        memcpy(d + i * row_len, s + pos_host[i] * row_len, row_len * sizeof(float));
    }
}
