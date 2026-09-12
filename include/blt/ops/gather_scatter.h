#ifndef BLT_OPS_GATHER_SCATTER_H
#define BLT_OPS_GATHER_SCATTER_H

#include <stddef.h>
#include <stdint.h>
#include "blt/core/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

// Gather/scatter primitives over tensor payloads. Index/id arrays are always
// HOST pointers (callers compute them from bytes and patch tables on CPU);
// CUDA implementations mirror them into device buffers internally, so model
// code never branches on backend.
//
// Dispatch is keyed on the first tensor argument's backend.

// out[i, :] = table[ids_host[i], :]; ids_host holds seq_len byte values,
// each validated against table row count.
void blt_embedding_lookup(const blt_tensor *table, const uint8_t *ids_host, blt_tensor *out);

// grad_table[ids_host[i], :] += grad_out[i, :]  (accumulates).
void blt_embedding_scatter_add(const blt_tensor *grad_table, const uint8_t *ids_host, const blt_tensor *grad_out);

// io[i, :] += table[idx_host[i], :] for every non-sentinel i. Used for n-gram adds.
#define BLT_IDX_SENTINEL UINT32_MAX
void blt_indexed_row_accumulate(const blt_tensor *table, const uint32_t *idx_host, blt_tensor *io);

// grad_table[idx_host[i], :] += scale * grad_out[i, :] for non-sentinel i
// (accumulates; duplicate indices accumulate in sequence order on CPU).
void blt_indexed_row_scatter_add(const blt_tensor *grad_table, const uint32_t *idx_host, const blt_tensor *grad_out,
                                 float scale);

// Like blt_indexed_row_scatter_add but normalizes each row's contribution by
// the number of positions sharing the same index, so a bucket receiving N
// contributions gets the AVERAGE gradient, not the SUM.
void blt_indexed_row_scatter_add_normalized(blt_backend backend, const blt_tensor *grad_table, const uint32_t *idx_host,
                                            const blt_tensor *grad_out, float scale);

// dst[i, :] = src[pos_host[i], :] (row-wise gather with full-row copies).
void blt_rows_gather(const blt_tensor *src, const size_t *pos_host, blt_tensor *dst);

#ifdef __cplusplus
}
#endif

#endif // BLT_OPS_GATHER_SCATTER_H
