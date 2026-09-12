#ifndef BLT_OPS_ROW_STATS_H
#define BLT_OPS_ROW_STATS_H

#include <stddef.h>
#include <stdint.h>
#include "blt/core/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

// Row-wise statistics helpers shared by inference and training paths.

// entropy_out[i] = -sum_v p log(p) over each row of probs [rows, vocab]
// (log base 2 when use_log2). Entries p <= 1e-9 contribute zero.
void blt_entropy_rows(const blt_tensor *probs, blt_tensor *entropy_out, int use_log2);

// out_ids_host[i] = argmax_j logits[i, j]; ties resolve to the lowest index.
// The destination is a caller-allocated HOST array of seq_len entries; the
// CUDA implementation copies results back before returning.
void blt_argmax_rows(const blt_tensor *logits, uint32_t *out_ids_host);

#ifdef __cplusplus
}
#endif

#endif // BLT_OPS_ROW_STATS_H
