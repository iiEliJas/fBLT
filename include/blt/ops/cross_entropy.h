#ifndef BLT_OPS_CROSS_ENTROPY_H
#define BLT_OPS_CROSS_ENTROPY_H

#ifdef __cplusplus
extern "C" {
#endif
#include "blt/core/tensor.h"

// logits: [seq_len, vocab_size] FP32 (not normalized)
// targets: [seq_len] UINT8 or INT32
// loss_out: scalar mean cross-entropy over seq_len
void blt_cross_entropy_forward(const blt_tensor* logits, const blt_tensor* targets, blt_tensor* loss_out);

// Writes dL/dlogits into grad_logits [seq_len, vocab_size]
// Computed as (softmax(logits) - one_hot(targets)) / seq_len
void blt_cross_entropy_backward(const blt_tensor* logits, const blt_tensor* targets, blt_tensor* grad_logits);

#ifdef __cplusplus
}
#endif
#endif
