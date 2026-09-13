#ifndef BLT_MODELS_ENTROPY_H
#define BLT_MODELS_ENTROPY_H

#ifdef __cplusplus
extern "C" {
#endif
#include "core/tensor.h"
#include "models/patcher.h"

typedef struct {
    size_t vocab_size;
    bool use_log2; // log2 for bits, ln for nats
} blt_entropy_config;

// probs: [batch_size * seq_len, vocab_size] after softmax.
// entropy_out: [batch_size * seq_len].
void blt_compute_entropy(const blt_tensor *probs, blt_tensor *entropy_out, const blt_entropy_config *config);

#ifdef __cplusplus
}
#endif
#endif // BLT_MODELS_ENTROPY_H
