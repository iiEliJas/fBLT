#ifndef BLT_MODELS_ENTROPY_H
#define BLT_MODELS_ENTROPY_H

#include "blt/core/tensor.h"

// Config for entropy calculation
typedef struct {
    float threshold;      // entropy spike threshold to trigger a new patch
    size_t vocab_size;
    bool use_log2;        // calculate entropy in bits (log2) or nats (ln)
} blt_entropy_config;

// Calculates the entropy of a probability distribution tensor
// Input: 2D tensor of shape [batch_size * seq_len, vocab_size] containing probabilities (after softmax)
// Output: 1D tensor of shape [batch_size * seq_len] containing entropy values
void blt_compute_entropy(const blt_tensor* probs, blt_tensor* entropy_out, const blt_entropy_config* config);


#endif //BLT_MODELS_ENTROPY_H