#include "blt/models/entropy.h"
#include "blt/core/backend.h"
#include <math.h>


/*
    For each row in the input probabilities tensor, compute the entropy and store it in the output tensor
    Ensure that the input tensor is 2D and the output tensor is 1D with the correct shape
    Entropy formula: H = -sum(p * log(p)) for each probability distribution in the input tensor
*/
void blt_compute_entropy(const blt_tensor* probs, blt_tensor* entropy_out, const blt_entropy_config* config){
    if (probs->ndim != 2) {
        BLT_FATAL("Input probabilities tensor must be 2D");
    }
    if (probs->shape[1] != config->vocab_size) {
        BLT_FATAL("Input probabilities tensor last dimension must match vocab_size");
    }
    if (entropy_out->ndim != 1 || entropy_out->shape[0] != probs->shape[0]) {
        BLT_FATAL("Output entropy tensor must be 1D and match the first dimension of input probabilities tensor");
    }

    // compute entropy for each row in the probabilities tensor
    const float* probs_data = (const float*)probs->data;
    float* entropy_data = (float*)entropy_out->data;
    for (size_t i = 0; i < probs->shape[0]; ++i) {
        float entropy = 0.0f;
        for (size_t j = 0; j < config->vocab_size; ++j) {
            float p = probs_data[i * config->vocab_size + j];
            if (p > 1e-9f) { // avoids log(0)
                if (config->use_log2) {
                    entropy -= p * log2f(p);
                } else {
                    entropy -= p * logf(p);
                }
            }
        }
        entropy_data[i] = entropy;
    }
}