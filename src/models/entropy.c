#include "blt/models/entropy.h"
#include "blt/core/backend.h"
#include <math.h>


//----------------------------------------------------------------
// Computes Entropy
//
// For each row in the input probabilities tensor, compute the entropy and store it in the output tensor
// Entropy formula: H = -sum(p * log(p)) for each probability distribution in the input tensor

void blt_compute_entropy(const blt_tensor* probs, blt_tensor* entropy_out, const blt_entropy_config* config){
    blt_check_nd_fp32(probs, 2, (const size_t[]){0, config->vocab_size}, 
                        "Input probabilities tensor must be 2D with last dimension equal to vocab_size");

    blt_check_nd_fp32(entropy_out, 1, (const size_t[]){probs->shape[0]}, 
                        "Output entropy tensor must be 1D with shape matching the number of rows in input probabilities tensor");

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