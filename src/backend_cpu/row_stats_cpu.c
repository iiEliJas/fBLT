#include "blt/ops/row_stats.h"
#include "blt/core/backend.h"

#include <math.h>

void blt_entropy_rows_cpu(const blt_tensor *probs, blt_tensor *entropy_out, int use_log2) {
    BLT_REQUIRE(probs->backend == BLT_BACKEND_CPU && entropy_out->backend == BLT_BACKEND_CPU,
                "blt_entropy_rows: CPU implementation called with non-CPU tensors");
    const size_t rows = probs->shape[0];
    const size_t vocab = probs->shape[1];

    const float *p = (const float *)probs->data;
    float *out = (float *)entropy_out->data;
    for (size_t i = 0; i < rows; ++i) {
        float entropy = 0.0f;
        for (size_t j = 0; j < vocab; ++j) {
            float pv = p[i * vocab + j];
            if (pv > 1e-9f) {
                if (use_log2) {
                    entropy -= pv * log2f(pv);
                } else {
                    entropy -= pv * logf(pv);
                }
            }
        }
        out[i] = entropy;
    }
}

void blt_argmax_rows_cpu(const blt_tensor *logits, uint32_t *out_ids_host) {
    BLT_REQUIRE(logits->backend == BLT_BACKEND_CPU, "blt_argmax_rows: CPU implementation called with non-CPU tensor");
    const size_t rows = logits->shape[0];
    const size_t vocab = logits->shape[1];

    const float *l = (const float *)logits->data;
    for (size_t i = 0; i < rows; i++) {
        size_t best = 0;
        float best_val = l[i * vocab];
        for (size_t j = 1; j < vocab; j++) {
            const float v = l[i * vocab + j];
            if (v > best_val) {
                best_val = v;
                best = j;
            }
        }
        out_ids_host[i] = (uint32_t)best;
    }
}
