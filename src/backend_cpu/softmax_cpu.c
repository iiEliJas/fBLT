#include "blt/core/backend.h"
#include "blt/ops/softmax.h"

#include <math.h>

void blt_softmax_cpu(const blt_tensor* in, blt_tensor* out) {
    if (in->dtype != BLT_DTYPE_FP32 || out->dtype != BLT_DTYPE_FP32) {
        BLT_FATAL("softmax only supports FP32 tensors");
    }
    if (in->numel != out->numel) {
        BLT_FATAL("softmax input/output numel mismatch");
    }

    const float* in_data = (const float*)in->data;
    float* out_data = (float*)out->data;

    size_t last_dim = in->ndim == 0 ? 0 : in->shape[in->ndim - 1];
    size_t rows = in->numel / last_dim;

    for (size_t row = 0; row < rows; ++row) {
        size_t base = row * last_dim;
        float max_val = in_data[base];
        for (size_t i = 1; i < last_dim; ++i) {
            float v = in_data[base + i];
            if (v > max_val) {
                max_val = v;
            }
        }

        float sum = 0.0f;
        for (size_t i = 0; i < last_dim; ++i) {
            float exp_val = expf(in_data[base + i] - max_val);
            out_data[base + i] = exp_val;
            sum += exp_val;
        }

        for (size_t i = 0; i < last_dim; ++i) {
            out_data[base + i] /= sum;
        }
    }
}
