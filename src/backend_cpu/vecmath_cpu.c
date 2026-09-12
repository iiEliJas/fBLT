#include "blt/ops/vecmath.h"
#include "blt/core/backend.h"

#include <math.h>

float blt_vec_dot_cpu(blt_backend backend, const float *a, const float *b, size_t n) {
    (void)backend;
    BLT_REQUIRE(backend == BLT_BACKEND_CPU, "blt_vec_dot: CPU implementation called with non-CPU backend");
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

void blt_softmax_masked_row_inplace_cpu(blt_backend backend, float *row, size_t row_len, size_t row_idx, bool is_causal,
                                        const float *mask_row, float scale) {
    (void)backend;
    BLT_REQUIRE(backend == BLT_BACKEND_CPU,
                "blt_softmax_masked_row_inplace: CPU implementation called with non-CPU backend");

    float max_val = -INFINITY;

    for (size_t col = 0; col < row_len; ++col) {
        if (mask_row != NULL) {
            row[col] = row[col] * scale + mask_row[col];
        } else if (is_causal && col > row_idx) {
            row[col] = -INFINITY;
        } else {
            row[col] *= scale;
        }

        if (isfinite(row[col]) && row[col] > max_val) {
            max_val = row[col];
        }
    }

    float sum = 0.0f;
    for (size_t col = 0; col < row_len; ++col) {
        if (isfinite(row[col])) {
            row[col] = expf(row[col] - max_val);
            sum += row[col];
        } else {
            row[col] = 0.0f;
        }
    }

    if (sum > 0.0f) {
        for (size_t col = 0; col < row_len; ++col) {
            row[col] /= sum;
        }
    }
}

void blt_strided_copy_cpu(blt_backend backend, float *dst, size_t dst_stride, const float *src, size_t src_stride,
                          size_t rows, size_t cols) {
    (void)backend;
    BLT_REQUIRE(backend == BLT_BACKEND_CPU, "blt_strided_copy: CPU implementation called with non-CPU backend");
    for (size_t r = 0; r < rows; r++) {
        memcpy(dst + r * dst_stride, src + r * src_stride, cols * sizeof(float));
    }
}

static float rng_uniform_u64(uint64_t *state, float scale) {
    uint64_t x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return (((float)(x >> 40) / 16777216.0f) * 2.0f - 1.0f) * scale;
}

void blt_fill_uniform_cpu(blt_backend backend, float *data, size_t n, uint64_t *rng_state) {
    (void)backend;
    BLT_REQUIRE(backend == BLT_BACKEND_CPU, "blt_fill_uniform: CPU implementation called with non-CPU backend");
    for (size_t i = 0; i < n; ++i) {
        data[i] = rng_uniform_u64(rng_state, 1.0f);
    }
}

void blt_fill_constant_cpu(blt_backend backend, float *data, size_t n, float v) {
    (void)backend;
    BLT_REQUIRE(backend == BLT_BACKEND_CPU, "blt_fill_constant: CPU implementation called with non-CPU backend");
    for (size_t i = 0; i < n; ++i) {
        data[i] = v;
    }
}
