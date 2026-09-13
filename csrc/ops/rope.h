#ifndef BLT_OPS_ROPE_H
#define BLT_OPS_ROPE_H

#ifdef __cplusplus
extern "C" {
#endif
#include "core/tensor.h"

typedef struct {
    float theta;     // BLT uses 500000.0f
    size_t head_dim; // must be even
} blt_rope_config;

// Precompute cos/sin tables for positions [0, max_seq_len) x head_dim/2 frequency bands.
// cos_out / sin_out must be [max_seq_len, head_dim/2] FP32.
void blt_rope_precompute(size_t max_seq_len, const blt_rope_config *config, blt_tensor *cos_out, blt_tensor *sin_out);

// Apply RoPE in-place. x and out are both [seq_len, num_heads, head_dim].
void blt_rope_apply(const blt_tensor *x, const blt_tensor *cos, const blt_tensor *sin, blt_tensor *out);

void blt_rope_apply_backward(const blt_tensor *grad_out, const blt_tensor *cos, const blt_tensor *sin,
                             blt_tensor *grad_in);

// Fused strided_copy + RoPE for packed QKV layout. Eliminates intermediate
// buffers and 3 kernel launches per head.
void blt_rope_apply_packed(float *qkv_data, size_t qkv_stride, size_t head_offset, size_t seq_len, size_t head_dim,
                           const float *cos, const float *sin, blt_backend backend);

void blt_rope_apply_packed_backward(float *qkv_data, size_t qkv_stride, size_t head_offset, size_t seq_len,
                                    size_t head_dim, const float *cos, const float *sin, blt_backend backend);

#ifdef __cplusplus
}
#endif
#endif
