#ifndef BLT_OPS_ROPE_H
#define BLT_OPS_ROPE_H

#include "blt/core/tensor.h"

// RoPE ... rotary position embedding

typedef struct {
    float theta;        // base, 10000.0f or 500000.0f (BLT uses 500000)
    size_t head_dim;     // dimension per attention head ((must be even)
} blt_rope_config;


// Precomputes cos/sin tables for positions [0, max_seq_len) and head_dim/2 frequency bands
// cos_out / sin_out must be [max_seq_len, head_dim/2] FP32 tensors
void blt_rope_precompute(size_t max_seq_len, const blt_rope_config* config,
                          blt_tensor* cos_out, blt_tensor* sin_out);


// Applies rotary embedding in place to the input tensor x, using precomputed cos/sin tables
// reads x [seq_len, num_heads, head_dim],
// writes rotated result into out (same shape)
void blt_rope_apply(const blt_tensor* x, const blt_tensor* cos, const blt_tensor* sin, blt_tensor* out);


// Backward pass for rotary embedding
void blt_rope_apply_backward(const blt_tensor* grad_out, const blt_tensor* cos,
                            const blt_tensor* sin, blt_tensor* grad_in);

#endif