#ifndef BLT_MODELS_ATTENTION_H
#define BLT_MODELS_ATTENTION_H

#include "blt/core/tensor.h"
#include "blt/core/allocator.h"


typedef struct {
    size_t embed_dim;    // Total embedding dimension
    size_t num_heads;    // Number of attention heads
    size_t head_dim;     // Dimension per head (embed_dim / num_heads)
    bool is_causal;      // apply masking matrix
} blt_attention_config;

// Computes Multi-Head Self-Attention
void blt_multihead_attention(
    const blt_tensor* input, 
    const blt_tensor* weight_qkv, 
    const blt_tensor* weight_proj, 
    blt_tensor* output, 
    const blt_attention_config* config,
    blt_arena* arena
);


#endif // BLT_MODELS_ATTENTION_H