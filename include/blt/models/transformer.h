#ifndef BLT_MODELS_TRANSFORMER_H
#define BLT_MODELS_TRANSFORMER_H

#include "blt/core/tensor.h"
#include "blt/models/attention.h"


typedef struct {
    blt_attention_config attn_config;
    size_t hidden_dim;   // Dimension of the FFN hidden layer (usually 4 * embed_dim)
    float layer_norm_eps; // Small epsilon for normalization stability
} blt_transformer_config;


// Executes a single Transformer block forward pass.
void blt_transformer_forward(
    const blt_tensor* input, 
    const blt_tensor* norm1_weight,
    const blt_tensor* norm1_bias,
    const blt_tensor* attn_qkv_w,
    const blt_tensor* attn_proj_w,
    const blt_tensor* norm2_weight,
    const blt_tensor* norm2_bias,
    const blt_tensor* ffn_up_w,
    const blt_tensor* ffn_down_w,
    blt_tensor* output,
    const blt_transformer_config* config
);

#endif // BLT_MODELS_TRANSFORMER_H