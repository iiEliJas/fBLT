#include "blt/models/transformer.h"
#include "blt/core/backend.h"
#include "blt/ops/elementwise.h"
#include "blt/ops/matmul.h"
#include "blt/ops/gelu.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

// =====================
// Helper: GELU Activation
// =====================
static void blt_gelu(blt_tensor* x) {
    if (x->dtype != BLT_DTYPE_FP32) {
        BLT_FATAL("GELU requires FP32 tensors");
    }
    
    float* data = (float*)x->data;
    for (size_t i = 0; i < x->numel; i++) {
        float val = data[i];
        // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
        float x_cubed = val * val * val;
        float inner = 0.7978845608f * (val + 0.044715f * x_cubed);
        float tanh_result = tanh(inner);
        data[i] = 0.5f * val * (1.0f + tanh_result);
    }
}

// =====================
// Helper: Layer Normalization
// =====================
static void blt_layer_norm(
    const blt_tensor* input,
    const blt_tensor* weight,
    const blt_tensor* bias,
    blt_tensor* output,
    float eps
) {
    // Validation
    if (input == NULL) {
        BLT_FATAL("LayerNorm: input cannot be NULL");
    }
    if (weight == NULL || bias == NULL) {
        BLT_FATAL("LayerNorm: weight and bias cannot be NULL");
    }
    if (output == NULL) {
        BLT_FATAL("LayerNorm: output cannot be NULL");
    }
    
    if (input->dtype != BLT_DTYPE_FP32) {
        BLT_FATAL("LayerNorm: input must be FP32");
    }
    if (weight->dtype != BLT_DTYPE_FP32) {
        BLT_FATAL("LayerNorm: weight must be FP32");
    }
    if (bias->dtype != BLT_DTYPE_FP32) {
        BLT_FATAL("LayerNorm: bias must be FP32");
    }
    if (output->dtype != BLT_DTYPE_FP32) {
        BLT_FATAL("LayerNorm: output must be FP32");
    }
    
    if (input->ndim != 2) {
        BLT_FATAL("LayerNorm: input must be 2D [seq_len, embed_dim]");
    }
    if (weight->ndim != 1) {
        BLT_FATAL("LayerNorm: weight must be 1D");
    }
    if (bias->ndim != 1) {
        BLT_FATAL("LayerNorm: bias must be 1D");
    }
    
    size_t embed_dim = input->shape[1];
    
    if (weight->shape[0] != embed_dim) {
        BLT_FATAL("LayerNorm: weight dimension mismatch with input");
    }
    if (bias->shape[0] != embed_dim) {
        BLT_FATAL("LayerNorm: bias dimension mismatch with input");
    }
    if (output->numel != input->numel) {
        BLT_FATAL("LayerNorm: output size mismatch with input");
    }
    if (output->shape[0] != input->shape[0] || output->shape[1] != input->shape[1]) {
        BLT_FATAL("LayerNorm: output shape mismatch with input");
    }
    
    const float* in_data = (const float*)input->data;
    const float* w_data = (const float*)weight->data;
    const float* b_data = (const float*)bias->data;
    float* out_data = (float*)output->data;
    
    size_t seq_len = input->shape[0];
    
    // Apply layer norm row-wise (normalize over embed_dim for each sequence position)
    for (size_t i = 0; i < seq_len; i++) {
        const float* row = in_data + i * embed_dim;
        float* out_row = out_data + i * embed_dim;
        
        // Compute mean
        float mean = 0.0f;
        for (size_t j = 0; j < embed_dim; j++) {
            mean += row[j];
        }
        mean /= (float)embed_dim;
        
        // Compute variance
        float variance = 0.0f;
        for (size_t j = 0; j < embed_dim; j++) {
            float diff = row[j] - mean;
            variance += diff * diff;
        }
        variance /= (float)embed_dim;
        
        // Normalize, scale by weight, and shift by bias
        float inv_std = 1.0f / sqrtf(variance + eps);
        for (size_t j = 0; j < embed_dim; j++) {
            float normalized = (row[j] - mean) * inv_std;
            out_row[j] = normalized * w_data[j] + b_data[j];
        }
    }
}

// =====================
// Main Transformer Forward Pass
// =====================
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
) {
    // =====================
    // Input Validation
    // =====================
    if (input == NULL) {
        BLT_FATAL("Transformer: input tensor cannot be NULL");
    }
    if (norm1_weight == NULL || norm1_bias == NULL) {
        BLT_FATAL("Transformer: norm1_weight and norm1_bias cannot be NULL");
    }
    if (attn_qkv_w == NULL || attn_proj_w == NULL) {
        BLT_FATAL("Transformer: attn_qkv_w and attn_proj_w cannot be NULL");
    }
    if (norm2_weight == NULL || norm2_bias == NULL) {
        BLT_FATAL("Transformer: norm2_weight and norm2_bias cannot be NULL");
    }
    if (ffn_up_w == NULL || ffn_down_w == NULL) {
        BLT_FATAL("Transformer: ffn_up_w and ffn_down_w cannot be NULL");
    }
    if (output == NULL) {
        BLT_FATAL("Transformer: output tensor cannot be NULL");
    }
    if (config == NULL) {
        BLT_FATAL("Transformer: config cannot be NULL");
    }
    
    // Check input shape
    if (input->ndim != 2) {
        BLT_FATAL("Transformer: input must be 2D [seq_len, embed_dim]");
    }
    
    size_t seq_len = input->shape[0];
    size_t embed_dim = input->shape[1];
    
    // Check output shape
    if (output->ndim != 2) {
        BLT_FATAL("Transformer: output must be 2D");
    }
    if (output->shape[0] != seq_len || output->shape[1] != embed_dim) {
        BLT_FATAL("Transformer: output shape must match input [seq_len, embed_dim]");
    }
    
    // Check dtypes
    if (input->dtype != BLT_DTYPE_FP32) {
        BLT_FATAL("Transformer: input must be FP32");
    }
    if (output->dtype != BLT_DTYPE_FP32) {
        BLT_FATAL("Transformer: output must be FP32");
    }
    
    // Check norm1
    if (norm1_weight->ndim != 1 || norm1_weight->shape[0] != embed_dim) {
        BLT_FATAL("Transformer: norm1_weight must be shape [embed_dim]");
    }
    if (norm1_bias->ndim != 1 || norm1_bias->shape[0] != embed_dim) {
        BLT_FATAL("Transformer: norm1_bias must be shape [embed_dim]");
    }
    
    // Check norm2
    if (norm2_weight->ndim != 1 || norm2_weight->shape[0] != embed_dim) {
        BLT_FATAL("Transformer: norm2_weight must be shape [embed_dim]");
    }
    if (norm2_bias->ndim != 1 || norm2_bias->shape[0] != embed_dim) {
        BLT_FATAL("Transformer: norm2_bias must be shape [embed_dim]");
    }
    
    // Check FFN layers
    if (ffn_up_w->ndim != 2) {
        BLT_FATAL("Transformer: ffn_up_w must be 2D");
    }
    if (ffn_up_w->shape[0] != embed_dim) {
        BLT_FATAL("Transformer: ffn_up_w shape[0] must equal embed_dim");
    }
    
    size_t hidden_dim = ffn_up_w->shape[1];
    if (hidden_dim != config->hidden_dim) {
        BLT_FATAL("Transformer: ffn_up_w hidden dimension mismatch with config");
    }
    
    if (ffn_down_w->ndim != 2) {
        BLT_FATAL("Transformer: ffn_down_w must be 2D");
    }
    if (ffn_down_w->shape[0] != hidden_dim || ffn_down_w->shape[1] != embed_dim) {
        BLT_FATAL("Transformer: ffn_down_w shape must be [hidden_dim, embed_dim]");
    }
    
    if (config->layer_norm_eps < 1e-12f) {
        BLT_FATAL("Transformer: layer_norm_eps is too small");
    }
    
    // =====================
    // Allocate Intermediate Tensors
    // =====================
    size_t seq_embed_bytes = seq_len * embed_dim * sizeof(float);
    size_t seq_hidden_bytes = seq_len * hidden_dim * sizeof(float);
    
    float* normalized_input = (float*)malloc(seq_embed_bytes);
    if (!normalized_input) {
        BLT_FATAL("Transformer: failed to allocate normalized_input");
    }
    
    float* attention_output = (float*)malloc(seq_embed_bytes);
    if (!attention_output) {
        free(normalized_input);
        BLT_FATAL("Transformer: failed to allocate attention_output");
    }
    
    float* attn_residual = (float*)malloc(seq_embed_bytes);
    if (!attn_residual) {
        free(normalized_input);
        free(attention_output);
        BLT_FATAL("Transformer: failed to allocate attn_residual");
    }
    
    float* normalized_attn = (float*)malloc(seq_embed_bytes);
    if (!normalized_attn) {
        free(normalized_input);
        free(attention_output);
        free(attn_residual);
        BLT_FATAL("Transformer: failed to allocate normalized_attn");
    }
    
    float* ffn_hidden = (float*)malloc(seq_hidden_bytes);
    if (!ffn_hidden) {
        free(normalized_input);
        free(attention_output);
        free(attn_residual);
        free(normalized_attn);
        BLT_FATAL("Transformer: failed to allocate ffn_hidden");
    }
    
    float* ffn_activated = (float*)malloc(seq_hidden_bytes);
    if (!ffn_activated) {
        free(normalized_input);
        free(attention_output);
        free(attn_residual);
        free(normalized_attn);
        free(ffn_hidden);
        BLT_FATAL("Transformer: failed to allocate ffn_activated");
    }
    
    // =====================
    // Create Tensor Wrappers
    // =====================
    blt_tensor norm_input_t = {
        .data = normalized_input,
        .shape = {seq_len, embed_dim, 0, 0},
        .strides = {embed_dim, 1, 0, 0},
        .ndim = 2,
        .numel = seq_len * embed_dim,
        .dtype = BLT_DTYPE_FP32,
        .backend = input->backend,
        .is_view = true
    };
    
    blt_tensor attn_output_t = {
        .data = attention_output,
        .shape = {seq_len, embed_dim, 0, 0},
        .strides = {embed_dim, 1, 0, 0},
        .ndim = 2,
        .numel = seq_len * embed_dim,
        .dtype = BLT_DTYPE_FP32,
        .backend = input->backend,
        .is_view = true
    };
    
    blt_tensor attn_residual_t = {
        .data = attn_residual,
        .shape = {seq_len, embed_dim, 0, 0},
        .strides = {embed_dim, 1, 0, 0},
        .ndim = 2,
        .numel = seq_len * embed_dim,
        .dtype = BLT_DTYPE_FP32,
        .backend = input->backend,
        .is_view = true
    };
    
    blt_tensor norm_attn_t = {
        .data = normalized_attn,
        .shape = {seq_len, embed_dim, 0, 0},
        .strides = {embed_dim, 1, 0, 0},
        .ndim = 2,
        .numel = seq_len * embed_dim,
        .dtype = BLT_DTYPE_FP32,
        .backend = input->backend,
        .is_view = true
    };
    
    blt_tensor ffn_hidden_t = {
        .data = ffn_hidden,
        .shape = {seq_len, hidden_dim, 0, 0},
        .strides = {hidden_dim, 1, 0, 0},
        .ndim = 2,
        .numel = seq_len * hidden_dim,
        .dtype = BLT_DTYPE_FP32,
        .backend = input->backend,
        .is_view = true
    };
    
    blt_tensor ffn_activated_t = {
        .data = ffn_activated,
        .shape = {seq_len, hidden_dim, 0, 0},
        .strides = {hidden_dim, 1, 0, 0},
        .ndim = 2,
        .numel = seq_len * hidden_dim,
        .dtype = BLT_DTYPE_FP32,
        .backend = input->backend,
        .is_view = true
    };
    
    // =====================
    // Step 1: Pre-Norm LayerNorm
    // =====================
    blt_layer_norm(
        input,
        norm1_weight,
        norm1_bias,
        &norm_input_t,
        config->layer_norm_eps
    );
    
    // =====================
    // Step 2: Multi-Head Attention
    // =====================
    blt_multihead_attention(
        &norm_input_t,
        attn_qkv_w,
        attn_proj_w,
        &attn_output_t,
        &config->attn_config,
        NULL
    );
    
    // =====================
    // Step 3: First Residual Connection (attention + input)
    // =====================
    blt_add(input, &attn_output_t, &attn_residual_t);
    
    // =====================
    // Step 4: Pre-Norm for FFN
    // =====================
    blt_layer_norm(
        &attn_residual_t,
        norm2_weight,
        norm2_bias,
        &norm_attn_t,
        config->layer_norm_eps
    );
    
    // =====================
    // Step 5: FFN Up Projection
    // =====================
    blt_matmul(&norm_attn_t, ffn_up_w, &ffn_hidden_t);
    
    // =====================
    // Step 6: GELU Activation
    // =====================
    memcpy(ffn_activated, ffn_hidden, seq_hidden_bytes);
    blt_gelu(&ffn_activated_t);
    
    // =====================
    // Step 7: FFN Down Projection
    // =====================
    blt_matmul(&ffn_activated_t, ffn_down_w, output);
    
    // =====================
    // Step 8: Final Residual Connection (FFN output + attention residual)
    // =====================
    blt_add(&attn_residual_t, output, output);
    
    // =====================
    // Cleanup
    // =====================
    free(normalized_input);
    free(attention_output);
    free(attn_residual);
    free(normalized_attn);
    free(ffn_hidden);
    free(ffn_activated);
}