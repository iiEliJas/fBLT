#include "blt/models/transformer.h"
#include "blt/core/backend.h"
#include "blt/ops/elementwise.h"
#include "blt/ops/matmul.h"
#include "blt/ops/gelu.h"
#include "blt/ops/swiglu.h"
#include "blt/ops/layernorm.h"
#include "blt/ops/rmsnorm.h"
#include "blt/ops/rope.h"


//--------------------------------
// Validation

static void validate_transformer_call(
    const blt_tensor* input,
    const blt_transformer_weights* w,
    const blt_tensor* output,
    const blt_transformer_config* config,
    size_t* out_seq_len,
    size_t* out_embed_dim,
    size_t* out_hidden_dim
) {
    BLT_REQUIRE(w != NULL, "Transformer: weights cannot be NULL");
    BLT_REQUIRE(config != NULL, "Transformer: config cannot be NULL");
 
    blt_check_nd_fp32(input, 2, (const size_t[]){0, 0}, "Transformer: input must be 2D [seq_len, embed_dim] FP32");
    size_t seq_len = input->shape[0];
    size_t embed_dim = input->shape[1];
 
    blt_check_nd_fp32(output, 2, (const size_t[]){seq_len, embed_dim}, "Transformer: output shape must match input [seq_len, embed_dim]");
 
    blt_check_nd_fp32(w->norm1_weight, 1, (const size_t[]){embed_dim}, "Transformer: norm1_weight must be [embed_dim] FP32");
    blt_check_nd_fp32(w->norm2_weight, 1, (const size_t[]){embed_dim}, "Transformer: norm2_weight must be [embed_dim] FP32");
 
    if (config->norm_type == BLT_NORM_LAYERNORM) {
        blt_check_nd_fp32(w->norm1_bias, 1, (const size_t[]){embed_dim}, "Transformer: norm1_bias must be [embed_dim] FP32 for LayerNorm");
        blt_check_nd_fp32(w->norm2_bias, 1, (const size_t[]){embed_dim}, "Transformer: norm2_bias must be [embed_dim] FP32 for LayerNorm");
        BLT_REQUIRE(config->layer_norm_eps >= 1e-12f, "Transformer: layer_norm_eps is too small");
    }
 
    BLT_REQUIRE(w->attn_qkv_w != NULL && w->attn_proj_w != NULL,
                "Transformer: attn_qkv_w and attn_proj_w cannot be NULL");
 
    BLT_REQUIRE(w->ffn_up_w != NULL && w->ffn_up_w->ndim == 2 && w->ffn_up_w->shape[0] == embed_dim,
                "Transformer: ffn_up_w must be [embed_dim, hidden_dim]");
    size_t hidden_dim = w->ffn_up_w->shape[1];
    BLT_REQUIRE(hidden_dim == config->hidden_dim, "Transformer: ffn_up_w hidden dimension mismatch with config");
 
    if (config->activation_type == BLT_ACTIVATION_SWIGLU) {
        blt_check_nd_fp32(w->ffn_gate_w, 2, (const size_t[]){embed_dim, hidden_dim},
                           "Transformer: ffn_gate_w must be [embed_dim, hidden_dim] FP32 for SwiGLU");
    }
 
    blt_check_nd_fp32(w->ffn_down_w, 2, (const size_t[]){hidden_dim, embed_dim}, 
                            "Transformer: ffn_down_w must be [hidden_dim, embed_dim] FP32");
 
    *out_seq_len = seq_len;
    *out_embed_dim = embed_dim;
    *out_hidden_dim = hidden_dim;
}
 


//--------------------------------
// Norm

static void apply_norm1(const blt_tensor* input, const blt_transformer_weights* w,
                         const blt_transformer_config* config, blt_tensor* out) {
    if (config->norm_type == BLT_NORM_RMSNORM) {
        blt_rmsnorm_forward(input, w->norm1_weight, out);
    } else {
        blt_layernorm_forward(input, w->norm1_weight, w->norm1_bias, out, config->layer_norm_eps);
    }
}
 
static void apply_norm2(const blt_tensor* input, const blt_transformer_weights* w,
                         const blt_transformer_config* config, blt_tensor* out) {
    if (config->norm_type == BLT_NORM_RMSNORM) {
        blt_rmsnorm_forward(input, w->norm2_weight, out);
    } else {
        blt_layernorm_forward(input, w->norm2_weight, w->norm2_bias, out, config->layer_norm_eps);
    }
}
 
// FFN: up projection -> activation -> down projection
static void apply_ffn(const blt_tensor* norm_attn, const blt_transformer_weights* w,
                       const blt_transformer_config* config, blt_arena* arena,
                       size_t seq_len, size_t hidden_dim, blt_tensor* ffn_out) {
    size_t shape[2] = { seq_len, hidden_dim };
    blt_tensor up_proj = blt_tensor_create(arena, shape, 2, BLT_DTYPE_FP32);
    blt_matmul(norm_attn, w->ffn_up_w, &up_proj);
 
    blt_tensor activated = blt_tensor_create(arena, shape, 2, BLT_DTYPE_FP32);
 
    if (config->activation_type == BLT_ACTIVATION_SWIGLU) {
        blt_tensor gate_proj = blt_tensor_create(arena, shape, 2, BLT_DTYPE_FP32);
        blt_matmul(norm_attn, w->ffn_gate_w, &gate_proj);
        blt_swiglu_forward(&gate_proj, &up_proj, &activated);
    } else {
        blt_gelu_forward(&up_proj, &activated);
    }
 
    blt_matmul(&activated, w->ffn_down_w, ffn_out);
}



//----------------------------------------------------------------
// Transformer Block Forward Pass
//
// Transformer Layer:
// 1. Attention Sub-layer:
//      Norm1_out     = Norm1(Input)                     [seq_len, embed_dim]
//      Attn_out      = MultiHeadAttention(Norm1_out)    [seq_len, embed_dim]
//      Attn_residual = Input + Attn_out                 [seq_len, embed_dim]
//
// 2. Feed-Forward (FFN) Sub-layer:
//      Norm2_out     = Norm2(Attn_residual)             [seq_len, embed_dim]
//      FFN_out       = FFN(Norm2_out)                   [seq_len, embed_dim]
//      Output        = Attn_residual + FFN_out          [seq_len, embed_dim]

void blt_transformer_forward(
    const blt_tensor* input,
    const blt_transformer_weights* weights,
    blt_tensor* output,
    const blt_transformer_config* config,
    blt_arena* arena
) {
    size_t seq_len, embed_dim, hidden_dim;
    validate_transformer_call(input, weights, output, config, &seq_len, &embed_dim, &hidden_dim);
    BLT_REQUIRE(arena != NULL, "Transformer: arena cannot be NULL");
 
    size_t embed_shape[2] = { seq_len, embed_dim };
 
    
    // -----------------------------------------------------------------
    // STEP 1: Multi-Head Attention Sub-Layer with Pre-Norm & Residual
    //
    // Apply pre-layer normalization (RMSNorm or LayerNorm)
    blt_tensor norm_input = blt_tensor_create(arena, embed_shape, 2, BLT_DTYPE_FP32);
    apply_norm1(input, weights, config, &norm_input);
 
    // Multi-Head Attention forward pass
    blt_tensor attn_out = blt_tensor_create(arena, embed_shape, 2, BLT_DTYPE_FP32);
    blt_multihead_attention(&norm_input, weights->attn_qkv_w, weights->attn_proj_w,
                             &attn_out, &config->attn_config, arena);
 
    // Add residual connection: Attn_residual = Input + Attn_out
    blt_tensor attn_residual = blt_tensor_create(arena, embed_shape, 2, BLT_DTYPE_FP32);
    blt_add(input, &attn_out, &attn_residual);
 

    // -----------------------------------------------------------------
    // STEP 2: Feed-Forward Network (FFN) Sub-Layer with Pre-Norm & Residual
    //
    // Apply second pre-layer normalization
    blt_tensor norm_attn = blt_tensor_create(arena, embed_shape, 2, BLT_DTYPE_FP32);
    apply_norm2(&attn_residual, weights, config, &norm_attn);
 
    // Feed-Forward Network forward pass (Up-projection -> Activation -> Down-projection)
    // Writes intermediate result into output
    apply_ffn(&norm_attn, weights, config, arena, seq_len, hidden_dim, output);

    // Add second residual connection: Output = Attn_residual + FFN_out
    blt_add(&attn_residual, output, output);
}