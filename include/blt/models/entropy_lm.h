#ifndef BLT_MODEL_ENTROPY_LM_H
#define BLT_MODEL_ENTROPY_LM_H

#ifdef __cplusplus
extern "C" {
#endif

#include "blt/core/tensor.h"
#include "blt/core/allocator.h"
#include "blt/models/transformer.h"
#include "blt/models/transformer_stack.h"

// Entropy LM
//
// byte embedding -> N transformer layers (RMSNorm + SwiGLU + RoPE) -> LM head ->
// next-byte cross-entropy loss

typedef struct {
    size_t embed_dim;   // transformer hidden width
    size_t num_layers;  // number of stacked transformer blocks
    size_t hidden_dim;  // FFN intermediate width
    size_t num_heads;   // attention heads (embed_dim must be divisible)
    size_t max_seq_len; // upper bound used to size the shared RoPE cache
    float rope_theta;   // RoPE base (e.g. 500000.0f)
} blt_entropy_lm_config;

typedef struct {
    blt_entropy_lm_config config;

    blt_tensor embedding_weight; // [256, embed_dim]
    blt_tensor lm_head_weight;   // [embed_dim, 256]

    blt_transformer_stack stack; // N transformer layers with RoPE
} blt_entropy_lm;

// Gradient part of blt_entropy_lm, one tensor per weight
typedef struct {
    blt_tensor embedding_grad;              // [256, embed_dim]
    blt_transformer_stack_grad *stack_grad; // [num_layers]
    blt_tensor lm_head_grad;                // [embed_dim, 256]
} blt_entropy_lm_grad;

// Allocates and zero-inits every weight tensor. Caller fills in data.
blt_entropy_lm *blt_entropy_lm_create(blt_arena *arena, const blt_entropy_lm_config *config);

// Allocates a matching gradient struct for model.
blt_entropy_lm_grad *blt_entropy_lm_grad_create(blt_arena *arena, const blt_entropy_lm *model);

// Forward: embedding -> N transformer layers -> LM head -> shifted next-byte CE loss.
// targets[t] = bytes_in[t+1], so loss is over seq_len-1 positions internally.
// logits_out is the full [seq_len, 256] logits tensor.
//   bytes_in:   [seq_len] UINT8, seq_len >= 2, seq_len <= config.max_seq_len
//   logits_out: [seq_len, 256] FP32, caller-allocated
//   loss_out:   scalar FP32, caller-allocated
void blt_entropy_lm_forward(const blt_entropy_lm *model, const blt_tensor *bytes_in, blt_tensor *logits_out,
                            blt_tensor *loss_out, blt_arena *arena);

// Backward pass for the forward above. Recomputes forward internally.
void blt_entropy_lm_backward(const blt_entropy_lm *model, const blt_tensor *bytes_in, blt_entropy_lm_grad *grad_out,
                             blt_arena *arena);

#ifdef __cplusplus
}
#endif

#endif // BLT_MODEL_ENTROPY_LM_H