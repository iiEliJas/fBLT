#ifndef BLT_MODEL_ENTROPY_LM_H
#define BLT_MODEL_ENTROPY_LM_H

#include "blt/core/tensor.h"
#include "blt/core/allocator.h"
#include "blt/models/transformer.h"

// -----------------------------------------------------------------------
// blt_entropy_lm — the Phase 1 "tiny causal byte LM": byte embedding ->
// N transformer layers (RMSNorm + SwiGLU + RoPE) -> LM head -> next-byte
// cross-entropy loss. This is pure assembly of already-implemented,
// individually-tested pieces (see API_REFERENCE.md); no new math lives
// in the model beyond correct wiring and shift-by-one target handling.
// -----------------------------------------------------------------------

typedef struct {
    size_t embed_dim;      // transformer hidden width
    size_t num_layers;      // number of stacked transformer blocks
    size_t hidden_dim;      // FFN intermediate width
    size_t num_heads;       // attention heads (embed_dim must be divisible)
    size_t max_seq_len;     // upper bound used to size the shared RoPE cache
    float  rope_theta;      // RoPE base (e.g. 500000.0f)
} blt_entropy_lm_config;

// Owned storage for one transformer layer's weights. A parallel
// blt_transformer_weights entry (of const pointers into this struct) is
// what actually gets passed to blt_transformer_forward.
typedef struct {
    blt_tensor norm1_weight;
    blt_tensor attn_qkv_w;
    blt_tensor attn_proj_w;
    blt_tensor norm2_weight;
    blt_tensor ffn_up_w;
    blt_tensor ffn_gate_w;
    blt_tensor ffn_down_w;
} blt_transformer_layer_storage;

typedef struct {
    blt_entropy_lm_config config;

    blt_tensor embedding_weight;   // [256, embed_dim]
    blt_tensor lm_head_weight;     // [embed_dim, 256]

    // RoPE cos/sin tables, precomputed once at model-init time for
    // max_seq_len and shared (by pointer, via layer_config) across every layer
    blt_tensor rope_cos_cache;
    blt_tensor rope_sin_cache;

    blt_transformer_config layer_config;                // shared by every layer
    blt_transformer_layer_storage* layer_storage;       // owns the tensors, [num_layers]
    blt_transformer_weights* layer_weights;             // const pointer views into layer_storage, [num_layers]
} blt_entropy_lm;

// Gradient part of blt_transformer_layer_storage
typedef struct {
    blt_tensor norm1_weight;
    blt_tensor attn_qkv_w;
    blt_tensor attn_proj_w;
    blt_tensor norm2_weight;
    blt_tensor ffn_up_w;
    blt_tensor ffn_gate_w;
    blt_tensor ffn_down_w;
} blt_transformer_layer_grad;

// Gradient part of blt_entropy_lm, one tensor per weight
typedef struct {
    blt_tensor embedding_grad;                    // [256, embed_dim]
    blt_transformer_layer_grad* layer_grads;       // [num_layers]
    blt_tensor lm_head_grad;                       // [embed_dim, 256]
} blt_entropy_lm_grad;



// Allocates and zero init every weight tensor in model
// Caller fills in weight data afterward
blt_entropy_lm* blt_entropy_lm_create(blt_arena* arena, const blt_entropy_lm_config* config);

// Allocates and zero init a matching gradient struct for model
blt_entropy_lm_grad* blt_entropy_lm_grad_create(blt_arena* arena, const blt_entropy_lm* model);


// Forward pass: embedding -> N transformer layers -> LM head -> shifted
// next-byte cross-entropy loss
// targets[t] = bytes_in[t+1] so loss and logits_out are computed/considered over seq_len-1 positions internally
// logits_out itself is the full [seq_len, 256] logits tensor
//      bytes_in:   [seq_len] UINT8, seq_len >= 2, seq_len <= config.max_seq_len
//      logits_out: [seq_len, 256] FP32, caller-allocated
//      loss_out:   scalar FP32, caller-allocated
void blt_entropy_lm_forward(
    const blt_entropy_lm* model,
    const blt_tensor* bytes_in,
    blt_tensor* logits_out,
    blt_tensor* loss_out,
    blt_arena* arena
);

// Backward pass for the same forward computation above. Recomputes forward internally
// since blt_transformer_forward does not save its intermediates
// Every gradient is produced by reusing an already-implemented op backward
void blt_entropy_lm_backward(
    const blt_entropy_lm* model,
    const blt_tensor* bytes_in,
    blt_entropy_lm_grad* grad_out,
    blt_arena* arena
);

#endif // BLT_MODEL_ENTROPY_LM_H