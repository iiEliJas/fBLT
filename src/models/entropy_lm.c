#include "blt/core/backend.h"
#include "blt/models/entropy_lm.h"
#include "blt/models/transformer_stack.h"
#include "blt/models/byte_embedding.h"
#include "blt/ops/matmul.h"
#include "blt/ops/elementwise.h"
#include "blt/ops/cross_entropy.h"


//----------------------------------------------------------------------
// LM Create

blt_entropy_lm* blt_entropy_lm_create(blt_arena* arena, const blt_entropy_lm_config* config) {
    BLT_REQUIRE(arena != NULL && config != NULL,
        "blt_entropy_lm_create: arena and config cannot be NULL");

    blt_entropy_lm* model = (blt_entropy_lm*)blt_arena_alloc(arena, sizeof(blt_entropy_lm), sizeof(void*));
    BLT_REQUIRE(model != NULL, "blt_entropy_lm_create: failed to allocate model struct");
    model->config = *config;

    size_t embed_dim = config->embed_dim;

    blt_transformer_stack_config stack_cfg = {
        .num_layers  = config->num_layers,
        .embed_dim   = config->embed_dim,
        .hidden_dim  = config->hidden_dim,
        .num_heads   = config->num_heads,
        .max_seq_len = config->max_seq_len,
        .rope_theta  = config->rope_theta,
    };
    blt_transformer_stack_init(arena, &model->stack, &stack_cfg);

    // Byte embedding table [256, embed_dim].
    size_t emb_shape[2] = { 256, embed_dim };
    model->embedding_weight = blt_tensor_create(arena, emb_shape, 2, BLT_DTYPE_FP32);

    // LM head [embed_dim, 256].
    size_t head_shape[2] = { embed_dim, 256 };
    model->lm_head_weight = blt_tensor_create(arena, head_shape, 2, BLT_DTYPE_FP32);

    return model;
}



blt_entropy_lm_grad* blt_entropy_lm_grad_create(blt_arena* arena, const blt_entropy_lm* model) {
    BLT_REQUIRE(arena != NULL && model != NULL,
        "blt_entropy_lm_grad_create: arena and model cannot be NULL");

    blt_entropy_lm_grad* grad = (blt_entropy_lm_grad*)blt_arena_alloc(arena, sizeof(blt_entropy_lm_grad), sizeof(void*));
    BLT_REQUIRE(grad != NULL, "blt_entropy_lm_grad_create: failed to allocate grad struct");

    size_t embed_dim = model->config.embed_dim;

    size_t emb_shape[2] = { 256, embed_dim };
    grad->embedding_grad = blt_tensor_create(arena, emb_shape, 2, BLT_DTYPE_FP32);

    size_t head_shape[2] = { embed_dim, 256 };
    grad->lm_head_grad = blt_tensor_create(arena, head_shape, 2, BLT_DTYPE_FP32);

    grad->stack_grad = blt_transformer_stack_grad_create(arena, &model->stack);

    return grad;
}



//----------------------------------------------------------------------
// LM forward path
//
// Pipeline steps:
// 1. Embedding Lookup:     x = Embedding(bytes_in) [seq_len, embed_dim]
// 2. Transformer Stack:    h = Stack(x)             [seq_len, embed_dim]
// 3. LM Projection:        Z = h * W_head           [seq_len, 256]
// 4. Shifted Loss:         L = CrossEntropy(Z[0..N-2], bytes_in[1..N-1])
//
// Shifted target handling: target[t] = bytes[t+1]
// -> the prediction distribution at index t predicts the byte at position t+1
// the loss operates over (seq_len - 1) positions

void blt_entropy_lm_forward(
    const blt_entropy_lm* model,
    const blt_tensor* bytes_in,
    blt_tensor* logits_out,
    blt_tensor* loss_out,
    blt_arena* arena
) {
    BLT_REQUIRE(model != NULL && bytes_in != NULL && logits_out != NULL && loss_out != NULL && arena != NULL,
        "blt_entropy_lm_forward: arguments cannot be NULL");
    BLT_REQUIRE(bytes_in->ndim == 1 && bytes_in->dtype == BLT_DTYPE_UINT8,
        "blt_entropy_lm_forward: bytes_in must be a 1D UINT8 tensor");

    size_t seq_len = bytes_in->shape[0];
    BLT_REQUIRE(seq_len >= 2, "blt_entropy_lm_forward: seq_len must be >= 2 (need a shifted target)");
    BLT_REQUIRE(seq_len <= model->config.max_seq_len,
        "blt_entropy_lm_forward: seq_len exceeds the model's max_seq_len (RoPE cache too small)");

    size_t embed_dim = model->config.embed_dim;
    blt_tensor rope_cos_view, rope_sin_view;
    blt_transformer_config layer_cfg = blt_transformer_stack_call_config(
        &model->stack, seq_len, &rope_cos_view, &rope_sin_view);


    // -----------------------------------------------------------------
    // STEP 1: Byte Embedding Lookup
    // Maps discrete byte values x_t in range [0, 255] to continuous vector
    // representations e_t of dimension (embed_dim) using embedding matrix 
    // W_emb of size (256 x embed_dim):
    //     e_t = W_emb[x_t]
    // Output tensor shape: [seq_len, embed_dim]
    // -----------------------------------------------------------------
    blt_byte_embedding emb = { .vocab_size = model->embedding_weight.shape[0], .weight = model->embedding_weight, .embed_dim = embed_dim };
    size_t x_shape[2] = { seq_len, embed_dim };
    blt_tensor x = blt_tensor_create(arena, x_shape, 2, BLT_DTYPE_FP32);
    blt_byte_embedding_forward(&emb, bytes_in, &x);


    // -----------------------------------------------------------------
    // STEP 2: Transformer Stack
    // Updates byte representations through N Transformer layers.
    // At step t, the output vector h_t encodes the history of past bytes
    // (x_1, ..., x_t) with causal self-attention:
    //     h_t = Transformer(e_1, ..., e_t)
    // Output tensor shape remains: [seq_len, embed_dim]
    // -----------------------------------------------------------------
    blt_tensor h;
    blt_transformer_stack_forward(&model->stack, &x, &layer_cfg, seq_len, &h, arena);


    // -----------------------------------------------------------------
    // STEP 3: Linear projection to unnormalized logits
    // Projects hidden context vectors h_t to raw prediction scores (logits)
    // z_t across the 256 possible bytes in vocabulary V:
    //     z_t = h_t * W_head  where W_head is size (embed_dim x 256)
    // Output tensor shape: [seq_len, 256]
    // -----------------------------------------------------------------
    size_t logits_shape[2] = { seq_len, 256 };
    blt_tensor logits_full = blt_tensor_create(arena, logits_shape, 2, BLT_DTYPE_FP32);
    blt_matmul(&h, &model->lm_head_weight, &logits_full);

    BLT_REQUIRE(logits_out->ndim == 2 && logits_out->shape[0] == seq_len && logits_out->shape[1] == 256,
        "blt_entropy_lm_forward: logits_out must be [seq_len, 256] FP32");
    memcpy(logits_out->data, logits_full.data, blt_tensor_bytes(&logits_full));


    // -----------------------------------------------------------------
    // STEP 4: Shifted Target Cross-Entropy Loss
    // Evaluates auto-regressive prediction quality.
    //
    // For position t in range [0, seq_len - 2]:
    //   - Inputs:  logits z_t predicting byte at position (t + 1)
    //   - Target:  true byte value y_t = bytes_in[t + 1]
    //
    // Converts logits to probabilities with Softmax:
    //     p(v | past bytes) = exp(z_t,v) / SUM_k(exp(z_t,k))
    //
    // Computes negative log-likelyhood loss (Cross-Entropy):
    //     Loss = - (1 / (N - 1)) * SUM_t(log(p(x_(t+1) | past bytes)))
    // -----------------------------------------------------------------

    // Slice logits for positions 0 to (seq_len - 2)
    blt_tensor shifted_logits;
    blt_tensor_view_2d(&shifted_logits, logits_full.data, seq_len - 1, 256, h.backend);

    // Slice target byte IDs for positions 1 to (seq_len - 1)
    blt_tensor shifted_targets;
    size_t elem_size = blt_dtype_sizeof(bytes_in->dtype);
    view_1d(&shifted_targets, (char*)bytes_in->data + elem_size, seq_len - 1,
            bytes_in->dtype, bytes_in->backend);

    // Compute cross-entropy loss over shifted sequence
    blt_cross_entropy_forward(&shifted_logits, &shifted_targets, loss_out);
}



//----------------------------------------------------------------------
// LM backward path
//
// Mirrors forward call order in reverse

void blt_entropy_lm_backward(
    const blt_entropy_lm* model,
    const blt_tensor* bytes_in,
    blt_entropy_lm_grad* grad_out,
    blt_arena* arena
) {
    BLT_REQUIRE(model != NULL && bytes_in != NULL && grad_out != NULL && arena != NULL,
        "blt_entropy_lm_backward: arguments cannot be NULL");
    BLT_REQUIRE(bytes_in->ndim == 1 && bytes_in->dtype == BLT_DTYPE_UINT8,
        "blt_entropy_lm_backward: bytes_in must be a 1D UINT8 tensor");

    size_t seq_len = bytes_in->shape[0];
    BLT_REQUIRE(seq_len >= 2, "blt_entropy_lm_backward: seq_len must be >= 2 (need a shifted target)");
    BLT_REQUIRE(seq_len <= model->config.max_seq_len,
        "blt_entropy_lm_backward: seq_len exceeds the model's max_seq_len (RoPE cache too small)");

    size_t embed_dim = model->config.embed_dim;

    blt_tensor rope_cos_view, rope_sin_view;
    blt_transformer_config layer_cfg = blt_transformer_stack_call_config(
        &model->stack, seq_len, &rope_cos_view, &rope_sin_view);


    // ----------------
    // Recompute forward with caching
    blt_byte_embedding emb = { .vocab_size = model->embedding_weight.shape[0], .weight = model->embedding_weight, .embed_dim = embed_dim };
    size_t x_shape[2] = { seq_len, embed_dim };
    blt_tensor x0 = blt_tensor_create(arena, x_shape, 2, BLT_DTYPE_FP32);
    blt_byte_embedding_forward(&emb, bytes_in, &x0);

    blt_tensor final_x;
    blt_transformer_stack_cache* stack_cache = blt_transformer_stack_forward_cached(
        &model->stack, &x0, &layer_cfg, seq_len, arena, &final_x);

    size_t logits_shape[2] = { seq_len, 256 };
    blt_tensor logits_full = blt_tensor_create(arena, logits_shape, 2, BLT_DTYPE_FP32);
    blt_matmul(&final_x, &model->lm_head_weight, &logits_full);

    blt_tensor shifted_logits;
    blt_tensor_view_2d(&shifted_logits, logits_full.data, seq_len - 1, 256, final_x.backend);

    blt_tensor shifted_targets;
    size_t elem_size = blt_dtype_sizeof(bytes_in->dtype);
    view_1d(&shifted_targets, (char*)bytes_in->data + elem_size, seq_len - 1,
            bytes_in->dtype, bytes_in->backend);


    // ----------------
    // Backward: loss -> LM head -> transformer stack -> embedding
    size_t shifted_logits_shape[2] = { seq_len - 1, 256 };
    blt_tensor grad_shifted_logits = blt_tensor_create(arena, shifted_logits_shape, 2, BLT_DTYPE_FP32);
    blt_cross_entropy_backward(&shifted_logits, &shifted_targets, &grad_shifted_logits);

    // The final logits row (predicting one byte past the sequence) never contributed to the loss
    // => the incoming gradient is zero
    blt_tensor grad_logits_full = blt_tensor_create(arena, logits_shape, 2, BLT_DTYPE_FP32);
    zero_tensor(&grad_logits_full);
    memcpy(grad_logits_full.data, grad_shifted_logits.data, blt_tensor_bytes(&grad_shifted_logits));

    blt_tensor grad_final_x = blt_tensor_create(arena, x_shape, 2, BLT_DTYPE_FP32);
    blt_matmul_backward(&final_x, &model->lm_head_weight, &grad_logits_full,
                         &grad_final_x, &grad_out->lm_head_grad);

    blt_tensor grad_x0;
    blt_transformer_stack_backward(&model->stack, stack_cache, &layer_cfg, seq_len,
                                    &grad_final_x, grad_out->stack_grad, &grad_x0, arena);

    // grad_x0 is now dL/d(embedding output)
    // scatter add into the embedding tables gradient
    blt_byte_embedding_backward(&emb, bytes_in, &grad_x0, &grad_out->embedding_grad);
}