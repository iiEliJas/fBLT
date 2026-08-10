#include "blt/core/backend.h"
#include "blt/models/entropy_lm.h"
#include "blt/models/attention.h"
#include "blt/models/transformer.h"
#include "blt/models/byte_embedding.h"
#include "blt/ops/matmul.h"
#include "blt/ops/elementwise.h"
#include "blt/ops/rmsnorm.h"
#include "blt/ops/swiglu.h"
#include "blt/ops/rope.h"
#include "blt/ops/optim.h"
#include "blt/ops/cross_entropy.h"



//----------------------------------------------------------------------
// Config helper

static blt_transformer_config make_seq_layer_config(
    const blt_entropy_lm* model, size_t seq_len, blt_tensor* cos_view, blt_tensor* sin_view
) {
    size_t head_dim = model->config.embed_dim / model->config.num_heads;
    size_t half = head_dim / 2;
 
    blt_tensor_view_2d(cos_view, model->rope_cos_cache.data, seq_len, half, model->rope_cos_cache.backend);
    blt_tensor_view_2d(sin_view, model->rope_sin_cache.data, seq_len, half, model->rope_sin_cache.backend);
 
    blt_transformer_config cfg = model->layer_config;   // copy configs
    cfg.attn_config.rope_cos_cache = cos_view;
    cfg.attn_config.rope_sin_cache = sin_view;
    return cfg;
}



//----------------------------------------------------------------------
// LM Create
 
blt_entropy_lm* blt_entropy_lm_create(blt_arena* arena, const blt_entropy_lm_config* config) {
    BLT_REQUIRE(arena != NULL && config != NULL,
        "blt_entropy_lm_create: arena and config cannot be NULL");
    BLT_REQUIRE(config->num_layers > 0, "blt_entropy_lm_create: num_layers must be > 0");
    BLT_REQUIRE(config->num_heads > 0 && config->embed_dim % config->num_heads == 0,
        "blt_entropy_lm_create: embed_dim must be divisible by num_heads");
    BLT_REQUIRE(config->max_seq_len > 0, "blt_entropy_lm_create: max_seq_len must be > 0");
 
    blt_entropy_lm* model = (blt_entropy_lm*)blt_arena_alloc(arena, sizeof(blt_entropy_lm), sizeof(void*));
    BLT_REQUIRE(model != NULL, "blt_entropy_lm_create: failed to allocate model struct");
    model->config = *config;
 
    size_t embed_dim  = config->embed_dim;
    size_t hidden_dim = config->hidden_dim;
    size_t head_dim   = embed_dim / config->num_heads;
 
    // Byte embedding table [256, embed_dim].
    size_t emb_shape[2] = { 256, embed_dim };
    model->embedding_weight = blt_tensor_create(arena, emb_shape, 2, BLT_DTYPE_FP32);
 
    // LM head [embed_dim, 256].
    size_t head_shape[2] = { embed_dim, 256 };
    model->lm_head_weight = blt_tensor_create(arena, head_shape, 2, BLT_DTYPE_FP32);
 
    // Shared RoPE cache is computed once (with models max_seq_length)
    size_t half = head_dim / 2;
    size_t rope_shape[2] = { config->max_seq_len, half };
    model->rope_cos_cache = blt_tensor_create(arena, rope_shape, 2, BLT_DTYPE_FP32);
    model->rope_sin_cache = blt_tensor_create(arena, rope_shape, 2, BLT_DTYPE_FP32);
    blt_rope_config rope_cfg = { .theta = config->rope_theta, .head_dim = head_dim };
    blt_rope_precompute(config->max_seq_len, &rope_cfg, &model->rope_cos_cache, &model->rope_sin_cache);
 
    // shared transformer config for every layer
    model->layer_config.attn_config.embed_dim = embed_dim;
    model->layer_config.attn_config.num_heads = config->num_heads;
    model->layer_config.attn_config.head_dim = head_dim;
    model->layer_config.attn_config.is_causal = true;
    model->layer_config.attn_config.use_rope = true;
    model->layer_config.attn_config.rope_theta = config->rope_theta;
    model->layer_config.attn_config.rope_cos_cache = &model->rope_cos_cache;
    model->layer_config.attn_config.rope_sin_cache = &model->rope_sin_cache;
    model->layer_config.hidden_dim = hidden_dim;
    model->layer_config.layer_norm_eps = 1e-6f; // not used by RMSNorm
    model->layer_config.norm_type = BLT_NORM_RMSNORM;
    model->layer_config.activation_type = BLT_ACTIVATION_SWIGLU;
 
    // layer weights
    model->layer_storage = (blt_transformer_layer_storage*)blt_arena_alloc(
        arena, config->num_layers * sizeof(blt_transformer_layer_storage), sizeof(void*));
    model->layer_weights = (blt_transformer_weights*)blt_arena_alloc(
        arena, config->num_layers * sizeof(blt_transformer_weights), sizeof(void*));
    BLT_REQUIRE(model->layer_storage != NULL && model->layer_weights != NULL,
        "blt_entropy_lm_create: failed to allocate layer arrays");
 
    for (size_t l = 0; l < config->num_layers; ++l) {
        blt_transformer_layer_storage* s = &model->layer_storage[l];
 
        size_t norm_shape[1] = { embed_dim };
        s->norm1_weight = blt_tensor_create(arena, norm_shape, 1, BLT_DTYPE_FP32);
        s->norm2_weight = blt_tensor_create(arena, norm_shape, 1, BLT_DTYPE_FP32);
 
        size_t qkv_shape[2] = { embed_dim, 3 * embed_dim };
        s->attn_qkv_w = blt_tensor_create(arena, qkv_shape, 2, BLT_DTYPE_FP32);
 
        size_t proj_shape[2] = { embed_dim, embed_dim };
        s->attn_proj_w = blt_tensor_create(arena, proj_shape, 2, BLT_DTYPE_FP32);
 
        size_t ffn_up_shape[2] = { embed_dim, hidden_dim };
        s->ffn_up_w   = blt_tensor_create(arena, ffn_up_shape, 2, BLT_DTYPE_FP32);
        s->ffn_gate_w = blt_tensor_create(arena, ffn_up_shape, 2, BLT_DTYPE_FP32);
 
        size_t ffn_down_shape[2] = { hidden_dim, embed_dim };
        s->ffn_down_w = blt_tensor_create(arena, ffn_down_shape, 2, BLT_DTYPE_FP32);
 
        blt_transformer_weights* w = &model->layer_weights[l];
        w->norm1_weight = &s->norm1_weight;
        w->norm1_bias   = NULL;     // no bias for RMSNorm
        w->attn_qkv_w   = &s->attn_qkv_w;
        w->attn_proj_w  = &s->attn_proj_w;
        w->norm2_weight = &s->norm2_weight;
        w->norm2_bias   = NULL;
        w->ffn_up_w     = &s->ffn_up_w;
        w->ffn_gate_w   = &s->ffn_gate_w;
        w->ffn_down_w   = &s->ffn_down_w;
    }
 
    return model;
}
 


blt_entropy_lm_grad* blt_entropy_lm_grad_create(blt_arena* arena, const blt_entropy_lm* model) {
    BLT_REQUIRE(arena != NULL && model != NULL,
        "blt_entropy_lm_grad_create: arena and model cannot be NULL");
 
    blt_entropy_lm_grad* grad = (blt_entropy_lm_grad*)blt_arena_alloc(arena, sizeof(blt_entropy_lm_grad), sizeof(void*));
    BLT_REQUIRE(grad != NULL, "blt_entropy_lm_grad_create: failed to allocate grad struct");
 
    size_t embed_dim  = model->config.embed_dim;
    size_t hidden_dim = model->config.hidden_dim;
    size_t num_layers = model->config.num_layers;
 
    size_t emb_shape[2] = { 256, embed_dim };
    grad->embedding_grad = blt_tensor_create(arena, emb_shape, 2, BLT_DTYPE_FP32);
 
    size_t head_shape[2] = { embed_dim, 256 };
    grad->lm_head_grad = blt_tensor_create(arena, head_shape, 2, BLT_DTYPE_FP32);
 
    grad->layer_grads = (blt_transformer_layer_grad*)blt_arena_alloc(
        arena, num_layers * sizeof(blt_transformer_layer_grad), sizeof(void*));
    BLT_REQUIRE(grad->layer_grads != NULL, "blt_entropy_lm_grad_create: failed to allocate layer grads");
 
    for (size_t l = 0; l < num_layers; ++l) {
        blt_transformer_layer_grad* lg = &grad->layer_grads[l];
 
        size_t norm_shape[1] = { embed_dim };
        lg->norm1_weight = blt_tensor_create(arena, norm_shape, 1, BLT_DTYPE_FP32);
        lg->norm2_weight = blt_tensor_create(arena, norm_shape, 1, BLT_DTYPE_FP32);
 
        size_t qkv_shape[2] = { embed_dim, 3 * embed_dim };
        lg->attn_qkv_w = blt_tensor_create(arena, qkv_shape, 2, BLT_DTYPE_FP32);
 
        size_t proj_shape[2] = { embed_dim, embed_dim };
        lg->attn_proj_w = blt_tensor_create(arena, proj_shape, 2, BLT_DTYPE_FP32);
 
        size_t ffn_up_shape[2] = { embed_dim, hidden_dim };
        lg->ffn_up_w   = blt_tensor_create(arena, ffn_up_shape, 2, BLT_DTYPE_FP32);
        lg->ffn_gate_w = blt_tensor_create(arena, ffn_up_shape, 2, BLT_DTYPE_FP32);
 
        size_t ffn_down_shape[2] = { hidden_dim, embed_dim };
        lg->ffn_down_w = blt_tensor_create(arena, ffn_down_shape, 2, BLT_DTYPE_FP32);
    }
 
    return grad;
}
 
 

//----------------------------------------------------------------------
// LM forward path

// embedding -> N transformer layers -> matmul against lm_head_weight ->
// logits -> cross-entropy against shifted targets
// Shifted target handling: targets[t] = bytes[t+1], so the loss operates over seq_len-1 positions

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
    blt_transformer_config layer_cfg = make_seq_layer_config(model, seq_len, &rope_cos_view, &rope_sin_view);
 

    // ----------------
    // Embedding
    blt_byte_embedding emb = { .weight = model->embedding_weight, .embed_dim = embed_dim };
    size_t x_shape[2] = { seq_len, embed_dim };
    blt_tensor x = blt_tensor_create(arena, x_shape, 2, BLT_DTYPE_FP32);
    blt_byte_embedding_forward(&emb, bytes_in, &x);
    

    // ----------------
    // N transformer layers
    for (size_t l = 0; l < model->config.num_layers; ++l) {
        blt_tensor next = blt_tensor_create(arena, x_shape, 2, BLT_DTYPE_FP32);
        blt_transformer_forward(&x, &model->layer_weights[l], &next, &layer_cfg, arena);
        x = next;
    }
 

    // ----------------
    // LM head
    // logits = x @ lm_head_weight
    size_t logits_shape[2] = { seq_len, 256 };
    blt_tensor logits_full = blt_tensor_create(arena, logits_shape, 2, BLT_DTYPE_FP32);
    blt_matmul(&x, &model->lm_head_weight, &logits_full);
 
    BLT_REQUIRE(logits_out->ndim == 2 && logits_out->shape[0] == seq_len && logits_out->shape[1] == 256,
        "blt_entropy_lm_forward: logits_out must be [seq_len, 256] FP32");
    memcpy(logits_out->data, logits_full.data, blt_tensor_bytes(&logits_full));
 

    // ----------------
    // Shifted target cross_entropy loss
    // logits[0, ..., seq_len-2] predict bytes[1, ..., seq_len-1]
    blt_tensor shifted_logits;
    blt_tensor_view_2d(&shifted_logits, logits_full.data, seq_len - 1, 256, x.backend);
 
    blt_tensor shifted_targets;
    size_t elem_size = blt_dtype_sizeof(bytes_in->dtype);
    view_1d(&shifted_targets, (char*)bytes_in->data + elem_size, seq_len - 1,
            bytes_in->dtype, bytes_in->backend);
 
    blt_cross_entropy_forward(&shifted_logits, &shifted_targets, loss_out);
}
 


//----------------------------------------------------------------------
// LM backward path

// Mirrors forward call order in reverse and reusing backward for every op 
// blt_transformer_forward doesnt save intermediates, so every layers forward gets recomputed

typedef struct {
    blt_tensor x_in;           // layer input
    blt_tensor norm1_out;
    blt_tensor attn_out;
    blt_tensor attn_residual;
    blt_tensor norm2_out;
    blt_tensor ffn_up;
    blt_tensor ffn_gate;
    blt_tensor ffn_activated;
} layer_fwd_cache;
 

// Recomputes one transformer blocks forward pass and caching
static void layer_forward_with_cache(
    const blt_tensor* x, const blt_transformer_weights* w, const blt_transformer_config* cfg,
    blt_arena* arena, size_t seq_len, size_t embed_dim, size_t hidden_dim,
    layer_fwd_cache* cache, blt_tensor* layer_out
) {
    size_t embed_shape[2] = { seq_len, embed_dim };
    size_t hidden_shape[2] = { seq_len, hidden_dim };
 
    cache->x_in = *x;
    

    // ----------------
    // pre-norm + attention with residual
    cache->norm1_out = blt_tensor_create(arena, embed_shape, 2, BLT_DTYPE_FP32);
    blt_rmsnorm_forward(x, w->norm1_weight, &cache->norm1_out);
 
    cache->attn_out = blt_tensor_create(arena, embed_shape, 2, BLT_DTYPE_FP32);
    blt_multihead_attention(&cache->norm1_out, w->attn_qkv_w, w->attn_proj_w,
                             &cache->attn_out, &cfg->attn_config, arena);
 
    cache->attn_residual = blt_tensor_create(arena, embed_shape, 2, BLT_DTYPE_FP32);
    blt_add(x, &cache->attn_out, &cache->attn_residual);
        

    // ----------------
    // pre-norm + FFN with residual
    cache->norm2_out = blt_tensor_create(arena, embed_shape, 2, BLT_DTYPE_FP32);
    blt_rmsnorm_forward(&cache->attn_residual, w->norm2_weight, &cache->norm2_out);
 
    cache->ffn_up = blt_tensor_create(arena, hidden_shape, 2, BLT_DTYPE_FP32);
    blt_matmul(&cache->norm2_out, w->ffn_up_w, &cache->ffn_up);
 
    cache->ffn_gate = blt_tensor_create(arena, hidden_shape, 2, BLT_DTYPE_FP32);
    blt_matmul(&cache->norm2_out, w->ffn_gate_w, &cache->ffn_gate);
 
    cache->ffn_activated = blt_tensor_create(arena, hidden_shape, 2, BLT_DTYPE_FP32);
    blt_swiglu_forward(&cache->ffn_gate, &cache->ffn_up, &cache->ffn_activated);
 
    blt_tensor ffn_out = blt_tensor_create(arena, embed_shape, 2, BLT_DTYPE_FP32);
    blt_matmul(&cache->ffn_activated, w->ffn_down_w, &ffn_out);
 
    *layer_out = blt_tensor_create(arena, embed_shape, 2, BLT_DTYPE_FP32);
    blt_add(&cache->attn_residual, &ffn_out, layer_out);
}
 


// Backward through one transformer block
// Writes grad with respect to input into grad_x
// Writes weight grads into lg
static void layer_backward(
    const layer_fwd_cache* cache, const blt_transformer_weights* w, const blt_transformer_config* cfg,
    blt_arena* arena, size_t seq_len, size_t embed_dim, size_t hidden_dim,
    const blt_tensor* grad_out, blt_transformer_layer_grad* lg, blt_tensor* grad_x
) {
    size_t embed_shape[2] = { seq_len, embed_dim };
    size_t hidden_shape[2] = { seq_len, hidden_dim };
    

    // ----------------
    // layer_out = attn_residual + ffn_out -> backward through matmul
    blt_tensor grad_ffn_activated = blt_tensor_create(arena, hidden_shape, 2, BLT_DTYPE_FP32);
    blt_matmul_backward(&cache->ffn_activated, w->ffn_down_w, grad_out,
                         &grad_ffn_activated, &lg->ffn_down_w);
 
    blt_tensor grad_gate = blt_tensor_create(arena, hidden_shape, 2, BLT_DTYPE_FP32);
    blt_tensor grad_up   = blt_tensor_create(arena, hidden_shape, 2, BLT_DTYPE_FP32);
    blt_swiglu_backward(&grad_ffn_activated, &cache->ffn_gate, &cache->ffn_up, &grad_gate, &grad_up);
 
    blt_tensor grad_norm2_a = blt_tensor_create(arena, embed_shape, 2, BLT_DTYPE_FP32);
    blt_matmul_backward(&cache->norm2_out, w->ffn_up_w, &grad_up, &grad_norm2_a, &lg->ffn_up_w);
 
    blt_tensor grad_norm2_b = blt_tensor_create(arena, embed_shape, 2, BLT_DTYPE_FP32);
    blt_matmul_backward(&cache->norm2_out, w->ffn_gate_w, &grad_gate, &grad_norm2_b, &lg->ffn_gate_w);
 
    blt_tensor grad_norm2_out = blt_tensor_create(arena, embed_shape, 2, BLT_DTYPE_FP32);
    blt_add(&grad_norm2_a, &grad_norm2_b, &grad_norm2_out);
 
    blt_tensor grad_attn_residual_from_norm2 = blt_tensor_create(arena, embed_shape, 2, BLT_DTYPE_FP32);
    blt_rmsnorm_backward(&grad_norm2_out, &cache->attn_residual, w->norm2_weight,
                          &grad_attn_residual_from_norm2, &lg->norm2_weight);
 

    // ----------------
    // attn_residual = x_in + attn_out
    // received grad_out directly and grad_attn_residual_from_norm2 -> sum both
    blt_tensor grad_attn_residual = blt_tensor_create(arena, embed_shape, 2, BLT_DTYPE_FP32);
    blt_add(grad_out, &grad_attn_residual_from_norm2, &grad_attn_residual);
    

    // ----------------
    // both x_in and attn_out receive grad_attn_residual directly 
    blt_tensor grad_norm1_out = blt_tensor_create(arena, embed_shape, 2, BLT_DTYPE_FP32);
    blt_multihead_attention_backward(&cache->norm1_out, w->attn_qkv_w, w->attn_proj_w,
                                      &grad_attn_residual, &grad_norm1_out,
                                      &lg->attn_qkv_w, &lg->attn_proj_w, &cfg->attn_config, arena);
 
    blt_tensor grad_x_from_norm1 = blt_tensor_create(arena, embed_shape, 2, BLT_DTYPE_FP32);
    blt_rmsnorm_backward(&grad_norm1_out, &cache->x_in, w->norm1_weight,
                          &grad_x_from_norm1, &lg->norm1_weight);
 
    *grad_x = blt_tensor_create(arena, embed_shape, 2, BLT_DTYPE_FP32);
    blt_add(&grad_attn_residual, &grad_x_from_norm1, grad_x);
}
 


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
 
    size_t embed_dim  = model->config.embed_dim;
    size_t hidden_dim = model->config.hidden_dim;
    size_t num_layers = model->config.num_layers;

    blt_tensor rope_cos_view, rope_sin_view;
    blt_transformer_config layer_cfg = make_seq_layer_config(model, seq_len, &rope_cos_view, &rope_sin_view);
 

    // ----------------
    // Recompute forward with caching
    blt_byte_embedding emb = { .weight = model->embedding_weight, .embed_dim = embed_dim };
    size_t x_shape[2] = { seq_len, embed_dim };
    blt_tensor x0 = blt_tensor_create(arena, x_shape, 2, BLT_DTYPE_FP32);
    blt_byte_embedding_forward(&emb, bytes_in, &x0);
 
    layer_fwd_cache* caches = (layer_fwd_cache*)blt_arena_alloc(
        arena, num_layers * sizeof(layer_fwd_cache), sizeof(void*));
    BLT_REQUIRE(caches != NULL, "blt_entropy_lm_backward: failed to allocate layer caches");
 
    blt_tensor x = x0;
    for (size_t l = 0; l < num_layers; ++l) {
        blt_tensor next;
        layer_forward_with_cache(&x, &model->layer_weights[l], &layer_cfg, arena,
                                  seq_len, embed_dim, hidden_dim, &caches[l], &next);
        x = next;
    }
    blt_tensor final_x = x;   // input to the LM head
 
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
    // Backward: loss -> LM head -> layers (reverse) -> embedding
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
 
    blt_tensor grad_x = grad_final_x;
    for (size_t i = 0; i < num_layers; ++i) {
        size_t l = num_layers - 1 - i;   // go through layers in reverse
        blt_tensor grad_prev;
        layer_backward(&caches[l], &model->layer_weights[l], &layer_cfg, arena,
                        seq_len, embed_dim, hidden_dim, &grad_x, &grad_out->layer_grads[l], &grad_prev);
        grad_x = grad_prev;
    }
 
    // grad_x is now dL/d(embedding output)
    // scatter add into the embedding tables gradient
    blt_byte_embedding_backward(&emb, bytes_in, &grad_x, &grad_out->embedding_grad);
}

