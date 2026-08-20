#include "blt/models/local_encoder.h"

#include "blt/core/backend.h"
#include "blt/core/tensor.h"
#include "blt/core/allocator.h"
#include "blt/models/attention.h"
#include "blt/models/transformer.h"
#include "blt/models/transformer_stack.h"
#include "blt/models/byte_embedding.h"
#include "blt/models/cross_attention.h"
#include "blt/ops/elementwise.h"
#include "blt/ops/matmul.h"
#include "blt/ops/rmsnorm.h"
#include "blt/ops/swiglu.h"
#include "blt/ops/rope.h"
#include "blt/ops/mask_builder.h"
#include "blt/ops/patch_pool.h"

#include <string.h>



//----------------------------------------------------------------------
// Config helpers
//

static bool cross_attn_fires(const blt_local_encoder_config* config, size_t layer) {
    // Table 7
    return config->cross_attn_all_layers || (layer + 1 == config->num_layers);
}



static blt_byte_embedding byte_embedding_view(const blt_local_encoder* model) {
    blt_byte_embedding emb;
    emb.vocab_size = 256;
    emb.weight = model->byte_embedding_weight;
    emb.embed_dim = model->config.embed_dim;
    return emb;
}



static blt_transformer_weights byte_layer_weights_view(const blt_local_encoder_layer_storage* s) {
    blt_transformer_weights w = {0};
    w.norm1_weight = &s->norm1_weight;      // RMSNorm has no bias so no bias weight
    w.attn_qkv_w = &s->attn_qkv_w;
    w.attn_proj_w = &s->attn_proj_w;
    w.norm2_weight = &s->norm2_weight;
    w.ffn_up_w = &s->ffn_up_w;
    w.ffn_gate_w = &s->ffn_gate_w;
    w.ffn_down_w = &s->ffn_down_w;
    return w;
}



static blt_cross_attention_weights cross_attn_weights_view(const blt_local_encoder_layer_storage* s) {
    blt_cross_attention_weights w = {0};
    w.weight_q = &s->cross_weight_q;
    w.weight_k = &s->cross_weight_k;
    w.weight_v = &s->cross_weight_v;
    w.weight_proj = &s->cross_weight_proj;
    return w;
}



static void validate_call(const blt_local_encoder* model, const blt_tensor* bytes_in,
                          const blt_patch_info* patches, size_t num_patches,
                          const blt_arena* arena, size_t* out_seq_len) {
    BLT_REQUIRE(model != NULL && bytes_in != NULL && arena != NULL,
        "blt_local_encoder: model, bytes_in and arena cannot be NULL");
    BLT_REQUIRE(patches != NULL && num_patches >= 1,
        "blt_local_encoder: need at least one patch");
    BLT_REQUIRE(bytes_in->ndim == 1 && bytes_in->dtype == BLT_DTYPE_UINT8,
        "blt_local_encoder: bytes_in must be 1D UINT8 [seq_len]");
    size_t seq_len = bytes_in->shape[0];
    BLT_REQUIRE(seq_len >= 1 && seq_len <= model->config.max_seq_len,
        "blt_local_encoder: seq_len must be in [1, max_seq_len]");
    *out_seq_len = seq_len;
}



//----------------------------------------------------------------------
// Setup: group ids, mask configs, RoPE views
//
// Forward and backward both need these things, so build them once here

typedef struct {
    blt_mask_config local_mask;   // byte self-attention: block-causal, window w_E
    blt_mask_config cross_mask;   // patch byte cross-attention: block-diagonal
    blt_tensor rope_cos_view;
    blt_tensor rope_sin_view;
} le_context;



static le_context make_context(const blt_local_encoder* model, size_t seq_len,
                               const blt_patch_info* patches, size_t num_patches,
                               const size_t* doc_boundaries, size_t num_docs,
                               blt_arena* arena) {
    size_t* query_group_ids = (size_t*)blt_arena_alloc(arena, num_patches * sizeof(size_t), 64);
    size_t* kv_group_ids    = (size_t*)blt_arena_alloc(arena, seq_len * sizeof(size_t), 64);
    BLT_REQUIRE(query_group_ids && kv_group_ids, "blt_local_encoder: failed to allocate group ids");
    blt_patch_build_group_ids(patches, num_patches, seq_len, query_group_ids, kv_group_ids);

    le_context ctx = {0};

    // Local byte mask
    // sliding window w_E (0 == full causal), causal
    ctx.local_mask.seq_len_q = seq_len;
    ctx.local_mask.seq_len_kv = seq_len;
    ctx.local_mask.sliding_window = model->config.local_window;
    ctx.local_mask.doc_boundaries = doc_boundaries;
    ctx.local_mask.num_docs = num_docs;
    ctx.local_mask.is_causal = true;

    // Cross mask
    ctx.cross_mask.seq_len_q = num_patches;
    ctx.cross_mask.seq_len_kv = seq_len;
    ctx.cross_mask.query_group_ids = query_group_ids;
    ctx.cross_mask.kv_group_ids = kv_group_ids;
    ctx.cross_mask.bidirectional_within_group = true;

    size_t head_dim = model->config.embed_dim / model->config.num_heads;
    size_t half = head_dim / 2;
    blt_tensor_view_2d(&ctx.rope_cos_view, model->rope_cos_cache.data, seq_len, half, model->rope_cos_cache.backend);
    blt_tensor_view_2d(&ctx.rope_sin_view, model->rope_sin_cache.data, seq_len, half, model->rope_sin_cache.backend);

    return ctx;
}



static blt_transformer_config make_byte_layer_config(const blt_local_encoder_config* config, const le_context* ctx) {
    blt_transformer_config t = {0};
    t.attn_config.embed_dim = config->embed_dim;
    t.attn_config.num_heads = config->num_heads;   // head_dim inferred (0)
    t.attn_config.is_causal = true;
    t.attn_config.use_rope = true;
    t.attn_config.rope_theta = config->rope_theta;
    t.attn_config.rope_cos_cache = &ctx->rope_cos_view;
    t.attn_config.rope_sin_cache = &ctx->rope_sin_view;
    t.attn_config.mask_config = &ctx->local_mask;
    t.hidden_dim = config->hidden_dim;
    t.norm_type = BLT_NORM_RMSNORM;
    t.activation_type = BLT_ACTIVATION_SWIGLU;
    return t;
}



static blt_cross_attention_config make_cross_attn_config(const blt_local_encoder_config* config, const le_context* ctx) {
    blt_cross_attention_config c = {0};
    c.embed_dim = config->embed_dim;
    c.num_heads = config->cross_attn_heads;   // head_dim inferred (0)
    c.mask_config = &ctx->cross_mask;
    return c;
}



//----------------------------------------------------------------------
// Create / grad-create

blt_local_encoder* blt_local_encoder_create(blt_arena* arena, const blt_local_encoder_config* config) {
    BLT_REQUIRE(arena != NULL && config != NULL,
        "blt_local_encoder_create: arena and config cannot be NULL");

    size_t E = config->embed_dim;
    size_t hidden = config->hidden_dim;
    BLT_REQUIRE(E > 0 && hidden > 0 && config->num_layers >= 1 && config->max_seq_len > 0,
        "blt_local_encoder_create: embed_dim/hidden_dim/num_layers/max_seq_len must be positive");
    BLT_REQUIRE(config->num_heads != 0 && E % config->num_heads == 0,
        "blt_local_encoder_create: embed_dim must be divisible by num_heads");
    size_t head_dim = E / config->num_heads;
    BLT_REQUIRE(head_dim % 2 == 0, "blt_local_encoder_create: RoPE requires an even head_dim");
    BLT_REQUIRE(config->cross_attn_heads != 0 && E % config->cross_attn_heads == 0,
        "blt_local_encoder_create: embed_dim must be divisible by cross_attn_heads");
    BLT_REQUIRE(config->ngram_config.embed_dim == E,
        "blt_local_encoder_create: ngram_config.embed_dim must equal embed_dim");
    BLT_REQUIRE(config->rope_theta > 0.0f, "blt_local_encoder_create: rope_theta must be positive");

    blt_local_encoder* m = (blt_local_encoder*)blt_arena_alloc(arena, sizeof(blt_local_encoder), sizeof(void*));
    BLT_REQUIRE(m != NULL, "blt_local_encoder_create: failed to allocate model");
    m->config = *config;

    // Byte embedding table [256, embed_dim]
    size_t emb_shape[2] = { 256, E };
    m->byte_embedding_weight = blt_tensor_create(arena, emb_shape, 2, BLT_DTYPE_FP32);

    // Hash n-gram tables
    m->ngram_weights = blt_hash_ngram_create(arena, &config->ngram_config);

    // RoPE cache - precomputed once
    size_t half = head_dim / 2;
    size_t rope_shape[2] = { config->max_seq_len, half };
    m->rope_cos_cache = blt_tensor_create(arena, rope_shape, 2, BLT_DTYPE_FP32);
    m->rope_sin_cache = blt_tensor_create(arena, rope_shape, 2, BLT_DTYPE_FP32);
    blt_rope_config rope_cfg = { .theta = config->rope_theta, .head_dim = head_dim };
    blt_rope_precompute(config->max_seq_len, &rope_cfg, &m->rope_cos_cache, &m->rope_sin_cache);

    // Per-layer weights 
    // cross_ fields are allocated on every layer even if not fired
    m->layers = (blt_local_encoder_layer_storage*)blt_arena_alloc(
        arena, config->num_layers * sizeof(blt_local_encoder_layer_storage), sizeof(void*));
    BLT_REQUIRE(m->layers != NULL, "blt_local_encoder_create: failed to allocate layers");

    size_t norm_shape[1]  = { E };
    size_t qkv_shape[2]   = { E, 3 * E };
    size_t proj_shape[2]  = { E, E };
    size_t ffn_up_shape[2]   = { E, hidden };
    size_t ffn_down_shape[2] = { hidden, E };

    for (size_t l = 0; l < config->num_layers; ++l) {
        blt_local_encoder_layer_storage* s = &m->layers[l];
        s->norm1_weight = blt_tensor_create(arena, norm_shape, 1, BLT_DTYPE_FP32);
        s->attn_qkv_w   = blt_tensor_create(arena, qkv_shape, 2, BLT_DTYPE_FP32);
        s->attn_proj_w  = blt_tensor_create(arena, proj_shape, 2, BLT_DTYPE_FP32);
        s->norm2_weight = blt_tensor_create(arena, norm_shape, 1, BLT_DTYPE_FP32);
        s->ffn_up_w     = blt_tensor_create(arena, ffn_up_shape, 2, BLT_DTYPE_FP32);
        s->ffn_gate_w   = blt_tensor_create(arena, ffn_up_shape, 2, BLT_DTYPE_FP32);
        s->ffn_down_w   = blt_tensor_create(arena, ffn_down_shape, 2, BLT_DTYPE_FP32);

        s->cross_norm_weight = blt_tensor_create(arena, norm_shape, 1, BLT_DTYPE_FP32);
        s->cross_weight_q    = blt_tensor_create(arena, proj_shape, 2, BLT_DTYPE_FP32);
        s->cross_weight_k    = blt_tensor_create(arena, proj_shape, 2, BLT_DTYPE_FP32);
        s->cross_weight_v    = blt_tensor_create(arena, proj_shape, 2, BLT_DTYPE_FP32);
        s->cross_weight_proj = blt_tensor_create(arena, proj_shape, 2, BLT_DTYPE_FP32);
    }
    return m;
}



blt_local_encoder_grad* blt_local_encoder_grad_create(blt_arena* arena, const blt_local_encoder* model) {
    BLT_REQUIRE(arena != NULL && model != NULL,
        "blt_local_encoder_grad_create: arena and model cannot be NULL");

    size_t E = model->config.embed_dim;
    size_t hidden = model->config.hidden_dim;
    size_t num_layers = model->config.num_layers;

    blt_local_encoder_grad* g = (blt_local_encoder_grad*)blt_arena_alloc(arena, sizeof(blt_local_encoder_grad), sizeof(void*));
    BLT_REQUIRE(g != NULL, "blt_local_encoder_grad_create: failed to allocate grad");

    size_t emb_shape[2] = { 256, E };
    g->embedding_grad = blt_tensor_create(arena, emb_shape, 2, BLT_DTYPE_FP32);

    g->ngram_grads.num_tables = model->ngram_weights.num_tables;
    for (size_t t = 0; t < model->ngram_weights.num_tables; ++t) {
        size_t vocab = model->ngram_weights.tables[t].shape[0];
        size_t table_shape[2] = { vocab, E };
        g->ngram_grads.tables[t] = blt_tensor_create(arena, table_shape, 2, BLT_DTYPE_FP32);
    }

    g->layer_grads = (blt_local_encoder_layer_grad*)blt_arena_alloc(
        arena, num_layers * sizeof(blt_local_encoder_layer_grad), sizeof(void*));
    BLT_REQUIRE(g->layer_grads != NULL, "blt_local_encoder_grad_create: failed to allocate layer grads");

    size_t norm_shape[1]  = { E };
    size_t qkv_shape[2]   = { E, 3 * E };
    size_t proj_shape[2]  = { E, E };
    size_t ffn_up_shape[2]   = { E, hidden };
    size_t ffn_down_shape[2] = { hidden, E };

    for (size_t l = 0; l < num_layers; ++l) {
        blt_local_encoder_layer_grad* lg = &g->layer_grads[l];
        lg->norm1_weight = blt_tensor_create(arena, norm_shape, 1, BLT_DTYPE_FP32);
        lg->attn_qkv_w   = blt_tensor_create(arena, qkv_shape, 2, BLT_DTYPE_FP32);
        lg->attn_proj_w  = blt_tensor_create(arena, proj_shape, 2, BLT_DTYPE_FP32);
        lg->norm2_weight = blt_tensor_create(arena, norm_shape, 1, BLT_DTYPE_FP32);
        lg->ffn_up_w     = blt_tensor_create(arena, ffn_up_shape, 2, BLT_DTYPE_FP32);
        lg->ffn_gate_w   = blt_tensor_create(arena, ffn_up_shape, 2, BLT_DTYPE_FP32);
        lg->ffn_down_w   = blt_tensor_create(arena, ffn_down_shape, 2, BLT_DTYPE_FP32);

        lg->cross_norm_weight = blt_tensor_create(arena, norm_shape, 1, BLT_DTYPE_FP32);
        lg->cross_weight_q    = blt_tensor_create(arena, proj_shape, 2, BLT_DTYPE_FP32);
        lg->cross_weight_k    = blt_tensor_create(arena, proj_shape, 2, BLT_DTYPE_FP32);
        lg->cross_weight_v    = blt_tensor_create(arena, proj_shape, 2, BLT_DTYPE_FP32);
        lg->cross_weight_proj = blt_tensor_create(arena, proj_shape, 2, BLT_DTYPE_FP32);
    }
    return g;
}



//----------------------------------------------------------------------
// Forward path
//
// Pipeline steps (BLT §3.2, Fig. 2 step 1-2, Fig. 5):
// 1. e_0     = byte_embedding(bytes) + hash_ngram_embeddings(bytes)
// 2. P_0     = pool(e_0, patches) 
// 3. for l in 1..num_layers:
//      h_l = byte_transformer_layer_l(h_{l-1})     (local block-causal, window w_E)
//      if cross_attn_fires(l): P_l = P_{l-1} + cross_attn_l(RMSNorm(P_{l-1}), h_l)
//      else:                   P_l = P_{l-1}
// Output: P_final -> global patch transformer
//         h_final -> kept for the decoder (FBLT paper)

void blt_local_encoder_forward(const blt_local_encoder* model,
                               const blt_tensor* bytes_in,
                               const blt_patch_info* patches,
                               size_t num_patches,
                               const size_t* doc_boundaries,
                               size_t num_docs,
                               blt_tensor* patch_out,
                               blt_tensor* byte_hidden_out,
                               blt_arena* arena) {
    size_t seq_len;
    validate_call(model, bytes_in, patches, num_patches, arena, &seq_len);
    BLT_REQUIRE(patch_out != NULL && byte_hidden_out != NULL,
        "blt_local_encoder_forward: patch_out and byte_hidden_out cannot be NULL");

    const blt_local_encoder_config* config = &model->config;
    size_t E = config->embed_dim;
    blt_check_nd_fp32(patch_out, 2, (const size_t[]){num_patches, E},
        "blt_local_encoder_forward: patch_out must be [num_patches, embed_dim] FP32");
    blt_check_nd_fp32(byte_hidden_out, 2, (const size_t[]){seq_len, E},
        "blt_local_encoder_forward: byte_hidden_out must be [seq_len, embed_dim] FP32");

    le_context ctx = make_context(model, seq_len, patches, num_patches, doc_boundaries, num_docs, arena);
    blt_transformer_config layer_config = make_byte_layer_config(config, &ctx);
    blt_cross_attention_config cross_config = make_cross_attn_config(config, &ctx);

    size_t byte_shape[2] = { seq_len, E };
    size_t patch_shape[2] = { num_patches, E };

    // -----------------------------------------------------------------
    // STEP 1: e_0 = byte_embedding(bytes) + hash_ngram_embeddings(bytes)
    //
    blt_byte_embedding emb = byte_embedding_view(model);
    blt_tensor byte_emb = blt_tensor_create(arena, byte_shape, 2, BLT_DTYPE_FP32);
    blt_byte_embedding_forward(&emb, bytes_in, &byte_emb);

    blt_tensor h = blt_tensor_create(arena, byte_shape, 2, BLT_DTYPE_FP32);
    blt_hash_ngram_forward(&model->ngram_weights, &config->ngram_config, bytes_in, &byte_emb, &h);

    // -----------------------------------------------------------------
    // STEP 2: P_0 = pool(e_0, patches)
    //
    blt_tensor p = blt_tensor_create(arena, patch_shape, 2, BLT_DTYPE_FP32);
    blt_patch_pool_forward(&h, patches, num_patches, config->pool_type, &p);

    // -----------------------------------------------------------------
    // STEP 3: layer loop
    //
    for (size_t l = 0; l < config->num_layers; ++l) {
        const blt_local_encoder_layer_storage* s = &model->layers[l];

        blt_transformer_weights w = byte_layer_weights_view(s);
        blt_tensor h_next = blt_tensor_create(arena, byte_shape, 2, BLT_DTYPE_FP32);
        blt_transformer_forward(&h, &w, &h_next, &layer_config, arena);
        h = h_next;   // byte states after transformer

        if (cross_attn_fires(config, l)) {
            blt_tensor normed_p = blt_tensor_create(arena, patch_shape, 2, BLT_DTYPE_FP32);
            blt_rmsnorm_forward(&p, &s->cross_norm_weight, &normed_p);

            blt_cross_attention_weights xw = cross_attn_weights_view(s);
            blt_tensor cross_out = blt_tensor_create(arena, patch_shape, 2, BLT_DTYPE_FP32);
            blt_cross_attention_forward(&normed_p, &h, &xw, &cross_out, &cross_config, arena);

            blt_tensor p_next = blt_tensor_create(arena, patch_shape, 2, BLT_DTYPE_FP32);
            blt_add(&p, &cross_out, &p_next);
            p = p_next;
        }
    }

    // -----------------------------------------------------------------
    // STEP 4: results into output
    //
    memcpy(patch_out->data, p.data, num_patches * E * sizeof(float));
    memcpy(byte_hidden_out->data, h.data, seq_len * E * sizeof(float));

    // Arena cleanup is callers responsibility
}



//----------------------------------------------------------------------
// Backward path
//
// Mirrors forwards call order in reverse
// forward doesnt save intermediates -> so each layers forward is recomputed here with caching

typedef struct {
    bool has_cross;
    blt_tensor p_in;         // P_l before this layer's cross-attn   [num_patches, E]
    blt_tensor normed_p;     // rmsnorm(p_in, cross_norm_weight)     [num_patches, E]
    blt_transformer_layer_cache* byte_cache;  // recomputed byte transformer block, from blt_transformer_layer_forward_cached
} layer_cache;



// Builds the blt_transformer_layer_grad view blt_transformer_layer_backward
// needs from this layers blt_local_encoder_layer_grad
static blt_transformer_layer_grad byte_layer_grad_view(blt_local_encoder_layer_grad* lg) {
    blt_transformer_layer_grad g;
    g.norm1_weight = lg->norm1_weight;
    g.attn_qkv_w   = lg->attn_qkv_w;
    g.attn_proj_w  = lg->attn_proj_w;
    g.norm2_weight = lg->norm2_weight;
    g.ffn_up_w     = lg->ffn_up_w;
    g.ffn_gate_w   = lg->ffn_gate_w;
    g.ffn_down_w   = lg->ffn_down_w;
    return g;
}


void blt_local_encoder_backward(const blt_local_encoder* model,
                                const blt_tensor* bytes_in,
                                const blt_patch_info* patches,
                                size_t num_patches,
                                const size_t* doc_boundaries,
                                size_t num_docs,
                                const blt_tensor* grad_patch_out,
                                const blt_tensor* grad_byte_hidden_out,
                                blt_local_encoder_grad* grad,
                                blt_arena* arena) {
    size_t seq_len;
    validate_call(model, bytes_in, patches, num_patches, arena, &seq_len);
    BLT_REQUIRE(grad_patch_out != NULL && grad != NULL && grad->layer_grads != NULL,
        "blt_local_encoder_backward: grad_patch_out and grad cannot be NULL");

    const blt_local_encoder_config* config = &model->config;
    const size_t E = config->embed_dim;
    const size_t L = config->num_layers;

    blt_check_nd_fp32(grad_patch_out, 2, (const size_t[]){num_patches, E},
        "blt_local_encoder_backward: grad_patch_out must be [num_patches, embed_dim] FP32");
    if (grad_byte_hidden_out != NULL) {
        blt_check_nd_fp32(grad_byte_hidden_out, 2, (const size_t[]){seq_len, E},
            "blt_local_encoder_backward: grad_byte_hidden_out must be [seq_len, embed_dim] FP32");
    }
    blt_check_nd_fp32(&grad->embedding_grad, 2, (const size_t[]){256, E},
        "blt_local_encoder_backward: embedding_grad must be [256, embed_dim] FP32");

    le_context ctx = make_context(model, seq_len, patches, num_patches, doc_boundaries, num_docs, arena);
    blt_transformer_config layer_config = make_byte_layer_config(config, &ctx);
    blt_cross_attention_config cross_config = make_cross_attn_config(config, &ctx);

    size_t byte_shape[2] = { seq_len, E };
    size_t patch_shape[2] = { num_patches, E };

    // -----------------------------------------------------------------
    // STEP 1: recompute forward, caching layer intermediates
    //   h[0..L] - byte states entering each layer (h[0] = e_0)
    //   caches[] - per-layer FFN/norm intermediates + cross-attn inputs
    // 
    blt_byte_embedding emb = byte_embedding_view(model);
    blt_tensor byte_emb = blt_tensor_create(arena, byte_shape, 2, BLT_DTYPE_FP32);
    blt_byte_embedding_forward(&emb, bytes_in, &byte_emb);

    blt_tensor* h = (blt_tensor*)blt_arena_alloc(arena, (L + 1) * sizeof(blt_tensor), sizeof(void*));
    layer_cache* caches = (layer_cache*)blt_arena_alloc(arena, L * sizeof(layer_cache), sizeof(void*));
    BLT_REQUIRE(h && caches, "blt_local_encoder_backward: failed to allocate recompute caches");
    memset(caches, 0, L * sizeof(layer_cache));

    h[0] = blt_tensor_create(arena, byte_shape, 2, BLT_DTYPE_FP32);
    blt_hash_ngram_forward(&model->ngram_weights, &config->ngram_config, bytes_in, &byte_emb, &h[0]);

    blt_tensor p = blt_tensor_create(arena, patch_shape, 2, BLT_DTYPE_FP32);
    blt_patch_pool_forward(&h[0], patches, num_patches, config->pool_type, &p);

        for (size_t l = 0; l < L; ++l) {
        const blt_local_encoder_layer_storage* s = &model->layers[l];
        layer_cache* c = &caches[l];

        blt_transformer_weights w = byte_layer_weights_view(s);
        c->byte_cache = blt_transformer_layer_forward_cached(
            &h[l], &w, &layer_config, arena, seq_len, E, config->hidden_dim, &h[l + 1]);

        if (cross_attn_fires(config, l)) {
            c->has_cross = true;
            c->p_in = p;   // P_l before cross-attn

            c->normed_p = blt_tensor_create(arena, patch_shape, 2, BLT_DTYPE_FP32);
            blt_rmsnorm_forward(&c->p_in, &s->cross_norm_weight, &c->normed_p);

            blt_cross_attention_weights xw = cross_attn_weights_view(s);
            blt_tensor cross_out = blt_tensor_create(arena, patch_shape, 2, BLT_DTYPE_FP32);
            blt_cross_attention_forward(&c->normed_p, &h[l + 1], &xw, &cross_out, &cross_config, arena);

            blt_tensor p_next = blt_tensor_create(arena, patch_shape, 2, BLT_DTYPE_FP32);
            blt_add(&c->p_in, &cross_out, &p_next);
            p = p_next;
        }
    }

    // -----------------------------------------------------------------
    // STEP 2: terminal gradients
    //   dh — running dL/d(h_{l+1}), starts at dL/d(h_final)
    //   dP — running dL/d(P_{l+1}), starts at dL/d(P_final)
    //
    blt_tensor dh = blt_tensor_create(arena, byte_shape, 2, BLT_DTYPE_FP32);
    if (grad_byte_hidden_out != NULL) {
        memcpy(dh.data, grad_byte_hidden_out->data, seq_len * E * sizeof(float));
    } // else stays zero

    blt_tensor dP = blt_tensor_create(arena, patch_shape, 2, BLT_DTYPE_FP32);
    memcpy(dP.data, grad_patch_out->data, num_patches * E * sizeof(float));

    // -----------------------------------------------------------------
    // STEP 3: reverse layer loop
    // 
        for (size_t li = L; li-- > 0;) {
        const blt_local_encoder_layer_storage* s = &model->layers[li];
        blt_local_encoder_layer_grad* lg = &grad->layer_grads[li];
        const layer_cache* c = &caches[li];

        if (c->has_cross) {
            // p_next = p_in + cross_out  =>  residual: d(p_in) += dP, d(cross_out) = dP
            blt_cross_attention_weights xw = cross_attn_weights_view(s);
            blt_cross_attention_grad xg = {0};
            xg.grad_weight_q = lg->cross_weight_q;
            xg.grad_weight_k = lg->cross_weight_k;
            xg.grad_weight_v = lg->cross_weight_v;
            xg.grad_weight_proj = lg->cross_weight_proj;

            blt_tensor grad_normed_p = blt_tensor_create(arena, patch_shape, 2, BLT_DTYPE_FP32);
            blt_tensor grad_kv = blt_tensor_create(arena, byte_shape, 2, BLT_DTYPE_FP32);
            blt_cross_attention_backward(&c->normed_p, &h[li + 1], &xw, &dP,
                                         &grad_normed_p, &grad_kv, &xg, &cross_config, arena);

            // cross-attn got h[li+1] after transformer call
            // so its kv gradient adds into dh before the layer backward
            blt_tensor dh_new = blt_tensor_create(arena, byte_shape, 2, BLT_DTYPE_FP32);
            blt_add(&dh, &grad_kv, &dh_new);
            dh = dh_new;

            // normed_p = rmsnorm(p_in, cross_norm_weight)
            blt_tensor grad_p_norm = blt_tensor_create(arena, patch_shape, 2, BLT_DTYPE_FP32);
            blt_rmsnorm_backward(&grad_normed_p, &c->p_in, &s->cross_norm_weight, &grad_p_norm, &lg->cross_norm_weight);

            // d(p_in) = dP (residual) + grad through the norm
            blt_tensor dP_new = blt_tensor_create(arena, patch_shape, 2, BLT_DTYPE_FP32);
            blt_add(&dP, &grad_p_norm, &dP_new);
            dP = dP_new;
        }

        // byte transformer layer backward: dh becomes dL/d(h[li])
        blt_transformer_weights w = byte_layer_weights_view(s);
        blt_transformer_layer_grad lgv = byte_layer_grad_view(lg);
        blt_tensor dh_next;
        blt_transformer_layer_backward(c->byte_cache, &w, &layer_config, arena,
                                        seq_len, E, config->hidden_dim, &dh, &lgv, &dh_next);
        dh = dh_next;
    }

    // -----------------------------------------------------------------
    // STEP 4: P_0 = pool(h_0) scatter pool gradient into h_0 grad
    //
    blt_tensor grad_h0_pool = blt_tensor_create(arena, byte_shape, 2, BLT_DTYPE_FP32);
    blt_patch_pool_backward(&dP, &h[0], patches, num_patches, config->pool_type, &grad_h0_pool);

    blt_tensor grad_e0 = blt_tensor_create(arena, byte_shape, 2, BLT_DTYPE_FP32);
    blt_add(&dh, &grad_h0_pool, &grad_e0);

    // -----------------------------------------------------------------
    // STEP 5: e_0 = byte_embedding + hash n-grams
    //
    blt_tensor grad_byte_emb = blt_tensor_create(arena, byte_shape, 2, BLT_DTYPE_FP32);
    blt_hash_ngram_backward(&config->ngram_config, bytes_in, &grad_e0, &grad_byte_emb, grad->ngram_grads.tables);

    // scatter-adds into grad->embedding_grad (zeroed by grad_create)
    blt_byte_embedding_backward(&emb, bytes_in, &grad_byte_emb, &grad->embedding_grad);

    // Arena cleanup is callers responsibility
}