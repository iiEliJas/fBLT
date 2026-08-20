#include "blt/models/local_decoder.h"
#include "blt/core/backend.h"
#include "blt/core/tensor.h"
#include "blt/core/allocator.h"
#include "blt/models/attention.h"
#include "blt/models/transformer.h"
#include "blt/models/transformer_stack.h"
#include "blt/models/cross_attention.h"
#include "blt/ops/elementwise.h"
#include "blt/ops/matmul.h"
#include "blt/ops/rmsnorm.h"
#include "blt/ops/swiglu.h"
#include "blt/ops/rope.h"
#include "blt/ops/mask_builder.h"
#include "blt/ops/patch_pool.h"
#include "blt/ops/cross_entropy.h"

#include <string.h>
#include <stdint.h>



//----------------------------------------------------------------------
// Config helpers
//

static bool cross_attn_fires(const blt_local_decoder_config* config, size_t layer) {
    // Table 7 finding: the decoder wants cross-attn on "All Layers" by default,
    // unlike the encoder's "Last Layer" default -- still just a config knob.
    return config->cross_attn_all_layers || (layer + 1 == config->num_layers);
}



static blt_transformer_weights byte_layer_weights_view(const blt_local_decoder_layer_storage* s) {
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



static blt_cross_attention_weights cross_attn_weights_view(const blt_local_decoder_layer_storage* s) {
    blt_cross_attention_weights w = {0};
    w.weight_q = &s->cross_weight_q;
    w.weight_k = &s->cross_weight_k;
    w.weight_v = &s->cross_weight_v;
    w.weight_proj = &s->cross_weight_proj;
    return w;
}



static void validate_call(const blt_local_decoder* model, const blt_tensor* byte_hidden_in,
                          const blt_tensor* patch_in, const blt_patch_info* patches, size_t num_patches,
                          const blt_tensor* bytes_in, const blt_arena* arena, size_t* out_seq_len) {
    BLT_REQUIRE(model != NULL && byte_hidden_in != NULL && patch_in != NULL && bytes_in != NULL && arena != NULL,
        "blt_local_decoder: model, byte_hidden_in, patch_in, bytes_in and arena cannot be NULL");
    BLT_REQUIRE(patches != NULL && num_patches >= 1,
        "blt_local_decoder: need at least one patch");

    size_t E = model->config.embed_dim;
    blt_check_nd_fp32(byte_hidden_in, 2, (const size_t[]){0, E},
        "blt_local_decoder: byte_hidden_in must be [seq_len, embed_dim] FP32");

    size_t seq_len = byte_hidden_in->shape[0];
    BLT_REQUIRE(seq_len >= 2 && seq_len <= model->config.max_seq_len,
        "blt_local_decoder: seq_len must be in [2, max_seq_len]");

    blt_check_nd_fp32(patch_in, 2, (const size_t[]){num_patches, E},
        "blt_local_decoder: patch_in must be [num_patches, embed_dim] FP32");

    BLT_REQUIRE(bytes_in->ndim == 1 && bytes_in->dtype == BLT_DTYPE_UINT8 && bytes_in->shape[0] == seq_len,
        "blt_local_decoder: bytes_in must be 1D UINT8 [seq_len] matching byte_hidden_in");

    *out_seq_len = seq_len;
}



//----------------------------------------------------------------------
// Setup: group ids, mask configs, RoPE views
//
// for both forward and backward

typedef struct {
    blt_mask_config local_mask;   // byte self-attention inside the decoder's transformer block: causal, window w_D
    blt_mask_config cross_mask;   // byte->patch cross-attention: block-diagonal, roles swapped from the encoder
    blt_tensor rope_cos_view;
    blt_tensor rope_sin_view;
} ld_context;



static ld_context make_context(const blt_local_decoder* model, size_t seq_len,
                               const blt_patch_info* patches, size_t num_patches,
                               const size_t* doc_boundaries, size_t num_docs,
                               blt_arena* arena) {
    // blt_patch_build_group_ids already produces exactly the two arrays the
    // role-swapped decoder cross-mask needs: query_group_ids_out[j] = j (trivial
    // patch identity) and kv_group_ids_out[i] = the patch id containing byte i.
    // We just wire them to the opposite mask fields from the encoder.
    size_t* patch_identity_ids = (size_t*)blt_arena_alloc(arena, num_patches * sizeof(size_t), 64);
    size_t* byte_patch_ids     = (size_t*)blt_arena_alloc(arena, seq_len * sizeof(size_t), 64);
    BLT_REQUIRE(patch_identity_ids && byte_patch_ids, "blt_local_decoder: failed to allocate group ids");
    blt_patch_build_group_ids(patches, num_patches, seq_len, patch_identity_ids, byte_patch_ids);

    ld_context ctx = {0};

    // Byte self-attention (inside the byte transformer block): causal, window w_D
    ctx.local_mask.seq_len_q = seq_len;
    ctx.local_mask.seq_len_kv = seq_len;
    ctx.local_mask.sliding_window = model->config.local_window;
    ctx.local_mask.doc_boundaries = doc_boundaries;
    ctx.local_mask.num_docs = num_docs;
    ctx.local_mask.is_causal = true;

    // Cross mask: bytes are now queries, patches are now keys/values
    ctx.cross_mask.seq_len_q = seq_len;
    ctx.cross_mask.seq_len_kv = num_patches;
    ctx.cross_mask.query_group_ids = byte_patch_ids;      // each bytes own patch id
    ctx.cross_mask.kv_group_ids = patch_identity_ids;     // patch j group id = j
    ctx.cross_mask.bidirectional_within_group = true;
    ctx.cross_mask.is_causal = false;

    size_t head_dim = model->config.embed_dim / model->config.num_heads;
    size_t half = head_dim / 2;
    blt_tensor_view_2d(&ctx.rope_cos_view, model->rope_cos_cache.data, seq_len, half, model->rope_cos_cache.backend);
    blt_tensor_view_2d(&ctx.rope_sin_view, model->rope_sin_cache.data, seq_len, half, model->rope_sin_cache.backend);

    return ctx;
}



static blt_transformer_config make_byte_layer_config(const blt_local_decoder_config* config, const ld_context* ctx) {
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



static blt_cross_attention_config make_cross_attn_config(const blt_local_decoder_config* config, const ld_context* ctx) {
    blt_cross_attention_config c = {0};
    c.embed_dim = config->embed_dim;
    c.num_heads = config->cross_attn_heads;   // head_dim inferred (0)
    c.mask_config = &ctx->cross_mask;
    return c;
}



//----------------------------------------------------------------------
// Create / grad-create

blt_local_decoder* blt_local_decoder_create(blt_arena* arena, const blt_local_decoder_config* config) {
    BLT_REQUIRE(arena != NULL && config != NULL,
        "blt_local_decoder_create: arena and config cannot be NULL");

    size_t E = config->embed_dim;
    size_t hidden = config->hidden_dim;
    BLT_REQUIRE(E > 0 && hidden > 0 && config->num_layers >= 1 && config->max_seq_len > 0,
        "blt_local_decoder_create: embed_dim/hidden_dim/num_layers/max_seq_len must be positive");
    BLT_REQUIRE(config->num_heads != 0 && E % config->num_heads == 0,
        "blt_local_decoder_create: embed_dim must be divisible by num_heads");
    size_t head_dim = E / config->num_heads;
    BLT_REQUIRE(head_dim % 2 == 0, "blt_local_decoder_create: RoPE requires an even head_dim");
    BLT_REQUIRE(config->cross_attn_heads != 0 && E % config->cross_attn_heads == 0,
        "blt_local_decoder_create: embed_dim must be divisible by cross_attn_heads");
    BLT_REQUIRE(config->rope_theta > 0.0f, "blt_local_decoder_create: rope_theta must be positive");
    BLT_REQUIRE(config->vocab_size > 0, "blt_local_decoder_create: vocab_size must be positive");

    blt_local_decoder* m = (blt_local_decoder*)blt_arena_alloc(arena, sizeof(blt_local_decoder), sizeof(void*));
    BLT_REQUIRE(m != NULL, "blt_local_decoder_create: failed to allocate model");
    m->config = *config;

    // RoPE cache - precomputed once
    size_t half = head_dim / 2;
    size_t rope_shape[2] = { config->max_seq_len, half };
    m->rope_cos_cache = blt_tensor_create(arena, rope_shape, 2, BLT_DTYPE_FP32);
    m->rope_sin_cache = blt_tensor_create(arena, rope_shape, 2, BLT_DTYPE_FP32);
    blt_rope_config rope_cfg = { .theta = config->rope_theta, .head_dim = head_dim };
    blt_rope_precompute(config->max_seq_len, &rope_cfg, &m->rope_cos_cache, &m->rope_sin_cache);

    // Per-layer weights
    m->layers = (blt_local_decoder_layer_storage*)blt_arena_alloc(
        arena, config->num_layers * sizeof(blt_local_decoder_layer_storage), sizeof(void*));
    BLT_REQUIRE(m->layers != NULL, "blt_local_decoder_create: failed to allocate layers");

    size_t norm_shape[1]  = { E };
    size_t qkv_shape[2]   = { E, 3 * E };
    size_t proj_shape[2]  = { E, E };
    size_t ffn_up_shape[2]   = { E, hidden };
    size_t ffn_down_shape[2] = { hidden, E };

    for (size_t l = 0; l < config->num_layers; ++l) {
        blt_local_decoder_layer_storage* s = &m->layers[l];
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

    // LM head
    size_t lm_head_shape[2] = { E, config->vocab_size };
    m->lm_head_weight = blt_tensor_create(arena, lm_head_shape, 2, BLT_DTYPE_FP32);

    return m;
}



blt_local_decoder_grad* blt_local_decoder_grad_create(blt_arena* arena, const blt_local_decoder* model) {
    BLT_REQUIRE(arena != NULL && model != NULL,
        "blt_local_decoder_grad_create: arena and model cannot be NULL");

    size_t E = model->config.embed_dim;
    size_t hidden = model->config.hidden_dim;
    size_t V = model->config.vocab_size;
    size_t num_layers = model->config.num_layers;

    blt_local_decoder_grad* g = (blt_local_decoder_grad*)blt_arena_alloc(arena, sizeof(blt_local_decoder_grad), sizeof(void*));
    BLT_REQUIRE(g != NULL, "blt_local_decoder_grad_create: failed to allocate grad");

    g->layer_grads = (blt_local_decoder_layer_grad*)blt_arena_alloc(
        arena, num_layers * sizeof(blt_local_decoder_layer_grad), sizeof(void*));
    BLT_REQUIRE(g->layer_grads != NULL, "blt_local_decoder_grad_create: failed to allocate layer grads");

    size_t norm_shape[1]  = { E };
    size_t qkv_shape[2]   = { E, 3 * E };
    size_t proj_shape[2]  = { E, E };
    size_t ffn_up_shape[2]   = { E, hidden };
    size_t ffn_down_shape[2] = { hidden, E };

    for (size_t l = 0; l < num_layers; ++l) {
        blt_local_decoder_layer_grad* lg = &g->layer_grads[l];
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

    size_t lm_head_shape[2] = { E, V };
    g->lm_head_grad = blt_tensor_create(arena, lm_head_shape, 2, BLT_DTYPE_FP32);

    return g;
}



//----------------------------------------------------------------------
// Forward path (BLT §3.3, Eq. 10-12; FBLT.md Checkpoint D)
//
// D_0 = byte_hidden_in (h_final from the local encoder -- not a fresh embedding)
// for l in 1..num_layers:
//   if cross_attn_fires(l): B = D_{l-1} + cross_attn_l(RMSNorm(D_{l-1}), patch_in)   [cross-attn FIRST]
//   else:                   B = D_{l-1}
//   D_l = byte_transformer_layer_l(B)     (causal self-attn + FFN, residuals inside)
// logits = D_final @ lm_head_weight
// loss   = shifted next-byte cross-entropy (targets[t] = bytes_in[t+1]), same convention as blt_entropy_lm

void blt_local_decoder_forward(const blt_local_decoder* model,
                               const blt_tensor* byte_hidden_in,
                               const blt_tensor* patch_in,
                               const blt_patch_info* patches,
                               size_t num_patches,
                               const blt_tensor* bytes_in,
                               const size_t* doc_boundaries,
                               size_t num_docs,
                               blt_tensor* logits_out,
                               blt_tensor* loss_out,
                               blt_arena* arena) {
    size_t seq_len;
    validate_call(model, byte_hidden_in, patch_in, patches, num_patches, bytes_in, arena, &seq_len);
    BLT_REQUIRE(logits_out != NULL && loss_out != NULL,
        "blt_local_decoder_forward: logits_out and loss_out cannot be NULL");

    const blt_local_decoder_config* config = &model->config;
    size_t E = config->embed_dim;
    size_t V = config->vocab_size;
    blt_check_nd_fp32(logits_out, 2, (const size_t[]){seq_len, V},
        "blt_local_decoder_forward: logits_out must be [seq_len, vocab_size] FP32");

    ld_context ctx = make_context(model, seq_len, patches, num_patches, doc_boundaries, num_docs, arena);
    blt_transformer_config layer_config = make_byte_layer_config(config, &ctx);
    blt_cross_attention_config cross_config = make_cross_attn_config(config, &ctx);

    size_t byte_shape[2] = { seq_len, E };

    // -----------------------------------------------------------------
    // STEP 1: D_0 = byte_hidden_in
    //
    blt_tensor d = *byte_hidden_in;

    // -----------------------------------------------------------------
    // STEP 2: layer loop (cross-attn first, then byte transformer block)
    //
    for (size_t l = 0; l < config->num_layers; ++l) {
        const blt_local_decoder_layer_storage* s = &model->layers[l];
        blt_tensor b;

        if (cross_attn_fires(config, l)) {
            blt_tensor normed_d = blt_tensor_create(arena, byte_shape, 2, BLT_DTYPE_FP32);
            blt_rmsnorm_forward(&d, &s->cross_norm_weight, &normed_d);

            blt_cross_attention_weights xw = cross_attn_weights_view(s);
            blt_tensor cross_out = blt_tensor_create(arena, byte_shape, 2, BLT_DTYPE_FP32);
            blt_cross_attention_forward(&normed_d, patch_in, &xw, &cross_out, &cross_config, arena);

            b = blt_tensor_create(arena, byte_shape, 2, BLT_DTYPE_FP32);
            blt_add(&d, &cross_out, &b);
        } else {
            b = d;
        }

        blt_transformer_weights w = byte_layer_weights_view(s);
        blt_tensor d_next = blt_tensor_create(arena, byte_shape, 2, BLT_DTYPE_FP32);
        blt_transformer_forward(&b, &w, &d_next, &layer_config, arena);
        d = d_next;
    }

    // -----------------------------------------------------------------
    // STEP 3: LM head + shifted next-byte cross-entropy
    //
    blt_matmul(&d, &model->lm_head_weight, logits_out);

    blt_tensor logits_view;
    blt_tensor_view_2d(&logits_view, logits_out->data, seq_len - 1, V, logits_out->backend);

    blt_tensor targets_view = {0};
    targets_view.data = (uint8_t*)bytes_in->data + 1;
    targets_view.ndim = 1;
    targets_view.shape[0] = seq_len - 1;
    targets_view.strides[0] = 1;
    targets_view.numel = seq_len - 1;
    targets_view.dtype = BLT_DTYPE_UINT8;
    targets_view.backend = bytes_in->backend;
    targets_view.is_view = true;

    blt_cross_entropy_forward(&logits_view, &targets_view, loss_out);

    // Arena cleanup is caller's responsibility
}



//----------------------------------------------------------------------
// Backward path
//

typedef struct {
    bool has_cross;
    blt_tensor d_in;         // D_l entering this layer, before cross-attn        [seq_len, E]
    blt_tensor normed_d;     // rmsnorm(d_in, cross_norm_weight)                  [seq_len, E]
    blt_transformer_layer_cache* byte_cache;  // recomputed byte transformer block, from blt_transformer_layer_forward_cached
} dec_layer_cache;



// Builds blt_transformer_layer_grad view blt_transformer_layer_backward
// needs from this layers blt_local_decoder_layer_grad
static blt_transformer_layer_grad byte_layer_grad_view(blt_local_decoder_layer_grad* lg) {
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



void blt_local_decoder_backward(const blt_local_decoder* model,
                                const blt_tensor* byte_hidden_in,
                                const blt_tensor* patch_in,
                                const blt_patch_info* patches,
                                size_t num_patches,
                                const blt_tensor* bytes_in,
                                const size_t* doc_boundaries,
                                size_t num_docs,
                                blt_tensor* grad_byte_hidden_in,
                                blt_tensor* grad_patch_in,
                                blt_local_decoder_grad* grad,
                                blt_arena* arena) {
    size_t seq_len;
    validate_call(model, byte_hidden_in, patch_in, patches, num_patches, bytes_in, arena, &seq_len);
    BLT_REQUIRE(grad_byte_hidden_in != NULL && grad_patch_in != NULL && grad != NULL && grad->layer_grads != NULL,
        "blt_local_decoder_backward: grad_byte_hidden_in, grad_patch_in and grad cannot be NULL");

    const blt_local_decoder_config* config = &model->config;
    const size_t E = config->embed_dim;
    const size_t V = config->vocab_size;
    const size_t L = config->num_layers;

    blt_check_nd_fp32(grad_byte_hidden_in, 2, (const size_t[]){seq_len, E},
        "blt_local_decoder_backward: grad_byte_hidden_in must be [seq_len, embed_dim] FP32");
    blt_check_nd_fp32(grad_patch_in, 2, (const size_t[]){num_patches, E},
        "blt_local_decoder_backward: grad_patch_in must be [num_patches, embed_dim] FP32");
    blt_check_nd_fp32(&grad->lm_head_grad, 2, (const size_t[]){E, V},
        "blt_local_decoder_backward: lm_head_grad must be [embed_dim, vocab_size] FP32");

    ld_context ctx = make_context(model, seq_len, patches, num_patches, doc_boundaries, num_docs, arena);
    blt_transformer_config layer_config = make_byte_layer_config(config, &ctx);
    blt_cross_attention_config cross_config = make_cross_attn_config(config, &ctx);

    size_t byte_shape[2] = { seq_len, E };
    size_t patch_shape[2] = { num_patches, E };
    size_t logits_shape[2] = { seq_len, V };

    // -----------------------------------------------------------------
    // STEP 1: recompute forward, caching per-layer intermediates
    //   d[0..L] - byte states entering each layer (d[0] = byte_hidden_in)
    //   caches[] - per-layer cross-attn + byte-transformer-block intermediates
    //
    blt_tensor* d = (blt_tensor*)blt_arena_alloc(arena, (L + 1) * sizeof(blt_tensor), sizeof(void*));
    dec_layer_cache* caches = (dec_layer_cache*)blt_arena_alloc(arena, L * sizeof(dec_layer_cache), sizeof(void*));
    BLT_REQUIRE(d && caches, "blt_local_decoder_backward: failed to allocate recompute caches");
    memset(caches, 0, L * sizeof(dec_layer_cache));

    d[0] = *byte_hidden_in;

        for (size_t l = 0; l < L; ++l) {
        const blt_local_decoder_layer_storage* s = &model->layers[l];
        dec_layer_cache* c = &caches[l];
        c->d_in = d[l];

        blt_tensor b;
        if (cross_attn_fires(config, l)) {
            c->has_cross = true;

            c->normed_d = blt_tensor_create(arena, byte_shape, 2, BLT_DTYPE_FP32);
            blt_rmsnorm_forward(&d[l], &s->cross_norm_weight, &c->normed_d);

            blt_cross_attention_weights xw = cross_attn_weights_view(s);
            blt_tensor cross_out = blt_tensor_create(arena, byte_shape, 2, BLT_DTYPE_FP32);
            blt_cross_attention_forward(&c->normed_d, patch_in, &xw, &cross_out, &cross_config, arena);

            b = blt_tensor_create(arena, byte_shape, 2, BLT_DTYPE_FP32);
            blt_add(&d[l], &cross_out, &b);
        } else {
            c->has_cross = false;
            b = d[l];
        }

        blt_transformer_weights w = byte_layer_weights_view(s);
        c->byte_cache = blt_transformer_layer_forward_cached(
            &b, &w, &layer_config, arena, seq_len, E, config->hidden_dim, &d[l + 1]);
    }

    // LM head, recomputed to obtain the terminal gradient
    blt_tensor logits = blt_tensor_create(arena, logits_shape, 2, BLT_DTYPE_FP32);
    blt_matmul(&d[L], &model->lm_head_weight, &logits);

    blt_tensor logits_view;
    blt_tensor_view_2d(&logits_view, logits.data, seq_len - 1, V, logits.backend);

    blt_tensor targets_view = {0};
    targets_view.data = (uint8_t*)bytes_in->data + 1;
    targets_view.ndim = 1;
    targets_view.shape[0] = seq_len - 1;
    targets_view.strides[0] = 1;
    targets_view.numel = seq_len - 1;
    targets_view.dtype = BLT_DTYPE_UINT8;
    targets_view.backend = bytes_in->backend;
    targets_view.is_view = true;

    // -----------------------------------------------------------------
    // STEP 2: terminal gradient dL/dlogits (only the first seq_len-1 rows
    // were used in the loss -- the last position's row gets zero gradient)
    //
    size_t grad_logits_view_shape[2] = { seq_len - 1, V };
    blt_tensor grad_logits_view = blt_tensor_create(arena, grad_logits_view_shape, 2, BLT_DTYPE_FP32);
    blt_cross_entropy_backward(&logits_view, &targets_view, &grad_logits_view);

    blt_tensor grad_logits = blt_tensor_create(arena, logits_shape, 2, BLT_DTYPE_FP32);   // zero-initialized
    memcpy(grad_logits.data, grad_logits_view.data, (seq_len - 1) * V * sizeof(float));

    // logits = d[L] @ lm_head_weight
    blt_tensor dh = blt_tensor_create(arena, byte_shape, 2, BLT_DTYPE_FP32);
    blt_matmul_backward(&d[L], &model->lm_head_weight, &grad_logits, &dh, &grad->lm_head_grad);

    // -----------------------------------------------------------------
    // STEP 3: reverse layer loop
    //   dh - running dL/d(d[l+1]), starts at dL/d(d[L])
    //   dP - running dL/d(patch_in), summed across every firing layer
    //
    blt_tensor dP = blt_tensor_create(arena, patch_shape, 2, BLT_DTYPE_FP32);   // zero-initialized

        for (size_t li = L; li-- > 0;) {
        const blt_local_decoder_layer_storage* s = &model->layers[li];
        blt_local_decoder_layer_grad* lg = &grad->layer_grads[li];
        const dec_layer_cache* c = &caches[li];

        // byte transformer block backward: dh becomes dL/d(b) for this layer
        blt_transformer_weights w = byte_layer_weights_view(s);
        blt_transformer_layer_grad lgv = byte_layer_grad_view(lg);
        blt_tensor grad_b;
        blt_transformer_layer_backward(c->byte_cache, &w, &layer_config, arena,
                                        seq_len, E, config->hidden_dim, &dh, &lgv, &grad_b);

        if (c->has_cross) {
            // b = d_in + cross_out  =>  residual: d(d_in) += grad_b, d(cross_out) = grad_b
            blt_cross_attention_weights xw = cross_attn_weights_view(s);
            blt_cross_attention_grad xg = {0};
            xg.grad_weight_q = lg->cross_weight_q;
            xg.grad_weight_k = lg->cross_weight_k;
            xg.grad_weight_v = lg->cross_weight_v;
            xg.grad_weight_proj = lg->cross_weight_proj;

            blt_tensor grad_normed_d = blt_tensor_create(arena, byte_shape, 2, BLT_DTYPE_FP32);
            blt_tensor grad_kv = blt_tensor_create(arena, patch_shape, 2, BLT_DTYPE_FP32);
            blt_cross_attention_backward(&c->normed_d, patch_in, &xw, &grad_b,
                                         &grad_normed_d, &grad_kv, &xg, &cross_config, arena);

            // patch_in feeds every firing layer's cross-attn, so its gradient accumulates
            blt_tensor dP_new = blt_tensor_create(arena, patch_shape, 2, BLT_DTYPE_FP32);
            blt_add(&dP, &grad_kv, &dP_new);
            dP = dP_new;

            // normed_d = rmsnorm(d_in, cross_norm_weight)
            blt_tensor grad_d_in_norm = blt_tensor_create(arena, byte_shape, 2, BLT_DTYPE_FP32);
            blt_rmsnorm_backward(&grad_normed_d, &c->d_in, &s->cross_norm_weight, &grad_d_in_norm, &lg->cross_norm_weight);

            // d(d_in) = grad_b (residual) + grad through the norm
            blt_tensor grad_d_in = blt_tensor_create(arena, byte_shape, 2, BLT_DTYPE_FP32);
            blt_add(&grad_b, &grad_d_in_norm, &grad_d_in);
            dh = grad_d_in;
        } else {
            dh = grad_b;
        }
    }

    // -----------------------------------------------------------------
    // STEP 4: write terminal gradients -- byte_hidden_in and patch_in are
    // external inputs (local encoder / global transformer outputs), so both
    // are pure sinks here rather than feeding further local computation.
    //
    memcpy(grad_byte_hidden_in->data, dh.data, seq_len * E * sizeof(float));
    memcpy(grad_patch_in->data, dP.data, num_patches * E * sizeof(float));

    // Arena cleanup is callers responsibility
}