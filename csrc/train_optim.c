#include "train_optim.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "ops/vecmath.h"
#include "core/cuda_shim.h"

struct sgd_ctx {
    float lr;
};

static void sgd_pair_fn(float *w, float *g, const blt_param_info *info, void *ctx) {
    float lr = ((struct sgd_ctx *)ctx)->lr;
    if (info->backend == BLT_BACKEND_CUDA) {
#ifdef BLT_WITH_CUDA
        blt_cuda_sgd_step(w, (const float *)g, lr, info->numel);
#endif
    } else {
        for (size_t i = 0; i < info->numel; i++) w[i] -= lr * g[i];
    }
}

void sgd_all(blt_model *m, blt_model_grad *g, float lr) {
    struct sgd_ctx ctx = {lr};
    blt_model_visit_param_pairs(m, g, sgd_pair_fn, &ctx);
}

// AdamW state: one exp_avg + one exp_avg_sq per parameter tensor.
// Allocated once before training; zero-initialized by arena.

static adamw_pair mk_pair(blt_arena *arena, const blt_tensor *ref) {
    adamw_pair p;
    p.em = blt_tensor_create(arena, ref->shape, ref->ndim, BLT_DTYPE_FP32);
    p.esq = blt_tensor_create(arena, ref->shape, ref->ndim, BLT_DTYPE_FP32);
    return p;
}

adamw_state *adamw_state_create(blt_arena *arena, blt_model *m) {
    adamw_state *s = (adamw_state *)malloc(sizeof(adamw_state));
    memset(s, 0, sizeof(*s));
    s->step = 0;
    int idx = 0;

    s->flat[idx++] = mk_pair(arena, &m->encoder->byte_embedding_weight);
    for (size_t i = 0; i < m->encoder->ngram_weights.num_tables; i++)
        s->flat[idx++] = mk_pair(arena, &m->encoder->ngram_weights.tables[i]);
    for (size_t i = 0; i < m->encoder->config.num_layers; i++) {
        blt_local_layer_storage *w = &m->encoder->layers[i];
        s->flat[idx++] = mk_pair(arena, &w->norm1_weight);
        s->flat[idx++] = mk_pair(arena, &w->attn_qkv_w);
        s->flat[idx++] = mk_pair(arena, &w->attn_proj_w);
        s->flat[idx++] = mk_pair(arena, &w->norm2_weight);
        s->flat[idx++] = mk_pair(arena, &w->ffn_up_w);
        s->flat[idx++] = mk_pair(arena, &w->ffn_gate_w);
        s->flat[idx++] = mk_pair(arena, &w->ffn_down_w);
        s->flat[idx++] = mk_pair(arena, &w->cross_norm_weight);
        s->flat[idx++] = mk_pair(arena, &w->cross_weight_q);
        s->flat[idx++] = mk_pair(arena, &w->cross_weight_k);
        s->flat[idx++] = mk_pair(arena, &w->cross_weight_v);
        s->flat[idx++] = mk_pair(arena, &w->cross_weight_proj);
    }
    for (size_t i = 0; i < m->global->stack.num_layers; i++) {
        blt_transformer_layer_storage *w = &m->global->stack.layer_storage[i];
        s->flat[idx++] = mk_pair(arena, &w->norm1_weight);
        s->flat[idx++] = mk_pair(arena, &w->attn_qkv_w);
        s->flat[idx++] = mk_pair(arena, &w->attn_proj_w);
        s->flat[idx++] = mk_pair(arena, &w->norm2_weight);
        s->flat[idx++] = mk_pair(arena, &w->ffn_up_w);
        s->flat[idx++] = mk_pair(arena, &w->ffn_gate_w);
        s->flat[idx++] = mk_pair(arena, &w->ffn_down_w);
    }
    for (size_t i = 0; i < m->decoder->config.num_layers; i++) {
        blt_local_layer_storage *w = &m->decoder->layers[i];
        s->flat[idx++] = mk_pair(arena, &w->norm1_weight);
        s->flat[idx++] = mk_pair(arena, &w->attn_qkv_w);
        s->flat[idx++] = mk_pair(arena, &w->attn_proj_w);
        s->flat[idx++] = mk_pair(arena, &w->norm2_weight);
        s->flat[idx++] = mk_pair(arena, &w->ffn_up_w);
        s->flat[idx++] = mk_pair(arena, &w->ffn_gate_w);
        s->flat[idx++] = mk_pair(arena, &w->ffn_down_w);
        s->flat[idx++] = mk_pair(arena, &w->cross_norm_weight);
        s->flat[idx++] = mk_pair(arena, &w->cross_weight_q);
        s->flat[idx++] = mk_pair(arena, &w->cross_weight_k);
        s->flat[idx++] = mk_pair(arena, &w->cross_weight_v);
        s->flat[idx++] = mk_pair(arena, &w->cross_weight_proj);
    }
    s->flat[idx++] = mk_pair(arena, &m->decoder->lm_head_weight);
    s->flat[idx++] = mk_pair(arena, &m->decoder->d0_embed_weight);
    s->n = (size_t)idx;
    return s;
}

struct adamw_pair_ctx {
    adamw_state *s;
    blt_adamw_config cfg;
};

static void adamw_pair_fn(float *w, float *g, const blt_param_info *info, void *ctx) {
    struct adamw_pair_ctx *c = (struct adamw_pair_ctx *)ctx;
    adamw_pair *p = &c->s->flat[info->idx];
    const float lr = c->cfg.lr;
    const float b1 = c->cfg.beta1;
    const float b2 = c->cfg.beta2;
    const float eps = c->cfg.eps;
    const float wd = c->cfg.weight_decay;
    const float bc1 = 1.0f - powf(b1, (float)c->cfg.step);
    const float bc2 = 1.0f - powf(b2, (float)c->cfg.step);
    if (info->backend == BLT_BACKEND_CUDA) {
#ifdef BLT_WITH_CUDA
        blt_cuda_adamw_step(w, (const float *)g, (float *)p->em.data, (float *)p->esq.data, lr, b1, b2, eps, wd, bc1,
                            bc2, info->numel);
#endif
    } else {
        float *m = (float *)p->em.data;
        float *v = (float *)p->esq.data;
        for (size_t i = 0; i < info->numel; i++) {
            m[i] = b1 * m[i] + (1.0f - b1) * g[i];
            v[i] = b2 * v[i] + (1.0f - b2) * g[i] * g[i];
            const float mhat = m[i] / bc1;
            const float vhat = v[i] / bc2;
            w[i] -= lr * (mhat / (sqrtf(vhat) + eps) + wd * w[i]);
        }
    }
}

void adamw_all(blt_model *m, blt_model_grad *g, adamw_state *s, const blt_adamw_config *cfg) {
    s->step++;
    struct adamw_pair_ctx ctx = {.s = s, .cfg = *cfg};
    ctx.cfg.step = s->step;
    blt_model_visit_param_pairs(m, g, adamw_pair_fn, &ctx);
}

struct adamw_unorm_ctx {
    adamw_state *s;
    float b1, b2, ep, wd;
    size_t step;
    float sq;
};

static void adamw_unorm_fn(float *w, float *g, const blt_param_info *info, void *ctx) {
    struct adamw_unorm_ctx *c = (struct adamw_unorm_ctx *)ctx;
    adamw_pair *p = &c->s->flat[info->idx];
    const float bc1 = 1.0f - powf(c->b1, (float)c->step);
    const float bc2 = 1.0f - powf(c->b2, (float)c->step);
    if (info->backend == BLT_BACKEND_CUDA) {
#ifdef BLT_WITH_CUDA
        c->sq += blt_cuda_adamw_sqnorm((const float *)w, (const float *)g, (const float *)p->em.data,
                                       (const float *)p->esq.data, c->b1, c->b2, c->ep, c->wd, bc1, bc2, info->numel);
#endif
    } else {
        float *m = (float *)p->em.data;
        float *v = (float *)p->esq.data;
        for (size_t i = 0; i < info->numel; i++) {
            float m_new = c->b1 * m[i] + (1.0f - c->b1) * g[i];
            float v_new = c->b2 * v[i] + (1.0f - c->b2) * g[i] * g[i];
            float m_hat = m_new / bc1;
            float v_hat = v_new / bc2;
            float u = m_hat / (sqrtf(v_hat) + c->ep) + c->wd * w[i];
            c->sq += u * u;
        }
    }
}

float adamw_update_norm(blt_model *m, blt_model_grad *g, adamw_state *s, const blt_adamw_config *cfg) {
    struct adamw_unorm_ctx ctx = {
        .s = s,
        .b1 = cfg->beta1,
        .b2 = cfg->beta2,
        .ep = cfg->eps,
        .wd = cfg->weight_decay,
        .step = s->step + 1,
        .sq = 0.0f,
    };
    blt_model_visit_param_pairs(m, g, adamw_unorm_fn, &ctx);
    return sqrtf(ctx.sq) * cfg->lr;
}
