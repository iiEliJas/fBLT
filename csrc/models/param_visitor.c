#include "models/param_visitor.h"

#include "models/local_common.h"

static blt_tensor *self_tensor(blt_local_layer_storage *l, size_t j) {
    blt_tensor *all[] = {
        &l->norm1_weight, &l->attn_qkv_w, &l->attn_proj_w, &l->norm2_weight,
        &l->ffn_up_w,     &l->ffn_gate_w, &l->ffn_down_w,
    };
    return all[j];
}

static blt_tensor *cross_tensor(blt_local_layer_storage *l, size_t j) {
    blt_tensor *all[] = {
        &l->cross_norm_weight, &l->cross_weight_q, &l->cross_weight_k, &l->cross_weight_v, &l->cross_weight_proj,
    };
    return all[j];
}

static blt_tensor *global_self_tensor(blt_transformer_layer_storage *l, size_t j) {
    return self_tensor((blt_local_layer_storage *)l, j);
}

static blt_tensor *self_grad(blt_local_layer_grad *g, size_t j) {
    blt_tensor *all[] = {
        &g->norm1_weight, &g->attn_qkv_w, &g->attn_proj_w, &g->norm2_weight,
        &g->ffn_up_w,     &g->ffn_gate_w, &g->ffn_down_w,
    };
    return all[j];
}

static blt_tensor *cross_grad(blt_local_layer_grad *g, size_t j) {
    blt_tensor *all[] = {
        &g->cross_norm_weight, &g->cross_weight_q, &g->cross_weight_k, &g->cross_weight_v, &g->cross_weight_proj,
    };
    return all[j];
}

static blt_tensor *global_self_grad(blt_transformer_layer_grad *g, size_t j) {
    blt_tensor *all[] = {
        &g->norm1_weight, &g->attn_qkv_w, &g->attn_proj_w, &g->norm2_weight,
        &g->ffn_up_w,     &g->ffn_gate_w, &g->ffn_down_w,
    };
    return all[j];
}

static int is_self_norm(size_t j) { return j == 0 || j == 3; }

static int is_cross_norm(size_t j) { return j == 0; }

static void fill_info(blt_param_info *info, float *ptr, int idx, int is_grad, int norm, int group, int layer, int pin,
                      size_t numel, int backend) {
    info->ptr = ptr;
    info->idx = idx;
    info->is_grad = is_grad;
    info->is_norm = norm;
    info->is_enc = (group == BLT_GROUP_EMBED || group == BLT_GROUP_ENC_SELF || group == BLT_GROUP_ENC_CROSS);
    info->is_glob = (group == BLT_GROUP_GLOB);
    info->is_dec = (group == BLT_GROUP_DEC_SELF || group == BLT_GROUP_DEC_CROSS || group == BLT_GROUP_HEAD);
    info->layer_idx = layer;
    info->param_in_layer = pin;
    info->component_group = group;
    info->numel = numel;
    info->backend = backend;
}

static void emit_weight(blt_param_visitor_fn fn, void *ctx, int *idx, float *ptr, int group, int norm, int layer,
                        int pin, size_t numel, int backend) {
    blt_param_info info;
    fill_info(&info, ptr, *idx, 0, norm, group, layer, pin, numel, backend);
    fn(ptr, &info, ctx);
    (*idx)++;
}

static void emit_grad(blt_param_visitor_fn fn, void *ctx, int *idx, float *ptr, int group, int norm, int layer, int pin,
                      size_t numel, int backend) {
    blt_param_info info;
    fill_info(&info, ptr, *idx, 1, norm, group, layer, pin, numel, backend);
    fn(ptr, &info, ctx);
    (*idx)++;
}

static void emit_pair(blt_param_pair_visitor_fn fn, void *ctx, int *idx, float *wptr, float *gptr, int group, int norm,
                      int layer, int pin, size_t numel, int backend) {
    blt_param_info info;
    fill_info(&info, wptr, *idx, 0, norm, group, layer, pin, numel, backend);
    fn(wptr, gptr, &info, ctx);
    (*idx)++;
}

void blt_model_visit_params(const blt_model *model, blt_model_grad *grad, blt_param_visitor_fn fn, void *ctx,
                            int visit_grads) {
    const size_t L = model->config.encoder_config.num_layers;
    const size_t G = model->config.global_config.num_layers;
    const size_t D = model->config.decoder_config.num_layers;
    const size_t NT = model->encoder->ngram_weights.num_tables;
    int idx = 0;

    if (!visit_grads) {
        // -- weight path --

        // enc.byte_embedding_weight
        emit_weight(fn, ctx, &idx, (float *)model->encoder->byte_embedding_weight.data, BLT_GROUP_EMBED, 0, -1, -1,
                    model->encoder->byte_embedding_weight.numel, model->encoder->byte_embedding_weight.backend);

        // enc.ngram_weights.tables[k]
        for (size_t k = 0; k < NT; k++) {
            emit_weight(fn, ctx, &idx, (float *)model->encoder->ngram_weights.tables[k].data, BLT_GROUP_EMBED, 0, -1,
                        -1, model->encoder->ngram_weights.tables[k].numel,
                        model->encoder->ngram_weights.tables[k].backend);
        }

        // enc layers: 7 self + 5 cross per layer
        for (size_t i = 0; i < L; i++) {
            blt_local_layer_storage *w = &model->encoder->layers[i];
            for (size_t j = 0; j < 7; j++) {
                blt_tensor *t = self_tensor(w, j);
                emit_weight(fn, ctx, &idx, (float *)t->data, BLT_GROUP_ENC_SELF, is_self_norm(j), (int)i, (int)j,
                            t->numel, t->backend);
            }
            for (size_t j = 0; j < 5; j++) {
                blt_tensor *t = cross_tensor(w, j);
                emit_weight(fn, ctx, &idx, (float *)t->data, BLT_GROUP_ENC_CROSS, is_cross_norm(j), (int)i, (int)j,
                            t->numel, t->backend);
            }
        }

        // glob layers: 7 self per layer
        for (size_t i = 0; i < G; i++) {
            blt_transformer_layer_storage *w = &model->global->stack.layer_storage[i];
            for (size_t j = 0; j < 7; j++) {
                blt_tensor *t = global_self_tensor(w, j);
                emit_weight(fn, ctx, &idx, (float *)t->data, BLT_GROUP_GLOB, is_self_norm(j), (int)i, (int)j, t->numel,
                            t->backend);
            }
        }

        // dec layers: 7 self + 5 cross per layer
        for (size_t i = 0; i < D; i++) {
            blt_local_layer_storage *w = &model->decoder->layers[i];
            for (size_t j = 0; j < 7; j++) {
                blt_tensor *t = self_tensor(w, j);
                emit_weight(fn, ctx, &idx, (float *)t->data, BLT_GROUP_DEC_SELF, is_self_norm(j), (int)i, (int)j,
                            t->numel, t->backend);
            }
            for (size_t j = 0; j < 5; j++) {
                blt_tensor *t = cross_tensor(w, j);
                emit_weight(fn, ctx, &idx, (float *)t->data, BLT_GROUP_DEC_CROSS, is_cross_norm(j), (int)i, (int)j,
                            t->numel, t->backend);
            }
        }

        // dec.lm_head_weight
        emit_weight(fn, ctx, &idx, (float *)model->decoder->lm_head_weight.data, BLT_GROUP_HEAD, 0, -1, -1,
                    model->decoder->lm_head_weight.numel, model->decoder->lm_head_weight.backend);

        // dec.d0_embed_weight
        emit_weight(fn, ctx, &idx, (float *)model->decoder->d0_embed_weight.data, BLT_GROUP_HEAD, 0, -1, -1,
                    model->decoder->d0_embed_weight.numel, model->decoder->d0_embed_weight.backend);
    } else {
        // -- gradient path --
        blt_local_encoder_grad *eg = grad->encoder_grad;
        blt_global_transformer_grad *gg = grad->global_grad;
        blt_local_decoder_grad *dg = grad->decoder_grad;

        // encoder embedding grad
        emit_grad(fn, ctx, &idx, (float *)eg->embedding_grad.data, BLT_GROUP_EMBED, 0, -1, -1, eg->embedding_grad.numel,
                  eg->embedding_grad.backend);

        // encoder ngram grads
        for (size_t k = 0; k < NT; k++) {
            emit_grad(fn, ctx, &idx, (float *)eg->ngram_grads.tables[k].data, BLT_GROUP_EMBED, 0, -1, -1,
                      eg->ngram_grads.tables[k].numel, eg->ngram_grads.tables[k].backend);
        }

        // encoder layer grads: 7 self + 5 cross
        for (size_t i = 0; i < L; i++) {
            blt_local_layer_grad *lg = &eg->layer_grads[i];
            for (size_t j = 0; j < 7; j++) {
                blt_tensor *t = self_grad(lg, j);
                emit_grad(fn, ctx, &idx, (float *)t->data, BLT_GROUP_ENC_SELF, is_self_norm(j), (int)i, (int)j,
                          t->numel, t->backend);
            }
            for (size_t j = 0; j < 5; j++) {
                blt_tensor *t = cross_grad(lg, j);
                emit_grad(fn, ctx, &idx, (float *)t->data, BLT_GROUP_ENC_CROSS, is_cross_norm(j), (int)i, (int)j,
                          t->numel, t->backend);
            }
        }

        // global layer grads: 7 self
        for (size_t i = 0; i < G; i++) {
            blt_transformer_layer_grad *tlg = &gg->stack_grad->layer_grads[i];
            for (size_t j = 0; j < 7; j++) {
                blt_tensor *t = global_self_grad(tlg, j);
                emit_grad(fn, ctx, &idx, (float *)t->data, BLT_GROUP_GLOB, is_self_norm(j), (int)i, (int)j, t->numel,
                          t->backend);
            }
        }

        // decoder layer grads: 7 self + 5 cross
        for (size_t i = 0; i < D; i++) {
            blt_local_layer_grad *lg = &dg->layer_grads[i];
            for (size_t j = 0; j < 7; j++) {
                blt_tensor *t = self_grad(lg, j);
                emit_grad(fn, ctx, &idx, (float *)t->data, BLT_GROUP_DEC_SELF, is_self_norm(j), (int)i, (int)j,
                          t->numel, t->backend);
            }
            for (size_t j = 0; j < 5; j++) {
                blt_tensor *t = cross_grad(lg, j);
                emit_grad(fn, ctx, &idx, (float *)t->data, BLT_GROUP_DEC_CROSS, is_cross_norm(j), (int)i, (int)j,
                          t->numel, t->backend);
            }
        }

        // decoder head grads
        emit_grad(fn, ctx, &idx, (float *)dg->lm_head_grad.data, BLT_GROUP_HEAD, 0, -1, -1, dg->lm_head_grad.numel,
                  dg->lm_head_grad.backend);
        emit_grad(fn, ctx, &idx, (float *)dg->d0_embed_grad.data, BLT_GROUP_HEAD, 0, -1, -1, dg->d0_embed_grad.numel,
                  dg->d0_embed_grad.backend);
    }
}

void blt_model_visit_param_pairs(const blt_model *model, blt_model_grad *grad, blt_param_pair_visitor_fn fn,
                                 void *ctx) {
    const size_t L = model->config.encoder_config.num_layers;
    const size_t G = model->config.global_config.num_layers;
    const size_t D = model->config.decoder_config.num_layers;
    const size_t NT = model->encoder->ngram_weights.num_tables;
    int idx = 0;

    blt_local_encoder_grad *eg = grad->encoder_grad;
    blt_global_transformer_grad *gg = grad->global_grad;
    blt_local_decoder_grad *dg = grad->decoder_grad;

    // enc.byte_embedding_weight
    emit_pair(fn, ctx, &idx, (float *)model->encoder->byte_embedding_weight.data, (float *)eg->embedding_grad.data,
              BLT_GROUP_EMBED, 0, -1, -1, model->encoder->byte_embedding_weight.numel,
              model->encoder->byte_embedding_weight.backend);

    // enc.ngram_weights.tables[k]
    for (size_t k = 0; k < NT; k++) {
        emit_pair(fn, ctx, &idx, (float *)model->encoder->ngram_weights.tables[k].data,
                  (float *)eg->ngram_grads.tables[k].data, BLT_GROUP_EMBED, 0, -1, -1,
                  model->encoder->ngram_weights.tables[k].numel, model->encoder->ngram_weights.tables[k].backend);
    }

    // enc layers: 7 self + 5 cross per layer
    for (size_t i = 0; i < L; i++) {
        blt_local_layer_storage *w = &model->encoder->layers[i];
        blt_local_layer_grad *lg = &eg->layer_grads[i];
        for (size_t j = 0; j < 7; j++) {
            blt_tensor *wt = self_tensor(w, j);
            blt_tensor *gt = self_grad(lg, j);
            emit_pair(fn, ctx, &idx, (float *)wt->data, (float *)gt->data, BLT_GROUP_ENC_SELF, is_self_norm(j), (int)i,
                      (int)j, wt->numel, wt->backend);
        }
        for (size_t j = 0; j < 5; j++) {
            blt_tensor *wt = cross_tensor(w, j);
            blt_tensor *gt = cross_grad(lg, j);
            emit_pair(fn, ctx, &idx, (float *)wt->data, (float *)gt->data, BLT_GROUP_ENC_CROSS, is_cross_norm(j),
                      (int)i, (int)j, wt->numel, wt->backend);
        }
    }

    // glob layers: 7 self per layer
    for (size_t i = 0; i < G; i++) {
        blt_transformer_layer_storage *w = &model->global->stack.layer_storage[i];
        blt_transformer_layer_grad *tlg = &gg->stack_grad->layer_grads[i];
        for (size_t j = 0; j < 7; j++) {
            blt_tensor *wt = global_self_tensor(w, j);
            blt_tensor *gt = global_self_grad(tlg, j);
            emit_pair(fn, ctx, &idx, (float *)wt->data, (float *)gt->data, BLT_GROUP_GLOB, is_self_norm(j), (int)i,
                      (int)j, wt->numel, wt->backend);
        }
    }

    // dec layers: 7 self + 5 cross per layer
    for (size_t i = 0; i < D; i++) {
        blt_local_layer_storage *w = &model->decoder->layers[i];
        blt_local_layer_grad *lg = &dg->layer_grads[i];
        for (size_t j = 0; j < 7; j++) {
            blt_tensor *wt = self_tensor(w, j);
            blt_tensor *gt = self_grad(lg, j);
            emit_pair(fn, ctx, &idx, (float *)wt->data, (float *)gt->data, BLT_GROUP_DEC_SELF, is_self_norm(j), (int)i,
                      (int)j, wt->numel, wt->backend);
        }
        for (size_t j = 0; j < 5; j++) {
            blt_tensor *wt = cross_tensor(w, j);
            blt_tensor *gt = cross_grad(lg, j);
            emit_pair(fn, ctx, &idx, (float *)wt->data, (float *)gt->data, BLT_GROUP_DEC_CROSS, is_cross_norm(j),
                      (int)i, (int)j, wt->numel, wt->backend);
        }
    }

    // dec.lm_head_weight
    emit_pair(fn, ctx, &idx, (float *)model->decoder->lm_head_weight.data, (float *)dg->lm_head_grad.data,
              BLT_GROUP_HEAD, 0, -1, -1, model->decoder->lm_head_weight.numel, model->decoder->lm_head_weight.backend);

    // dec.d0_embed_weight
    emit_pair(fn, ctx, &idx, (float *)model->decoder->d0_embed_weight.data, (float *)dg->d0_embed_grad.data,
              BLT_GROUP_HEAD, 0, -1, -1, model->decoder->d0_embed_weight.numel,
              model->decoder->d0_embed_weight.backend);
}
