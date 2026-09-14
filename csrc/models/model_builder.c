#include "model_builder.h"
#include "models/checkpoint.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

void blt_model_config_defaults(blt_model_config *cfg,
                               size_t embed, size_t hidden,
                               size_t enc_layers, size_t glob_layers,
                               size_t dec_layers, size_t max_seq_len,
                               int cross_last) {
    memset(cfg, 0, sizeof(*cfg));

    // Encoder
    cfg->encoder_config.embed_dim = embed;
    cfg->encoder_config.patch_dim = 0;
    cfg->encoder_config.num_layers = enc_layers;
    cfg->encoder_config.hidden_dim = hidden;
    cfg->encoder_config.num_heads = 4;
    cfg->encoder_config.cross_attn_heads = 4;
    cfg->encoder_config.local_window = 0;
    cfg->encoder_config.cross_attn_all_layers = !cross_last;
    cfg->encoder_config.pool_type = BLT_POOL_MEAN;
    cfg->encoder_config.rope_theta = 500000.0f;
    cfg->encoder_config.max_seq_len = max_seq_len;
    cfg->encoder_config.ngram_config.ngram_sizes[0] = 3;
    cfg->encoder_config.ngram_config.ngram_sizes[1] = 4;
    cfg->encoder_config.ngram_config.num_ngram_sizes = 2;
    cfg->encoder_config.ngram_config.per_ngram_vocab = 50;
    cfg->encoder_config.ngram_config.hash_prime = 1000000007ULL;
    cfg->encoder_config.ngram_config.normalize = true;
    cfg->encoder_config.ngram_config.embed_dim = embed;

    // Global transformer
    cfg->global_config.embed_dim = embed;
    cfg->global_config.num_layers = glob_layers;
    cfg->global_config.hidden_dim = hidden;
    cfg->global_config.num_heads = 4;
    cfg->global_config.rope_theta = 500000.0f;
    cfg->global_config.max_seq_len = max_seq_len;

    // Decoder
    cfg->decoder_config.embed_dim = embed;
    cfg->decoder_config.patch_dim = 0;
    cfg->decoder_config.num_layers = dec_layers;
    cfg->decoder_config.hidden_dim = hidden;
    cfg->decoder_config.num_heads = 4;
    cfg->decoder_config.cross_attn_heads = 4;
    cfg->decoder_config.local_window = 0;
    cfg->decoder_config.cross_attn_all_layers = !cross_last;
    cfg->decoder_config.rope_theta = 500000.0f;
    cfg->decoder_config.max_seq_len = max_seq_len;
    cfg->decoder_config.vocab_size = 256;
}

void blt_fill_small_uniform(blt_tensor *t, float scale) {
    float *data = (float *)t->data;
    for (size_t i = 0; i < t->numel; i++) {
        float r = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
        data[i] = r * scale;
    }
}

blt_entropy_lm *blt_make_entropy_lm(blt_arena *arena, size_t ms,
                                    const char *load_path, uint64_t seed) {
    blt_entropy_lm_config ecfg;
    memset(&ecfg, 0, sizeof(ecfg));
    ecfg.embed_dim = 32;
    ecfg.num_layers = 1;
    ecfg.num_heads = 2;
    ecfg.hidden_dim = 64;
    ecfg.max_seq_len = ms;
    ecfg.rope_theta = 10000.0f;

    blt_entropy_lm *lm = blt_entropy_lm_create(arena, &ecfg);

    srand((unsigned)seed);
    blt_fill_small_uniform(&lm->embedding_weight, 0.1f);
    for (size_t i = 0; i < lm->stack.num_layers; i++) {
        blt_transformer_layer_storage *l = &lm->stack.layer_storage[i];
        blt_fill_small_uniform(&l->attn_qkv_w, 0.1f);
        blt_fill_small_uniform(&l->attn_proj_w, 0.1f);
        blt_fill_small_uniform(&l->ffn_up_w, 0.1f);
        blt_fill_small_uniform(&l->ffn_gate_w, 0.1f);
        blt_fill_small_uniform(&l->ffn_down_w, 0.1f);
    }
    blt_fill_small_uniform(&lm->lm_head_weight, 0.1f);

    if (load_path) {
        blt_entropy_lm_load(lm, load_path);
    }
    return lm;
}

void blt_make_patcher_cfg(blt_patcher_config *pcfg,
                          int fixed,
                          float threshold_global,
                          float threshold_monotonic,
                          size_t max_patch_length) {
    memset(pcfg, 0, sizeof(*pcfg));
    if (fixed) {
        pcfg->threshold_global = 1e9f;
        pcfg->threshold_monotonic = 1e9f;
        pcfg->max_patch_length = 4;
    } else {
        pcfg->threshold_global = threshold_global;
        pcfg->threshold_monotonic = threshold_monotonic;
        pcfg->max_patch_length = max_patch_length;
    }
    pcfg->rule = BLT_PATCH_RULE_GLOBAL;
    pcfg->reset_on_newline = false;
}
