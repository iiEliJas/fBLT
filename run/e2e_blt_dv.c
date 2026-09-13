// end-to-end validation driver.
//
// Loads the real trained BLT-D checkpoint (runs/bltd_l03_40k.fblt by
// default), continues a deep held-out prompt three ways, and checks the
// load-bearing correctness invariant:
//
//   BLT-DV output == plain greedy BLT output, byte for byte.
//
// (Every DV-committed byte is the causal argmax at its position by
// construction: accept-until-mismatch, replace on mismatch, free byte on
// full match -- see blt_verify_draft / Fast-BLT Algorithm 2.)
//
// Also prints raw BLT-D drafts (no verification) and NFE accounting so
// drafting quality / acceptance can be eyeballed across checkpoints and
// unmasking settings. Build + run:  make e2e-dv

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "core/allocator.h"
#include "core/backend.h"
#include "models/model.h"
#include "models/checkpoint.h"
#include "models/entropy_lm.h"
#include "models/patcher.h"
#include "infer/block_generation.h"
#include "core/generate_greedy.h"

static void fill_small_uniform(blt_tensor *t, float scale) {
    float *data = (float *)t->data;
    for (size_t i = 0; i < t->numel; i++) {
        float r = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
        data[i] = r * scale;
    }
}

int main(int argc, char **argv) {
    const char *ckpt_path = "runs/bltd_l03_40k.fblt";
    size_t new_bytes = 48;
    size_t block_size = 8;
    const char *prompt_arg = NULL;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--ckpt") && i + 1 < argc) ckpt_path = argv[++i];
        else if (!strcmp(argv[i], "--new-bytes") && i + 1 < argc) new_bytes = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--block-size") && i + 1 < argc) block_size = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--prompt") && i + 1 < argc) prompt_arg = argv[++i];
    }

    const size_t E = 64, HID = 128, L = 2, MS = 512;

    blt_arena *model_arena = blt_arena_create(64 * 1024 * 1024, BLT_BACKEND_CPU);
    blt_arena *scratch = blt_arena_create(256 * 1024 * 1024, BLT_BACKEND_CPU);

    // Must mirror bin/train_blt_d's config exactly (checkpoint validates).
    blt_model_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.encoder_config.embed_dim = E;
    cfg.encoder_config.patch_dim = 0;
    cfg.encoder_config.num_layers = L;
    cfg.encoder_config.hidden_dim = HID;
    cfg.encoder_config.num_heads = 4;
    cfg.encoder_config.cross_attn_heads = 4;
    cfg.encoder_config.local_window = 0;
    cfg.encoder_config.cross_attn_all_layers = true;
    cfg.encoder_config.pool_type = BLT_POOL_MEAN;
    cfg.encoder_config.rope_theta = 500000.0f;
    cfg.encoder_config.max_seq_len = MS;
    cfg.encoder_config.ngram_config.ngram_sizes[0] = 3;
    cfg.encoder_config.ngram_config.ngram_sizes[1] = 4;
    cfg.encoder_config.ngram_config.num_ngram_sizes = 2;
    cfg.encoder_config.ngram_config.per_ngram_vocab = 50;
    cfg.encoder_config.ngram_config.hash_prime = 1000000007ULL;
    cfg.encoder_config.ngram_config.normalize = true;
    cfg.encoder_config.ngram_config.embed_dim = E;
    cfg.global_config.embed_dim = E;
    cfg.global_config.num_layers = L;
    cfg.global_config.hidden_dim = HID;
    cfg.global_config.num_heads = 4;
    cfg.global_config.rope_theta = 500000.0f;
    cfg.global_config.max_seq_len = MS;
    cfg.decoder_config.embed_dim = E;
    cfg.decoder_config.patch_dim = 0;
    cfg.decoder_config.num_layers = L;
    cfg.decoder_config.hidden_dim = HID;
    cfg.decoder_config.num_heads = 4;
    cfg.decoder_config.cross_attn_heads = 4;
    cfg.decoder_config.local_window = 0;
    cfg.decoder_config.cross_attn_all_layers = true;
    cfg.decoder_config.rope_theta = 500000.0f;
    cfg.decoder_config.max_seq_len = MS;
    cfg.decoder_config.vocab_size = 256;

    blt_model *model = blt_model_create(model_arena, &cfg);
    blt_model_load(model, ckpt_path);
    printf("[E2E] checkpoint loaded: %s\n", ckpt_path);

    // Entropy LM drives patch boundaries (any reasonable config; every
    // generation path shares it, so boundary consistency holds).
    blt_entropy_lm_config ecfg;
    memset(&ecfg, 0, sizeof(ecfg));
    ecfg.embed_dim = 32;
    ecfg.num_layers = 1;
    ecfg.num_heads = 2;
    ecfg.hidden_dim = 64;
    ecfg.max_seq_len = MS;
    ecfg.rope_theta = 10000.0f;
    blt_entropy_lm *lm = blt_entropy_lm_create(model_arena, &ecfg);
    srand(11);
    fill_small_uniform(&lm->embedding_weight, 0.1f);
    for (size_t i = 0; i < lm->stack.num_layers; i++) {
        blt_transformer_layer_storage *l = &lm->stack.layer_storage[i];
        fill_small_uniform(&l->attn_qkv_w, 0.1f);
        fill_small_uniform(&l->attn_proj_w, 0.1f);
        fill_small_uniform(&l->ffn_up_w, 0.1f);
        fill_small_uniform(&l->ffn_gate_w, 0.1f);
        fill_small_uniform(&l->ffn_down_w, 0.1f);
    }
    fill_small_uniform(&lm->lm_head_weight, 0.1f);

    blt_patcher_config pcfg;
    memset(&pcfg, 0, sizeof(pcfg));
    pcfg.threshold_global = 2.5f;
    pcfg.threshold_monotonic = 1.0f;
    pcfg.max_patch_length = 16;
    pcfg.rule = BLT_PATCH_RULE_GLOBAL;
    pcfg.reset_on_newline = false;

    static const char *default_prompt = "static int parse_header(const char *buf, size_t len)\n{";
    const uint8_t *prompt = (const uint8_t *)(prompt_arg ? prompt_arg : default_prompt);
    const size_t prompt_len = strlen((const char *)prompt);

    uint8_t *ref = malloc(prompt_len + new_bytes);
    uint8_t *dv = malloc(prompt_len + new_bytes);
    uint8_t *draft_only = malloc(prompt_len + new_bytes);

    blt_generate_greedy(model, lm, &pcfg, prompt, prompt_len, new_bytes, ref, scratch);

    blt_block_gen_config gc;
    memset(&gc, 0, sizeof(gc));
    gc.block_size = block_size;
    gc.d0_mode = BLT_D0_LEARNED;
    gc.opts.strategy = BLT_UNMASK_CONFIDENCE;
    gc.opts.threshold = 0.7f;

    blt_infer_stats st;
    memset(&st, 0, sizeof(st));
    blt_generate_greedy_blockdiff_verify(model, lm, &pcfg, prompt, prompt_len, new_bytes, dv, &gc, &st, scratch);

    blt_generate_greedy_blockdiff(model, lm, &pcfg, prompt, prompt_len, new_bytes, draft_only, &gc, NULL, scratch);

    const int identical = memcmp(ref, dv, prompt_len + new_bytes) == 0;
    printf("[E2E] DV == greedy: %s\n", identical ? "YES" : "NO");

    printf("\n--- plain greedy ---\n%.*s\n", (int)new_bytes, ref + prompt_len);
    printf("--- BLT-DV ---\n%.*s\n", (int)new_bytes, dv + prompt_len);
    printf("--- raw BLT-D drafts ---\n%.*s\n", (int)new_bytes, draft_only + prompt_len);

    blt_generate_greedy_blockdiff_verify(model, lm, &pcfg, prompt, prompt_len, new_bytes, dv, &gc, &st, scratch);
    printf("\n[E2E] DV stats: enc_glob=%zu dec=%zu drafted=%zu accepted=%zu (%.1f%%)\n", st.nfe_encoder_global,
           st.nfe_decoder, st.bytes_drafted, st.bytes_accepted,
           st.bytes_drafted ? 100.0 * st.bytes_accepted / st.bytes_drafted : 0.0);
    printf("[E2E] baseline dec NFE would be %zu\n", new_bytes);

    free(ref);
    free(dv);
    free(draft_only);
    blt_arena_destroy(model_arena);
    blt_arena_destroy(scratch);
    return identical ? 0 : 1;
}
