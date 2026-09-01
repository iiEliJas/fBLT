// Loads a JSON sweep config, builds the encoder+global+decoder model
// (plus a small co-trained entropy LM for entropy-based patching),
// streams fixed-length windows from train.bin, trains train.steps steps 
// with SGD + global-norm grad clipping
// evaluates held-out BPB (overall + per-domain c/h) every eval_every steps 
// and at he end, and writes ONE bench_result JSON line via bench/harness.h.
//
// xorshift64* RNG seeded from config
// sequential window order, single-threaded math => same seed + same config
// reproduces the identical loss trajectory
// --resume restores weights, step counter and RNG state
//
// Usage:
//   ./bin/train_sweep --config configs/ablations/foo.json
//       [--resume bench/ckpts/foo.ckpt] [--results bench/results.jsonl]
//       [--steps N]   (smoke-test override, do not use for real runs)

#include <math.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "blt/core/allocator.h"
#include "blt/core/backend.h"
#include "blt/core/json.h"
#include "blt/core/tensor.h"
#include "blt/models/model.h"
#include "blt/models/entropy_lm.h"
#include "blt/models/patcher.h"
#include "blt/models/entropy.h"
#include "blt/ops/softmax.h"
#include "blt/ops/optim.h"
#include "blt/ops/vecmath.h"

#include "harness.h"
#include "flops.h"

#define CKPT_MAGIC 0x4B434246ULL   // "FBCK" little-endian
#define CKPT_VERSION 1
#define DEFAULT_CKPT_EVERY 200
#define DEFAULT_EVAL_WINDOWS 128
#define DEFAULT_MAX_WALL_SEC (25 * 60)
#define ENT_LM_EMBED 64
#define ENT_LM_LAYERS 2
#define ENT_LM_HIDDEN 256
#define ENT_LM_HEADS 4
#define GRAD_CLIP_NORM 1.0f

// ---------------------------------------------------------------------------
// RNG (xorshift64*)
// ---------------------------------------------------------------------------

static uint64_t g_rng_state = 0x9E3779B97F4A7C15ULL;

static uint64_t rng_next_u64(void) {
    uint64_t x = g_rng_state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    g_rng_state = x;
    return x * 0x2545F4914F6CDD1DULL;
}


static void fill_uniform(blt_tensor* t, float scale) {
    static uint64_t fill_rng = 0x123456789ABCDEF0ULL;
    size_t n = t->numel;
    float* host_buf = (float*)malloc(n * sizeof(float));
    BLT_REQUIRE(host_buf != NULL, "fill_uniform: staging alloc failed");
    for (size_t i = 0; i < n; i++) {
        uint64_t x = fill_rng;
        x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
        fill_rng = x;
        host_buf[i] = (((float)(x >> 40) / 16777216.0f) * 2.0f - 1.0f) * scale;
    }
    blt_tensor_upload(t, host_buf, n * sizeof(float));
    free(host_buf);
}

static void fill_constant(blt_tensor* t, float v) {
    blt_fill_constant(t->backend, (float*)t->data, t->numel, v);
}

// ---------------------------------------------------------------------------
// Sweep config
// ---------------------------------------------------------------------------

typedef enum {
    PATCH_RULE_ENTROPY_GLOBAL = 0,
    PATCH_RULE_ENTROPY_MONOTONIC,
    PATCH_RULE_ENTROPY_BOTH,
    PATCH_RULE_FIXED,
    PATCH_RULE_WHITESPACE
} sweep_patch_rule;

typedef struct {
    char name[64];
    char tag[64];
    char phase[32];

    size_t enc_embed_dim, enc_num_layers;
    bool ngram_enabled;
    size_t ngram_vocab;
    size_t ngram_sizes[BLT_MAX_NGRAM_SIZES];
    size_t num_ngram_sizes;

    size_t glob_embed_dim, glob_num_layers;
    size_t dec_embed_dim, dec_num_layers;

    blt_xattn_placement enc_xattn;
    blt_xattn_placement dec_xattn;
    bool pooling_init_mean;

    sweep_patch_rule patch_rule;
    blt_patcher_config patcher;
    size_t fixed_stride;

    long steps;
    double lr;
    size_t seq_len;
    uint64_t seed;
    long eval_every;
    long ckpt_every;
    long ent_warmup_steps;
    long eval_windows;
    long max_wall_sec;
    char train_bin[256];
    char heldout_bin[256];
    char heldout_c_bin[256];
    char heldout_h_bin[256];
} sweep_config;

static void warn_unknown_keys(const blt_json_value* obj, const char* const* known,
                              size_t n_known, const char* ctx) {
    if (!obj || obj->type != BLT_JSON_OBJECT) return;
    for (size_t i = 0; i < obj->num_children; i++) {
        bool found = false;
        for (size_t k = 0; k < n_known; k++) {
            if (strcmp(obj->keys[i], known[k]) == 0) { found = true; break; }
        }
        if (!found) {
            BLT_WARN("config %s: unknown key \"%s\" ignored", ctx, obj->keys[i]);
        }
    }
}

static void parse_ngram(const blt_json_value* enc, sweep_config* c) {
    static const char* known[] = {"enabled", "vocab_size", "sizes", "embed_dim"};
    const blt_json_value* ng = blt_json_get(enc, "ngram");
    if (!ng) return;
    warn_unknown_keys(ng, known, 4, "model.encoder.ngram");
    c->ngram_enabled = blt_json_get_bool(ng, "enabled", false);
    c->ngram_vocab = (size_t)blt_json_get_int(ng, "vocab_size", 200000);
    const blt_json_value* sizes = blt_json_get(ng, "sizes");
    size_t n = blt_json_array_size(sizes);
    if (n > BLT_MAX_NGRAM_SIZES) n = BLT_MAX_NGRAM_SIZES;
    c->num_ngram_sizes = n;
    for (size_t i = 0; i < n; i++) {
        c->ngram_sizes[i] = (size_t)blt_json_as_int(blt_json_array_at(sizes, i));
    }
}

// placement strings -> (encoder, decoder) modes. "both" = all layers in both
// modules (matches the unit_model wiring used by the baseline).
static void parse_placement(const char* s, sweep_config* c) {
    if (strcmp(s, "none") == 0) {
        c->enc_xattn = BLT_XATTN_NONE; c->dec_xattn = BLT_XATTN_NONE;
    } else if (strcmp(s, "encoder_all") == 0) {
        c->enc_xattn = BLT_XATTN_ALL; c->dec_xattn = BLT_XATTN_NONE;
    } else if (strcmp(s, "encoder_last") == 0) {
        c->enc_xattn = BLT_XATTN_LAST; c->dec_xattn = BLT_XATTN_NONE;
    } else if (strcmp(s, "decoder_all") == 0) {
        c->enc_xattn = BLT_XATTN_NONE; c->dec_xattn = BLT_XATTN_ALL;
    } else if (strcmp(s, "decoder_first") == 0) {
        c->enc_xattn = BLT_XATTN_NONE; c->dec_xattn = BLT_XATTN_FIRST;
    } else if (strcmp(s, "both") == 0) {
        c->enc_xattn = BLT_XATTN_ALL; c->dec_xattn = BLT_XATTN_ALL;
    } else {
        BLT_FATAL("train_sweep: unknown cross_attention.placement \"%s\"", s);
    }
}

static void parse_patcher_rule(const char* s, sweep_config* c) {
    if (strcmp(s, "global") == 0) {
        c->patch_rule = PATCH_RULE_ENTROPY_GLOBAL;
    } else if (strcmp(s, "monotonic") == 0) {
        c->patch_rule = PATCH_RULE_ENTROPY_MONOTONIC;
    } else if (strcmp(s, "both") == 0) {
        c->patch_rule = PATCH_RULE_ENTROPY_BOTH;
    } else if (strncmp(s, "fixed:", 6) == 0) {
        c->patch_rule = PATCH_RULE_FIXED;
        c->fixed_stride = (size_t)atoi(s + 6);
        if (c->fixed_stride == 0) BLT_FATAL("train_sweep: bad fixed stride in \"%s\"", s);
    } else if (strcmp(s, "whitespace") == 0) {
        c->patch_rule = PATCH_RULE_WHITESPACE;
    } else {
        BLT_FATAL("train_sweep: unknown patcher.rule \"%s\"", s);
    }
}

static void load_config(const char* path, sweep_config* c) {
    memset(c, 0, sizeof(*c));
    snprintf(c->name, sizeof(c->name), "train_eval");
    snprintf(c->heldout_c_bin, sizeof(c->heldout_c_bin), "data/heldout_c.bin");
    snprintf(c->heldout_h_bin, sizeof(c->heldout_h_bin), "data/heldout_h.bin");

    blt_arena* arena = blt_arena_create(1024 * 1024, BLT_BACKEND_CPU);
    blt_json_value* root = blt_json_parse_file(arena, path);
    if (root->type != BLT_JSON_OBJECT) BLT_FATAL("config %s: root must be object", path);

    static const char* top_keys[] = {"name", "tag", "phase", "model",
                                     "cross_attention", "patcher", "train"};
    warn_unknown_keys(root, top_keys, 7, path);

    const char* tag = blt_json_get_string(root, "tag", NULL);
    if (!tag || !tag[0]) BLT_FATAL("config %s: missing tag", path);
    snprintf(c->tag, sizeof(c->tag), "%s", tag);
    const char* name = blt_json_get_string(root, "name", NULL);
    if (name && name[0]) snprintf(c->name, sizeof(c->name), "%s", name);
    const char* phase = blt_json_get_string(root, "phase", "");
    snprintf(c->phase, sizeof(c->phase), "%.31s", phase);

    const blt_json_value* model = blt_json_get(root, "model");
    static const char* model_keys[] = {"encoder", "global", "decoder"};
    warn_unknown_keys(model, model_keys, 3, "model");

    const blt_json_value* enc = blt_json_get(model, "encoder");
    static const char* enc_keys[] = {"embed_dim", "num_layers", "ngram"};
    warn_unknown_keys(enc, enc_keys, 3, "model.encoder");
    c->enc_embed_dim = (size_t)blt_json_get_int(enc, "embed_dim", 128);
    c->enc_num_layers = (size_t)blt_json_get_int(enc, "num_layers", 1);
    parse_ngram(enc, c);

    const blt_json_value* glob = blt_json_get(model, "global");
    static const char* glob_keys[] = {"embed_dim", "num_layers", "num_heads"};
    warn_unknown_keys(glob, glob_keys, 3, "model.global");
    c->glob_embed_dim = (size_t)blt_json_get_int(glob, "embed_dim", 128);
    c->glob_num_layers = (size_t)blt_json_get_int(glob, "num_layers", 2);

    const blt_json_value* dec = blt_json_get(model, "decoder");
    static const char* dec_keys[] = {"embed_dim", "num_layers"};
    warn_unknown_keys(dec, dec_keys, 2, "model.decoder");
    c->dec_embed_dim = (size_t)blt_json_get_int(dec, "embed_dim", 128);
    c->dec_num_layers = (size_t)blt_json_get_int(dec, "num_layers", 2);

    const blt_json_value* xa = blt_json_get(root, "cross_attention");
    static const char* xa_keys[] = {"placement", "pooling_init"};
    warn_unknown_keys(xa, xa_keys, 2, "cross_attention");
    parse_placement(blt_json_get_string(xa, "placement", "both"), c);
    c->pooling_init_mean = blt_json_get_bool(xa, "pooling_init", true);

    const blt_json_value* pa = blt_json_get(root, "patcher");
    static const char* pa_keys[] = {"rule", "threshold_global", "threshold_monotonic",
                                    "max_patch_length"};
    warn_unknown_keys(pa, pa_keys, 4, "patcher");
    parse_patcher_rule(blt_json_get_string(pa, "rule", "global"), c);
    c->patcher.threshold_global =
        (float)blt_json_get_float(pa, "threshold_global", 1.0);
    c->patcher.threshold_monotonic =
        (float)blt_json_get_float(pa, "threshold_monotonic", 0.5);
    c->patcher.max_patch_length =
        (size_t)blt_json_get_int(pa, "max_patch_length", 64);

    const blt_json_value* tr = blt_json_get(root, "train");
    static const char* tr_keys[] = {"steps", "lr", "seq_len", "seed", "eval_every",
                                    "train_bin", "heldout_bin", "heldout_c_bin",
                                    "heldout_h_bin", "ckpt_every", "max_wall_min",
                                    "eval_windows", "ent_warmup_steps"};
    warn_unknown_keys(tr, tr_keys, 13, "train");
    c->steps = (long)blt_json_get_int(tr, "steps", 1000);
    c->lr = blt_json_get_float(tr, "lr", 0.01f);
    c->seq_len = (size_t)blt_json_get_int(tr, "seq_len", 512);
    c->seed = (uint64_t)blt_json_get_int(tr, "seed", 42);
    c->eval_every = (long)blt_json_get_int(tr, "eval_every", 500);
    c->ckpt_every = (long)blt_json_get_int(tr, "ckpt_every", DEFAULT_CKPT_EVERY);
    c->ent_warmup_steps = (long)blt_json_get_int(tr, "ent_warmup_steps", 0);
    c->eval_windows = (long)blt_json_get_int(tr, "eval_windows", DEFAULT_EVAL_WINDOWS);
    c->max_wall_sec = 60L * (long)blt_json_get_int(tr, "max_wall_min", 25);
    const char* tb = blt_json_get_string(tr, "train_bin", "data/train.bin");
    snprintf(c->train_bin, sizeof(c->train_bin), "%.255s", tb);
    const char* hb = blt_json_get_string(tr, "heldout_bin", "data/heldout.bin");
    snprintf(c->heldout_bin, sizeof(c->heldout_bin), "%.255s", hb);
    const char* hc = blt_json_get_string(tr, "heldout_c_bin", NULL);
    if (hc) snprintf(c->heldout_c_bin, sizeof(c->heldout_c_bin), "%.255s", hc);
    const char* hh = blt_json_get_string(tr, "heldout_h_bin", NULL);
    if (hh) snprintf(c->heldout_h_bin, sizeof(c->heldout_h_bin), "%.255s", hh);

    if (c->patcher.max_patch_length == 0 || c->patcher.max_patch_length > c->seq_len) {
        c->patcher.max_patch_length = c->seq_len;
    }
    c->patcher.rule = BLT_PATCH_RULE_GLOBAL;
    switch (c->patch_rule) {
        case PATCH_RULE_ENTROPY_GLOBAL:
            c->patcher.rule = BLT_PATCH_RULE_GLOBAL; break;
        case PATCH_RULE_ENTROPY_MONOTONIC:
            c->patcher.rule = BLT_PATCH_RULE_MONOTONIC; break;
        case PATCH_RULE_ENTROPY_BOTH:
            c->patcher.rule = BLT_PATCH_RULE_BOTH; break;
        default:
            break;
    }

    blt_arena_destroy(arena);
}
// ---------------------------------------------------------------------------
// Model building + deterministic init
// ---------------------------------------------------------------------------

#define INIT_SCALE 0.05f

static void build_model_config(const sweep_config* c, blt_model_config* mc) {
    memset(mc, 0, sizeof(*mc));
    size_t heads = c->enc_embed_dim / 32;
    if (heads == 0) heads = 1;

    mc->encoder_config.embed_dim = c->enc_embed_dim;
    mc->encoder_config.patch_dim = 0;
    mc->encoder_config.num_layers = c->enc_num_layers;
    mc->encoder_config.hidden_dim = 4 * c->enc_embed_dim;
    mc->encoder_config.num_heads = heads;
    mc->encoder_config.cross_attn_heads = heads;
    mc->encoder_config.local_window = 0;
    mc->encoder_config.cross_attn_placement = c->enc_xattn;
    mc->encoder_config.pool_type =
        c->pooling_init_mean ? BLT_POOL_MEAN : BLT_POOL_MAX;
    mc->encoder_config.rope_theta = 10000.0f;
    mc->encoder_config.max_seq_len = c->seq_len;

    // ngram_config fields are required valid even when the module is disabled
    // (blt_local_encoder_create validates them unconditionally)
    mc->encoder_config.ngram_config.embed_dim = c->enc_embed_dim;
    mc->encoder_config.ngram_config.per_ngram_vocab = c->ngram_vocab;
    mc->encoder_config.ngram_config.hash_prime = 1000000007ULL;
    mc->encoder_config.ngram_config.normalize = true;
    if (c->ngram_enabled && c->num_ngram_sizes > 0) {
        for (size_t i = 0; i < c->num_ngram_sizes; i++) {
            mc->encoder_config.ngram_config.ngram_sizes[i] = c->ngram_sizes[i];
        }
        mc->encoder_config.ngram_config.num_ngram_sizes = c->num_ngram_sizes;
    }

    size_t gheads = c->glob_embed_dim / 32;
    if (gheads == 0) gheads = 1;
    mc->global_config.embed_dim = c->glob_embed_dim;
    mc->global_config.num_layers = c->glob_num_layers;
    mc->global_config.hidden_dim = 4 * c->glob_embed_dim;
    mc->global_config.num_heads = gheads;
    mc->global_config.rope_theta = 10000.0f;
    mc->global_config.max_seq_len = c->seq_len; // upper bound on num_patches

    size_t dheads = c->dec_embed_dim / 32;
    if (dheads == 0) dheads = 1;
    mc->decoder_config.embed_dim = c->dec_embed_dim;
    mc->decoder_config.patch_dim = 0;
    mc->decoder_config.num_layers = c->dec_num_layers;
    mc->decoder_config.hidden_dim = 4 * c->dec_embed_dim;
    mc->decoder_config.num_heads = dheads;
    mc->decoder_config.cross_attn_heads = dheads;
    mc->decoder_config.local_window = 0;
    mc->decoder_config.cross_attn_placement = c->dec_xattn;
    mc->decoder_config.rope_theta = 10000.0f;
    mc->decoder_config.max_seq_len = c->seq_len;
    mc->decoder_config.vocab_size = 256;
}

static void fill_layer_transformer(blt_transformer_layer_storage* s) {
    fill_constant(&s->norm1_weight, 1.0f);
    fill_uniform(&s->attn_qkv_w, INIT_SCALE);
    fill_uniform(&s->attn_proj_w, INIT_SCALE);
    fill_constant(&s->norm2_weight, 1.0f);
    fill_uniform(&s->ffn_up_w, INIT_SCALE);
    fill_uniform(&s->ffn_gate_w, INIT_SCALE);
    fill_uniform(&s->ffn_down_w, INIT_SCALE);
}

static void fill_layer_cross(blt_local_layer_storage* s) {
    fill_constant(&s->cross_norm_weight, 1.0f);
    fill_uniform(&s->cross_weight_q, INIT_SCALE);
    fill_uniform(&s->cross_weight_k, INIT_SCALE);
    fill_uniform(&s->cross_weight_v, INIT_SCALE);
    fill_uniform(&s->cross_weight_proj, INIT_SCALE);
}

static void init_model_weights(blt_model* m) {
    fill_uniform(&m->encoder->byte_embedding_weight, INIT_SCALE);
    for (size_t i = 0; i < m->encoder->ngram_weights.num_tables; i++) {
        fill_uniform(&m->encoder->ngram_weights.tables[i], INIT_SCALE);
    }
    for (size_t l = 0; l < m->encoder->config.num_layers; l++) {
        fill_layer_cross(&m->encoder->layers[l]);
        fill_layer_transformer((blt_transformer_layer_storage*)&m->encoder->layers[l]);
    }
    for (size_t l = 0; l < m->global->stack.num_layers; l++) {
        fill_layer_transformer(&m->global->stack.layer_storage[l]);
    }
    for (size_t l = 0; l < m->decoder->config.num_layers; l++) {
        fill_layer_cross(&m->decoder->layers[l]);
        fill_layer_transformer((blt_transformer_layer_storage*)&m->decoder->layers[l]);
    }
    fill_uniform(&m->decoder->lm_head_weight, INIT_SCALE);
}

static bool uses_entropy_patcher(const sweep_config* c) {
    return c->patch_rule == PATCH_RULE_ENTROPY_GLOBAL ||
           c->patch_rule == PATCH_RULE_ENTROPY_MONOTONIC ||
           c->patch_rule == PATCH_RULE_ENTROPY_BOTH;
}

static void init_entropy_lm(blt_entropy_lm* lm) {
    fill_uniform(&lm->embedding_weight, INIT_SCALE);
    for (size_t l = 0; l < lm->stack.num_layers; l++) {
        fill_layer_transformer(&lm->stack.layer_storage[l]);
    }
    fill_uniform(&lm->lm_head_weight, INIT_SCALE);
}
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Data + windows
//
// Windows never cross file boundaries: they are carved from the manifest's
// per-file {offset,length} index (floor(len/seq_len) per file, tails
// dropped). Order is deterministic; training wraps around after one epoch.
// ---------------------------------------------------------------------------

typedef struct {
    uint8_t* data;
    size_t len;
} byte_buf;

static void load_file(const char* path, byte_buf* b) {
    FILE* f = fopen(path, "rb");
    if (!f) BLT_FATAL("load_file: cannot open %s", path);
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0) BLT_FATAL("load_file: empty file %s", path);
    b->data = (uint8_t*)malloc((size_t)sz + 1);
    if (!b->data) BLT_FATAL("load_file: OOM for %s", path);
    if (fread(b->data, 1, (size_t)sz, f) != (size_t)sz) {
        BLT_FATAL("load_file: short read on %s", path);
    }
    b->data[sz] = '\0';
    fclose(f);
    b->len = (size_t)sz;
}

typedef struct {
    size_t offset;
} window;

typedef struct {
    window* items;
    size_t num;
} window_list;

// Carve windows from a raw stream (used for heldout domain bins).
static void windows_from_stream(size_t stream_len, size_t seq_len,
                                long max_windows, window_list* wl) {
    size_t n = stream_len / seq_len;
    if (max_windows > 0 && (long)n > max_windows) n = (size_t)max_windows;
    wl->items = (window*)malloc(n * sizeof(window));
    if (!wl->items && n > 0) BLT_FATAL("windows_from_stream: OOM");
    for (size_t i = 0; i < n; i++) wl->items[i].offset = i * seq_len;
    wl->num = n;
}

// Carve windows from the manifest index of train.bin/heldout.bin.
static void windows_from_manifest(const byte_buf* manifest_bytes,
                                  const char* split, size_t seq_len,
                                  long max_windows, window_list* wl) {
    // manifest.json is several MB; give the parse arena generous headroom
    blt_arena* arena = blt_arena_create(64 * 1024 * 1024, BLT_BACKEND_CPU);
    blt_json_value* root = blt_json_parse(arena, (const char*)manifest_bytes->data);
    const blt_json_value* idx = blt_json_get(blt_json_get(root, "index"), split);
    size_t total = blt_json_array_size(idx);

    // First pass: count.
    size_t count = 0;
    for (size_t i = 0; i < total; i++) {
        const blt_json_value* e = blt_json_array_at(idx, i);
        int64_t off = blt_json_get_int(e, "offset", -1);
        int64_t len = blt_json_get_int(e, "length", -1);
        if (off < 0 || len < (int64_t)seq_len) continue;
        count += (size_t)len / seq_len;
    }
    if (max_windows > 0 && (long)count > max_windows) count = (size_t)max_windows;

    wl->items = (window*)malloc((count ? count : 1) * sizeof(window));
    if (!wl->items) BLT_FATAL("windows_from_manifest: OOM");
    wl->num = 0;
    for (size_t i = 0; i < total && wl->num < count; i++) {
        const blt_json_value* e = blt_json_array_at(idx, i);
        int64_t off = blt_json_get_int(e, "offset", -1);
        int64_t len = blt_json_get_int(e, "length", -1);
        if (off < 0 || len < (int64_t)seq_len) continue;
        size_t nw = (size_t)len / seq_len;
        for (size_t wj = 0; wj < nw && wl->num < count; wj++) {
            wl->items[wl->num++].offset = (size_t)off + wj * seq_len;
        }
    }
    blt_arena_destroy(arena);
}

// ---------------------------------------------------------------------------
// Patch construction (trainer-level; model code stays agnostic)
// ---------------------------------------------------------------------------

static size_t build_fixed_patches(size_t seq_len, size_t stride,
                                  blt_patch_info* out) {
    size_t n = 0, start = 0;
    while (start < seq_len) {
        size_t len = (start + stride <= seq_len) ? stride : seq_len - start;
        out[n].start_idx = start;
        out[n].length = len;
        out[n].peak_entropy = 0.0f;
        n++;
        start += len;
    }
    return n;
}

static bool is_ws_break(uint8_t c) {
    return c == ' ' || c == '\n' || c == '\t' || c == '\r';
}

static size_t build_whitespace_patches(const uint8_t* bytes, size_t seq_len,
                                       size_t max_len, blt_patch_info* out) {
    size_t n = 0, start = 0;
    while (start < seq_len) {
        size_t len = 0;
        while (start + len < seq_len && len < max_len &&
               !(len > 0 && is_ws_break(bytes[start + len - 1]))) {
            len++;
        }
        if (len == 0) len = 1;
        out[n].start_idx = start;
        out[n].length = len;
        out[n].peak_entropy = 0.0f;
        n++;
        start += len;
    }
    return n;
}
// ---------------------------------------------------------------------------
// Per-step pipeline: entropy -> patches -> model forward
//
// The entropy LM is co-trained on its own next-byte loss only; because it
// sees the same data in the same order with the same seed in every run,
// patch boundaries are identical across all sweep configs.
// ---------------------------------------------------------------------------

typedef struct {
    blt_model* model;
    blt_entropy_lm* ent_lm;
    const sweep_config* cfg;
} train_ctx;

static size_t make_patches(const train_ctx* tc, const uint8_t* window_bytes,
                           size_t seq_len, blt_tensor* bytes_in,
                           blt_arena* arena, blt_patch_info* patches) {
    if (!uses_entropy_patcher(tc->cfg)) {
        if (tc->cfg->patch_rule == PATCH_RULE_FIXED) {
            return build_fixed_patches(seq_len, tc->cfg->fixed_stride, patches);
        }
        return build_whitespace_patches(window_bytes, seq_len,
                                        tc->cfg->patcher.max_patch_length, patches);
    }

    size_t logits_shape[2] = {seq_len, 256};
    blt_tensor ent_logits = blt_tensor_create(arena, logits_shape, 2, BLT_DTYPE_FP32);
    size_t scalar_shape[1] = {1};
    blt_tensor ent_loss = blt_tensor_create(arena, scalar_shape, 1, BLT_DTYPE_FP32);
    blt_entropy_lm_forward(tc->ent_lm, bytes_in, &ent_logits, &ent_loss, arena);

    blt_tensor probs = blt_tensor_create(arena, logits_shape, 2, BLT_DTYPE_FP32);
    blt_softmax(&ent_logits, &probs);

    size_t ent_shape[1] = {seq_len};
    blt_tensor ent_vals = blt_tensor_create(arena, ent_shape, 1, BLT_DTYPE_FP32);
    blt_entropy_config ecfg = {.vocab_size = 256, .use_log2 = false};
    blt_compute_entropy(&probs, &ent_vals, &ecfg);

    // blt_segment_patches expects CPU pointers; copy entropy to host if on CUDA
    float* ent_host = (float*)malloc(seq_len * sizeof(float));
    blt_tensor_copy_to_host(&ent_vals, ent_host, seq_len * sizeof(float));

    blt_tensor ent_vals_host;
    memset(&ent_vals_host, 0, sizeof(ent_vals_host));
    ent_vals_host.data = ent_host;
    ent_vals_host.shape[0] = seq_len;
    ent_vals_host.ndim = 1;
    ent_vals_host.numel = seq_len;
    ent_vals_host.dtype = BLT_DTYPE_FP32;
    ent_vals_host.backend = BLT_BACKEND_CPU;
    ent_vals_host.strides[0] = 1;

    size_t n = blt_segment_patches(&ent_vals_host, window_bytes, patches, seq_len,
                               &tc->cfg->patcher);
    free(ent_host);
    return n;
}

static void forward_model(const train_ctx* tc, blt_tensor* bytes_in,
                          const blt_patch_info* patches, size_t num_patches,
                          blt_tensor* logits, blt_tensor* loss, blt_arena* arena) {
    size_t doc_boundaries[1] = {0};
    blt_model_forward(tc->model, bytes_in, patches, num_patches,
                      doc_boundaries, 1, logits, loss, arena);
}

// ---------------------------------------------------------------------------
// Grad clipping + SGD (mirrors tests/unit/models/unit_model.c wiring)
// ---------------------------------------------------------------------------

static float tensor_sq_norm(const blt_tensor* t) {
    float* d_host = (float*)malloc(t->numel * sizeof(float));
    blt_tensor_copy_to_host(t, d_host, t->numel * sizeof(float));
    float sum = 0.0f;
    for (size_t i = 0; i < t->numel; i++) sum += d_host[i] * d_host[i];
    free(d_host);
    return sum;
}

static float model_grad_sq_norm(blt_model_grad* g, const blt_model* m) {
    float total = tensor_sq_norm(&g->encoder_grad->embedding_grad);
    for (size_t i = 0; i < m->encoder->ngram_weights.num_tables; i++) {
        total += tensor_sq_norm(&g->encoder_grad->ngram_grads.tables[i]);
    }
    for (size_t l = 0; l < m->encoder->config.num_layers; l++) {
        blt_local_layer_grad* s = &g->encoder_grad->layer_grads[l];
        total += tensor_sq_norm(&s->norm1_weight) + tensor_sq_norm(&s->attn_qkv_w) +
                 tensor_sq_norm(&s->attn_proj_w) + tensor_sq_norm(&s->norm2_weight) +
                 tensor_sq_norm(&s->ffn_up_w) + tensor_sq_norm(&s->ffn_gate_w) +
                 tensor_sq_norm(&s->ffn_down_w) + tensor_sq_norm(&s->cross_norm_weight) +
                 tensor_sq_norm(&s->cross_weight_q) + tensor_sq_norm(&s->cross_weight_k) +
                 tensor_sq_norm(&s->cross_weight_v) + tensor_sq_norm(&s->cross_weight_proj);
    }
    for (size_t l = 0; l < m->global->stack.num_layers; l++) {
        blt_transformer_layer_grad* s = &g->global_grad->stack_grad->layer_grads[l];
        total += tensor_sq_norm(&s->norm1_weight) + tensor_sq_norm(&s->attn_qkv_w) +
                 tensor_sq_norm(&s->attn_proj_w) + tensor_sq_norm(&s->norm2_weight) +
                 tensor_sq_norm(&s->ffn_up_w) + tensor_sq_norm(&s->ffn_gate_w) +
                 tensor_sq_norm(&s->ffn_down_w);
    }
    for (size_t l = 0; l < m->decoder->config.num_layers; l++) {
        blt_local_layer_grad* s = &g->decoder_grad->layer_grads[l];
        total += tensor_sq_norm(&s->cross_norm_weight) + tensor_sq_norm(&s->cross_weight_q) +
                 tensor_sq_norm(&s->cross_weight_k) + tensor_sq_norm(&s->cross_weight_v) +
                 tensor_sq_norm(&s->cross_weight_proj) + tensor_sq_norm(&s->norm1_weight) +
                 tensor_sq_norm(&s->attn_qkv_w) + tensor_sq_norm(&s->attn_proj_w) +
                 tensor_sq_norm(&s->norm2_weight) + tensor_sq_norm(&s->ffn_up_w) +
                 tensor_sq_norm(&s->ffn_gate_w) + tensor_sq_norm(&s->ffn_down_w);
    }
    total += tensor_sq_norm(&g->decoder_grad->lm_head_grad);
    return total;
}
static void scale_model_grads(blt_model_grad* g, const blt_model* m, float s) {
    blt_scale(&g->encoder_grad->embedding_grad, s);
    for (size_t i = 0; i < m->encoder->ngram_weights.num_tables; i++) {
        blt_scale(&g->encoder_grad->ngram_grads.tables[i], s);
    }
    for (size_t l = 0; l < m->encoder->config.num_layers; l++) {
        blt_local_layer_grad* t = &g->encoder_grad->layer_grads[l];
        blt_tensor* all[12] = {&t->norm1_weight, &t->attn_qkv_w, &t->attn_proj_w,
                               &t->norm2_weight, &t->ffn_up_w, &t->ffn_gate_w,
                               &t->ffn_down_w, &t->cross_norm_weight,
                               &t->cross_weight_q, &t->cross_weight_k,
                               &t->cross_weight_v, &t->cross_weight_proj};
        for (int i = 0; i < 12; i++) blt_scale(all[i], s);
    }
    for (size_t l = 0; l < m->global->stack.num_layers; l++) {
        blt_transformer_layer_grad* t = &g->global_grad->stack_grad->layer_grads[l];
        blt_tensor* all[7] = {&t->norm1_weight, &t->attn_qkv_w, &t->attn_proj_w,
                              &t->norm2_weight, &t->ffn_up_w, &t->ffn_gate_w,
                              &t->ffn_down_w};
        for (int i = 0; i < 7; i++) blt_scale(all[i], s);
    }
    for (size_t l = 0; l < m->decoder->config.num_layers; l++) {
        blt_local_layer_grad* t = &g->decoder_grad->layer_grads[l];
        blt_tensor* all[12] = {&t->cross_norm_weight, &t->cross_weight_q,
                               &t->cross_weight_k, &t->cross_weight_v,
                               &t->cross_weight_proj, &t->norm1_weight,
                               &t->attn_qkv_w, &t->attn_proj_w,
                               &t->norm2_weight, &t->ffn_up_w,
                               &t->ffn_gate_w, &t->ffn_down_w};
        for (int i = 0; i < 12; i++) blt_scale(all[i], s);
    }
    blt_scale(&g->decoder_grad->lm_head_grad, s);
}

static void clip_model_grads(blt_model_grad* g, const blt_model* m) {
    float sq = model_grad_sq_norm(g, m);
    if (sq > GRAD_CLIP_NORM * GRAD_CLIP_NORM && sq > 0.0f) {
        scale_model_grads(g, m, GRAD_CLIP_NORM / sqrtf(sq));
    }
}

static void zero_scatter_grads(blt_model_grad* g, const blt_local_encoder* enc, const blt_model_config* cfg) {
    zero_tensor(&g->encoder_grad->embedding_grad);
    for (size_t i = 0; i < enc->ngram_weights.num_tables; i++) {
        zero_tensor(&g->encoder_grad->ngram_grads.tables[i]);
    }
    for (size_t l = 0; l < enc->config.num_layers; l++) {
        blt_local_layer_grad* s = &g->encoder_grad->layer_grads[l];
        zero_tensor(&s->norm1_weight); zero_tensor(&s->attn_qkv_w);
        zero_tensor(&s->attn_proj_w); zero_tensor(&s->norm2_weight);
        zero_tensor(&s->ffn_up_w); zero_tensor(&s->ffn_gate_w);
        zero_tensor(&s->ffn_down_w); zero_tensor(&s->cross_norm_weight);
        zero_tensor(&s->cross_weight_q); zero_tensor(&s->cross_weight_k);
        zero_tensor(&s->cross_weight_v); zero_tensor(&s->cross_weight_proj);
    }
    for (size_t l = 0; l < cfg->global_config.num_layers; l++) {
        blt_transformer_layer_grad* s = &g->global_grad->stack_grad->layer_grads[l];
        zero_tensor(&s->norm1_weight); zero_tensor(&s->attn_qkv_w);
        zero_tensor(&s->attn_proj_w); zero_tensor(&s->norm2_weight);
        zero_tensor(&s->ffn_up_w); zero_tensor(&s->ffn_gate_w);
        zero_tensor(&s->ffn_down_w);
    }
    for (size_t l = 0; l < cfg->decoder_config.num_layers; l++) {
        blt_local_layer_grad* s = &g->decoder_grad->layer_grads[l];
        zero_tensor(&s->cross_norm_weight); zero_tensor(&s->cross_weight_q);
        zero_tensor(&s->cross_weight_k); zero_tensor(&s->cross_weight_v);
        zero_tensor(&s->cross_weight_proj); zero_tensor(&s->norm1_weight);
        zero_tensor(&s->attn_qkv_w); zero_tensor(&s->attn_proj_w);
        zero_tensor(&s->norm2_weight); zero_tensor(&s->ffn_up_w);
        zero_tensor(&s->ffn_gate_w); zero_tensor(&s->ffn_down_w);
    }
    zero_tensor(&g->decoder_grad->lm_head_grad);
}

static void sgd_layer_transformer(blt_transformer_layer_storage* w,
                                  blt_transformer_layer_grad* g, float lr) {
    blt_sgd_step(&w->norm1_weight, &g->norm1_weight, lr);
    blt_sgd_step(&w->attn_qkv_w, &g->attn_qkv_w, lr);
    blt_sgd_step(&w->attn_proj_w, &g->attn_proj_w, lr);
    blt_sgd_step(&w->norm2_weight, &g->norm2_weight, lr);
    blt_sgd_step(&w->ffn_up_w, &g->ffn_up_w, lr);
    blt_sgd_step(&w->ffn_gate_w, &g->ffn_gate_w, lr);
    blt_sgd_step(&w->ffn_down_w, &g->ffn_down_w, lr);
}

static void sgd_apply_model(blt_model* m, blt_model_grad* g, float lr) {
    blt_sgd_step(&m->encoder->byte_embedding_weight, &g->encoder_grad->embedding_grad, lr);
    for (size_t i = 0; i < m->encoder->ngram_weights.num_tables; i++) {
        blt_sgd_step(&m->encoder->ngram_weights.tables[i],
                     &g->encoder_grad->ngram_grads.tables[i], lr);
    }
    for (size_t l = 0; l < m->encoder->config.num_layers; l++) {
        blt_local_layer_storage* w = &m->encoder->layers[l];
        blt_local_layer_grad* t = &g->encoder_grad->layer_grads[l];
        blt_sgd_step(&w->norm1_weight, &t->norm1_weight, lr);
        blt_sgd_step(&w->attn_qkv_w, &t->attn_qkv_w, lr);
        blt_sgd_step(&w->attn_proj_w, &t->attn_proj_w, lr);
        blt_sgd_step(&w->norm2_weight, &t->norm2_weight, lr);
        blt_sgd_step(&w->ffn_up_w, &t->ffn_up_w, lr);
        blt_sgd_step(&w->ffn_gate_w, &t->ffn_gate_w, lr);
        blt_sgd_step(&w->ffn_down_w, &t->ffn_down_w, lr);
        blt_sgd_step(&w->cross_norm_weight, &t->cross_norm_weight, lr);
        blt_sgd_step(&w->cross_weight_q, &t->cross_weight_q, lr);
        blt_sgd_step(&w->cross_weight_k, &t->cross_weight_k, lr);
        blt_sgd_step(&w->cross_weight_v, &t->cross_weight_v, lr);
        blt_sgd_step(&w->cross_weight_proj, &t->cross_weight_proj, lr);
    }
    for (size_t l = 0; l < m->global->stack.num_layers; l++) {
        sgd_layer_transformer(&m->global->stack.layer_storage[l],
                              &g->global_grad->stack_grad->layer_grads[l], lr);
    }
    for (size_t l = 0; l < m->decoder->config.num_layers; l++) {
        blt_local_layer_storage* w = &m->decoder->layers[l];
        blt_local_layer_grad* t = &g->decoder_grad->layer_grads[l];
        blt_sgd_step(&w->cross_norm_weight, &t->cross_norm_weight, lr);
        blt_sgd_step(&w->cross_weight_q, &t->cross_weight_q, lr);
        blt_sgd_step(&w->cross_weight_k, &t->cross_weight_k, lr);
        blt_sgd_step(&w->cross_weight_v, &t->cross_weight_v, lr);
        blt_sgd_step(&w->cross_weight_proj, &t->cross_weight_proj, lr);
        blt_sgd_step(&w->norm1_weight, &t->norm1_weight, lr);
        blt_sgd_step(&w->attn_qkv_w, &t->attn_qkv_w, lr);
        blt_sgd_step(&w->attn_proj_w, &t->attn_proj_w, lr);
        blt_sgd_step(&w->norm2_weight, &t->norm2_weight, lr);
        blt_sgd_step(&w->ffn_up_w, &t->ffn_up_w, lr);
        blt_sgd_step(&w->ffn_gate_w, &t->ffn_gate_w, lr);
        blt_sgd_step(&w->ffn_down_w, &t->ffn_down_w, lr);
    }
    blt_sgd_step(&m->decoder->lm_head_weight, &g->decoder_grad->lm_head_grad, lr);
}

// Entropy LM: grads are overwritten per call except the embedding scatter-add.
static void clip_entropy_grads(blt_entropy_lm* m, blt_entropy_lm_grad* g) {
    float total = tensor_sq_norm(&g->embedding_grad);
    for (size_t l = 0; l < m->stack.num_layers; l++) {
        blt_transformer_layer_grad* s = &g->stack_grad->layer_grads[l];
        total += tensor_sq_norm(&s->norm1_weight) + tensor_sq_norm(&s->attn_qkv_w) +
                 tensor_sq_norm(&s->attn_proj_w) + tensor_sq_norm(&s->norm2_weight) +
                 tensor_sq_norm(&s->ffn_up_w) + tensor_sq_norm(&s->ffn_gate_w) +
                 tensor_sq_norm(&s->ffn_down_w);
    }
    total += tensor_sq_norm(&g->lm_head_grad);
    if (total > GRAD_CLIP_NORM * GRAD_CLIP_NORM && total > 0.0f) {
        float s = GRAD_CLIP_NORM / sqrtf(total);
        blt_scale(&g->embedding_grad, s);
        for (size_t l = 0; l < m->stack.num_layers; l++) {
            blt_transformer_layer_grad* t = &g->stack_grad->layer_grads[l];
            blt_tensor* all[7] = {&t->norm1_weight, &t->attn_qkv_w, &t->attn_proj_w,
                                  &t->norm2_weight, &t->ffn_up_w, &t->ffn_gate_w,
                                  &t->ffn_down_w};
            for (int i = 0; i < 7; i++) blt_scale(all[i], s);
        }
        blt_scale(&g->lm_head_grad, s);
    }
}

static void zero_entropy_scatter(blt_entropy_lm_grad* g, const blt_entropy_lm* lm) {
    zero_tensor(&g->embedding_grad);
    for (size_t l = 0; l < lm->stack.num_layers; l++) {
        blt_transformer_layer_grad* s = &g->stack_grad->layer_grads[l];
        zero_tensor(&s->norm1_weight); zero_tensor(&s->attn_qkv_w);
        zero_tensor(&s->attn_proj_w); zero_tensor(&s->norm2_weight);
        zero_tensor(&s->ffn_up_w); zero_tensor(&s->ffn_gate_w);
        zero_tensor(&s->ffn_down_w);
    }
    zero_tensor(&g->lm_head_grad);
}

static void sgd_apply_entropy_lm(blt_entropy_lm* m, blt_entropy_lm_grad* g, float lr) {
    blt_sgd_step(&m->embedding_weight, &g->embedding_grad, lr);
    for (size_t l = 0; l < m->stack.num_layers; l++) {
        sgd_layer_transformer(&m->stack.layer_storage[l],
                              &g->stack_grad->layer_grads[l], lr);
    }
    blt_sgd_step(&m->lm_head_weight, &g->lm_head_grad, lr);
}

// ---------------------------------------------------------------------------
// Evaluation: held-out BPB (bits per byte) over a fixed window subset
//
// BPB = sum(shifted cross-entropy) / (ln 2 * positions). The loss tensor from
// blt_model_forward is the MEAN over the seq_len-1 shifted target positions.
// ---------------------------------------------------------------------------

typedef struct {
    double bpb;
    double avg_patch_len;
} eval_out;

static void eval_stream(const train_ctx* tc, const byte_buf* buf,
                        const window_list* wl, blt_arena* arena, eval_out* out) {
    const size_t L = tc->cfg->seq_len;
    double loss_sum = 0.0;
    size_t positions = 0;
    double patch_bytes = 0.0, patch_count = 0.0;

    size_t logits_shape[2] = {L, 256};
    size_t scalar_shape[1] = {1};

    for (size_t i = 0; i < wl->num; i++) {
        blt_arena_reset(arena);
        const uint8_t* wbytes = buf->data + wl->items[i].offset;

        size_t bytes_shape[1] = {L};
        blt_tensor bytes_in = blt_tensor_create(arena, bytes_shape, 1, BLT_DTYPE_UINT8);
        blt_tensor_copy_from_host(&bytes_in, wbytes, L);

        blt_patch_info* patches = (blt_patch_info*)malloc(L * sizeof(blt_patch_info));
        size_t num_patches = make_patches(tc, wbytes, L, &bytes_in, arena, patches);
        if (num_patches == 0) {
            free(patches);
            continue;
        }

        blt_tensor logits = blt_tensor_create(arena, logits_shape, 2, BLT_DTYPE_FP32);
        blt_tensor loss = blt_tensor_create(arena, scalar_shape, 1, BLT_DTYPE_FP32);
        forward_model(tc, &bytes_in, patches, num_patches, &logits, &loss, arena);

        float loss_val = 0.0f;
        blt_tensor_copy_to_host(&loss, &loss_val, sizeof(float));
        loss_sum += loss_val * (double)(L - 1);
        positions += L - 1;
        for (size_t p = 0; p < num_patches; p++) {
            patch_bytes += (double)patches[p].length;
            patch_count += 1.0;
        }
        free(patches);
    }

    out->bpb = (positions > 0)
        ? loss_sum / (log(2.0) * (double)positions) : 0.0;
    out->avg_patch_len = (patch_count > 0.0) ? patch_bytes / patch_count : 0.0;
}

// ---------------------------------------------------------------------------
// Wall clock
// ---------------------------------------------------------------------------

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static volatile sig_atomic_t g_stop_requested = 0;

static void on_signal(int sig) {
    (void)sig;
    g_stop_requested = 1;
}
// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

static void usage(const char* argv0) {
    fprintf(stderr,
            "usage: %s --config <json> [--resume <ckpt>] "
            "[--results bench/results.jsonl] [--steps N] [--seed N] [--backend cpu|cuda]\n",
            argv0);
}

int main(int argc, char** argv) {
    const char* config_path = NULL;
    const char* resume_path = NULL;
    const char* results_path = "bench/results.jsonl";
    long steps_override = -1;
    long long seed_override = -1;
    blt_backend backend = BLT_BACKEND_CPU;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--config") == 0 && i + 1 < argc) {
            config_path = argv[++i];
        } else if (strcmp(argv[i], "--resume") == 0 && i + 1 < argc) {
            resume_path = argv[++i];
        } else if (strcmp(argv[i], "--results") == 0 && i + 1 < argc) {
            results_path = argv[++i];
        } else if (strcmp(argv[i], "--steps") == 0 && i + 1 < argc) {
            steps_override = atol(argv[++i]);
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            seed_override = atoll(argv[++i]);
        } else if (strcmp(argv[i], "--backend") == 0 && i + 1 < argc) {
            const char* b = argv[++i];
            if (strcmp(b, "cpu") == 0) backend = BLT_BACKEND_CPU;
            else if (strcmp(b, "cuda") == 0) backend = BLT_BACKEND_CUDA;
            else { fprintf(stderr, "Invalid backend: %s (cpu|cuda)\n", b); return 2; }
        } else {
            usage(argv[0]);
            return 2;
        }
    }
    if (!config_path) {
        usage(argv[0]);
        return 2;
    }

    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);

    sweep_config cfg;
    load_config(config_path, &cfg);
    if (seed_override >= 0) cfg.seed = (uint64_t)seed_override;
    long total_steps = (steps_override > 0) ? steps_override : cfg.steps;

    printf("[sweep] tag=%s phase=%s steps=%ld seq_len=%zu lr=%g seed=%llu\n",
           cfg.tag, cfg.phase, total_steps, cfg.seq_len, cfg.lr,
           (unsigned long long)cfg.seed);

    // ---- RNG seed (splitmix64 spread so nearby seeds diverge fast)
    g_rng_state = cfg.seed ^ 0x9E3779B97F4A7C15ULL;
    rng_next_u64();

    printf("[sweep] backend=%s\n", backend == BLT_BACKEND_CUDA ? "cuda" : "cpu");

    // ---- Model + entropy LM
    blt_model_config mc;
    build_model_config(&cfg, &mc);

    blt_arena* model_arena = blt_arena_create(2048UL * 1024 * 1024, backend);
    blt_arena* scratch = blt_arena_create(4096UL * 1024 * 1024, backend);
    if (!model_arena || !scratch) BLT_FATAL("main: arena creation failed");

    blt_model* model = blt_model_create(model_arena, &mc);
    blt_model_grad* grad = blt_model_grad_create(model_arena, model);
    if (!model || !grad) BLT_FATAL("main: model creation failed");
    init_model_weights(model);

    blt_entropy_lm_config ecfg = {0};
    ecfg.embed_dim = ENT_LM_EMBED;
    ecfg.num_layers = ENT_LM_LAYERS;
    ecfg.hidden_dim = ENT_LM_HIDDEN;
    ecfg.num_heads = ENT_LM_HEADS;
    ecfg.max_seq_len = cfg.seq_len;
    ecfg.rope_theta = 10000.0f;

    blt_entropy_lm* ent_lm = NULL;
    blt_entropy_lm_grad* ent_grad = NULL;
    if (uses_entropy_patcher(&cfg)) {
        ent_lm = blt_entropy_lm_create(model_arena, &ecfg);
        ent_grad = blt_entropy_lm_grad_create(model_arena, ent_lm);
        if (!ent_lm || !ent_grad) BLT_FATAL("main: entropy LM creation failed");
        init_entropy_lm(ent_lm);
    }

    train_ctx tc = {.model = model, .ent_lm = ent_lm, .cfg = &cfg};

    long start_step = 0;
    // ---- Data
    byte_buf train_buf, held_buf, held_c, held_h, manifest_buf;
    load_file(cfg.train_bin, &train_buf);
    load_file("data/manifest.json", &manifest_buf);
    load_file(cfg.heldout_bin, &held_buf);
    load_file(cfg.heldout_c_bin, &held_c);
    load_file(cfg.heldout_h_bin, &held_h);

    window_list train_wl, held_wl, held_c_wl, held_h_wl;
    windows_from_manifest(&manifest_buf, "train", cfg.seq_len, 0, &train_wl);
    if (train_wl.num == 0) BLT_FATAL("main: no training windows (seq_len too big?)");
    windows_from_manifest(&manifest_buf, "heldout", cfg.seq_len,
                          cfg.eval_windows, &held_wl);
    windows_from_stream(held_c.len, cfg.seq_len, cfg.eval_windows, &held_c_wl);
    windows_from_stream(held_h.len, cfg.seq_len, cfg.eval_windows, &held_h_wl);

    printf("[sweep] train windows=%zu eval windows: all=%zu c=%zu h=%zu\n",
           train_wl.num, held_wl.num, held_c_wl.num, held_h_wl.num);

    // ---- Training loop
    const size_t L = cfg.seq_len;
    size_t logits_shape[2] = {L, 256};
    size_t scalar_shape[1] = {1};
    double t_start = now_sec();
    double bytes_trained = 0.0;
    double samples_sum = 0.0;
    long n_samples = 0;

    double* step_samples = (double*)malloc(sizeof(double) *
        (total_steps - start_step > 0 ? (size_t)(total_steps - start_step) : 1));
    if (!step_samples) BLT_FATAL("main: OOM for step samples");

    // ---- Entropy LM warmup: pretrain the patcher's entropy model on its own
    // next-byte loss so dynamic patching is meaningful from the first main
    // step. Identical across all sweep configs (same seed + data order).
    if (ent_lm && cfg.ent_warmup_steps > 0 && start_step < cfg.ent_warmup_steps) {
        printf("[sweep] entropy LM warmup: %ld steps\n", cfg.ent_warmup_steps);
        fflush(stdout);
        double w_start = now_sec();
        double ent_sum = 0.0, ent_sq = 0.0, ent_max = 0.0;
        size_t ent_n = 0;
        for (long ws = start_step; ws < cfg.ent_warmup_steps; ws++) {
            if (g_stop_requested) {
                printf("[sweep] signal during warmup; exiting\n");
                return 130;
            }
            blt_arena_reset(scratch);
            const uint8_t* wb =
                train_buf.data + train_wl.items[ws % (long)train_wl.num].offset;
            size_t bshape[1] = {L};
            blt_tensor bin = blt_tensor_create(scratch, bshape, 1, BLT_DTYPE_UINT8);
            blt_tensor_copy_from_host(&bin, wb, L);
            size_t lshape[2] = {L, 256};
            blt_tensor elog = blt_tensor_create(scratch, lshape, 2, BLT_DTYPE_FP32);
            size_t sshape[1] = {1};
            blt_tensor eloss = blt_tensor_create(scratch, sshape, 1, BLT_DTYPE_FP32);
            blt_entropy_lm_forward(ent_lm, &bin, &elog, &eloss, scratch);

            // track entropy stats on natural-log scale
            blt_tensor pr = blt_tensor_create(scratch, lshape, 2, BLT_DTYPE_FP32);
            blt_softmax(&elog, &pr);
            size_t eshape[1] = {L};
            blt_tensor ev = blt_tensor_create(scratch, eshape, 1, BLT_DTYPE_FP32);
            blt_entropy_config ec = {.vocab_size = 256, .use_log2 = false};
            blt_compute_entropy(&pr, &ev, &ec);
            const float* evd = (const float*)ev.data;
            float* evd_host = (float*)malloc(L * sizeof(float));
            blt_tensor_copy_to_host(&ev, evd_host, L * sizeof(float));
            for (size_t i = 0; i < L; i++) {
                double e = evd_host[i];
                ent_sum += e; ent_sq += e * e;
                if (e > ent_max) ent_max = e;
                ent_n++;
            }
            free(evd_host);

            zero_entropy_scatter(ent_grad, ent_lm);
            blt_entropy_lm_backward(ent_lm, &bin, ent_grad, scratch);
            clip_entropy_grads(ent_lm, ent_grad);
            sgd_apply_entropy_lm(ent_lm, ent_grad, (float)cfg.lr);

            if ((ws + 1) % 100 == 0) {
                float loss_val = 0.0f;
                blt_tensor_copy_to_host(&eloss, &loss_val, sizeof(float));
                printf("[sweep] warmup %ld/%ld loss=%.4f (%.0fs)\n",
                       ws + 1, cfg.ent_warmup_steps,
                       loss_val, now_sec() - w_start);
                fflush(stdout);
            }
        }
        double mean = ent_n ? ent_sum / ent_n : 0.0;
        double var = ent_n ? ent_sq / ent_n - mean * mean : 0.0;
        printf("[sweep] warmup done in %.0fs; entropy stats: mean=%.3f sd=%.3f max=%.3f\n",
               now_sec() - w_start, mean, sqrt(var > 0 ? var : 0), ent_max);
        fflush(stdout);
    }

    long main_loop_start = (start_step > cfg.ent_warmup_steps) ? start_step : cfg.ent_warmup_steps;
    for (long step = main_loop_start; step < total_steps; step++) {
        if (g_stop_requested) {
            printf("\n[sweep] signal received at step %ld — exiting\n", step);
            return 130;
        }
        double elapsed = now_sec() - t_start;
        if (elapsed > (double)cfg.max_wall_sec) {
            printf("\n[sweep] wall-clock guard hit (%.0fs > %ld s) at step %ld\n",
                   elapsed, cfg.max_wall_sec, step);
            return 124;
        }

        double t_step = now_sec();
        blt_arena_reset(scratch);

        const uint8_t* wbytes =
            train_buf.data + train_wl.items[step % (long)train_wl.num].offset;
        size_t bytes_shape[1] = {L};
        blt_tensor bytes_in = blt_tensor_create(scratch, bytes_shape, 1, BLT_DTYPE_UINT8);
        blt_tensor_copy_from_host(&bytes_in, wbytes, L);

        blt_patch_info* patches = (blt_patch_info*)malloc(L * sizeof(blt_patch_info));
        size_t num_patches = make_patches(&tc, wbytes, L, &bytes_in, scratch, patches);

        blt_tensor logits = blt_tensor_create(scratch, logits_shape, 2, BLT_DTYPE_FP32);
        blt_tensor loss = blt_tensor_create(scratch, scalar_shape, 1, BLT_DTYPE_FP32);
        forward_model(&tc, &bytes_in, patches, num_patches, &logits, &loss, scratch);

        zero_scatter_grads(grad, model->encoder, &mc);
        size_t doc_boundaries[1] = {0};
        blt_model_backward(model, &bytes_in, patches, num_patches,
                           doc_boundaries, 1, grad, scratch);
        clip_model_grads(grad, model);
        sgd_apply_model(model, grad, (float)cfg.lr);

        if (ent_lm) {
            size_t ent_logits_shape[2] = {L, 256};
            blt_tensor ent_logits =
                blt_tensor_create(scratch, ent_logits_shape, 2, BLT_DTYPE_FP32);
            blt_tensor ent_loss =
                blt_tensor_create(scratch, scalar_shape, 1, BLT_DTYPE_FP32);
            blt_entropy_lm_forward(ent_lm, &bytes_in, &ent_logits, &ent_loss, scratch);
            zero_entropy_scatter(ent_grad, ent_lm);
            blt_entropy_lm_backward(ent_lm, &bytes_in, ent_grad, scratch);
            clip_entropy_grads(ent_lm, ent_grad);
            sgd_apply_entropy_lm(ent_lm, ent_grad, (float)cfg.lr);
        }
        free(patches);
        double dt = now_sec() - t_step;
        step_samples[n_samples++] = dt;
        samples_sum += dt;
        bytes_trained += (double)L;

        if ((step + 1) % cfg.eval_every == 0) {
            blt_arena_reset(scratch);
            eval_out eo;
            eval_stream(&tc, &held_buf, &held_wl, scratch, &eo);
            float loss_val = 0.0f;
            blt_tensor_copy_to_host(&loss, &loss_val, sizeof(float));
            printf("[sweep] step %ld/%ld train_loss=%.4f bpb=%.4f avg_patch=%.2f "
                   "step_ms=%.1f elapsed=%.0fs\n",
                   step + 1, total_steps, loss_val,
                   eo.bpb, eo.avg_patch_len, dt * 1e3, now_sec() - t_start);
            fflush(stdout);
        }
    }

    double wall = now_sec() - t_start;

    // ---- Final evaluation
    blt_arena_reset(scratch);
    // Per-domain BPB over capped windows from the raw domain streams.
    eval_out eo, eo_c, eo_h;
    eval_stream(&tc, &held_buf, &held_wl, scratch, &eo);
    long cap = cfg.eval_windows > 0 ? cfg.eval_windows : 0;
    if (cap > 0 && (long)held_c_wl.num > cap) held_c_wl.num = (size_t)cap;
    if (cap > 0 && (long)held_h_wl.num > cap) held_h_wl.num = (size_t)cap;
    eval_stream(&tc, &held_c, &held_c_wl, scratch, &eo_c);
    eval_stream(&tc, &held_h, &held_h_wl, scratch, &eo_h);

    printf("[sweep] FINAL tag=%s bpb=%.4f bpb_c=%.4f bpb_h=%.4f avg_patch=%.2f\n",
           cfg.tag, eo.bpb, eo_c.bpb, eo_h.bpb, eo.avg_patch_len);

    // ---- FLOPs/byte via tools/flops.c
    size_t heads = mc.encoder_config.num_heads;
    blt_flops_config fc = {0};
    fc.h_G = cfg.glob_embed_dim; fc.l_G = cfg.glob_num_layers;
    fc.n_ctx = L; fc.global_heads = mc.global_config.num_heads;
    fc.global_head_dim = cfg.glob_embed_dim / mc.global_config.num_heads;
    fc.d_ff_G = 4 * cfg.glob_embed_dim;
    fc.h_E = cfg.enc_embed_dim; fc.l_E = cfg.enc_num_layers; fc.w_E = 0;
    fc.enc_heads = heads; fc.enc_head_dim = cfg.enc_embed_dim / heads;
    fc.d_ff_E = 4 * cfg.enc_embed_dim;
    fc.h_D = cfg.dec_embed_dim; fc.l_D = cfg.dec_num_layers; fc.w_D = 0;
    fc.dec_heads = mc.decoder_config.num_heads;
    fc.dec_head_dim = cfg.dec_embed_dim / mc.decoder_config.num_heads;
    fc.d_ff_D = 4 * cfg.dec_embed_dim;
    size_t n_p = (size_t)(eo.avg_patch_len + 0.5);
    if (n_p == 0) n_p = 1;
    fc.n_p = n_p; fc.k = 1; fc.vocab_size = 256;

    // ---- Results line
    bench_result r;
    bench_result_init(&r, cfg.name, cfg.tag, cfg.phase);
    r.latency.n = n_samples;
    if (n_samples > 0) {
        bench_stats_compute(&r.latency, step_samples, (size_t)n_samples);
    }
    bench_result_add_metric(&r, "bpb", eo.bpb);
    bench_result_add_metric(&r, "bpb_c", eo_c.bpb);
    bench_result_add_metric(&r, "bpb_h", eo_h.bpb);
    bench_result_add_metric(&r, "avg_patch_len", eo.avg_patch_len);
    bench_result_add_metric(&r, "train_steps", (double)total_steps);
    bench_result_add_metric(&r, "wall_clock_sec", wall);
    if (eo.bpb > 0.0) {
        bench_result_add_metric(&r, "flops_per_byte",
                                blt_flops_per_byte(&fc));
    }
    bench_result_add_metric(&r, "throughput_bytes_per_sec",
                            bytes_trained / wall);

    if (bench_write_json(results_path, &r) != 0) {
        BLT_FATAL("main: failed to append results to %s", results_path);
    }
    printf("[sweep] wrote %s\n", results_path);

    free(step_samples);
    free(train_wl.items); free(held_wl.items);
    free(train_buf.data); free(manifest_buf.data);
    free(held_buf.data); free(held_c.data); free(held_h.data);
    blt_arena_destroy(scratch);
    blt_arena_destroy(model_arena);
    return 0;
}
