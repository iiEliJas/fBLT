#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "test_helpers.h"
#include "test_suite.h"

#include "blt/core/allocator.h"
#include "blt/core/backend.h"
#include "blt/models/local_decoder.h"
#include "blt/models/model.h"

//----------------------------------------------------------------------
// Helpers

static void fill_small_uniform(blt_tensor* t, float scale) {
    float* data = (float*)t->data;
    for (size_t i = 0; i < t->numel; i++) {
        float r = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
        data[i] = r * scale;
    }
}

static void fill_constant(blt_tensor* t, float value) {
    float* data = (float*)t->data;
    for (size_t i = 0; i < t->numel; i++) {
        data[i] = value;
    }
}

static int tensors_equal_exact(const blt_tensor* a, const blt_tensor* b) {
    TEST_ASSERT(a->numel == b->numel);
    const float* x = (const float*)a->data;
    const float* y = (const float*)b->data;
    for (size_t i = 0; i < a->numel; i++) {
        if (x[i] != y[i]) {
            fprintf(stderr, "  mismatch at %zu: %f vs %f\n", i, (double)x[i], (double)y[i]);
            TEST_ASSERT(0);
        }
    }
    return 1;
}

// Compares the first `rows` rows of two [*, V] logit tensors.
static int logit_rows_equal_exact(const blt_tensor* a, const blt_tensor* b, size_t rows, size_t V) {
    TEST_ASSERT(a->shape[1] == V && b->shape[1] == V);
    const float* x = (const float*)a->data;
    const float* y = (const float*)b->data;
    for (size_t i = 0; i < rows * V; i++) {
        if (x[i] != y[i]) {
            fprintf(stderr, "  row-block mismatch at flat idx %zu: %f vs %f\n", i, (double)x[i], (double)y[i]);
            TEST_ASSERT(0);
        }
    }
    return 1;
}

static void make_small_model_config(blt_model_config* config) {
    const size_t embed_dim = 16;
    const size_t hidden_dim = 32;
    const size_t max_seq_len = 64;

    memset(config, 0, sizeof(*config));

    config->encoder_config.embed_dim = embed_dim;
    config->encoder_config.patch_dim = 0;
    config->encoder_config.num_layers = 1;
    config->encoder_config.hidden_dim = hidden_dim;
    config->encoder_config.num_heads = 2;
    config->encoder_config.cross_attn_heads = 2;
    config->encoder_config.local_window = 0;
    config->encoder_config.cross_attn_all_layers = true;
    config->encoder_config.pool_type = BLT_POOL_MEAN;
    config->encoder_config.rope_theta = 10000.0f;
    config->encoder_config.max_seq_len = max_seq_len;
    config->encoder_config.ngram_config.ngram_sizes[0] = 3;
    config->encoder_config.ngram_config.ngram_sizes[1] = 4;
    config->encoder_config.ngram_config.num_ngram_sizes = 2;
    config->encoder_config.ngram_config.per_ngram_vocab = 64;
    config->encoder_config.ngram_config.hash_prime = 1000000007ULL;
    config->encoder_config.ngram_config.normalize = true;
    config->encoder_config.ngram_config.embed_dim = embed_dim;

    config->global_config.embed_dim = embed_dim;
    config->global_config.num_layers = 1;
    config->global_config.hidden_dim = hidden_dim;
    config->global_config.num_heads = 2;
    config->global_config.rope_theta = 10000.0f;
    config->global_config.max_seq_len = max_seq_len;

    config->decoder_config.embed_dim = embed_dim;
    config->decoder_config.patch_dim = 0;
    config->decoder_config.num_layers = 1;
    config->decoder_config.hidden_dim = hidden_dim;
    config->decoder_config.num_heads = 2;
    config->decoder_config.cross_attn_heads = 2;
    config->decoder_config.local_window = 0;
    config->decoder_config.cross_attn_all_layers = true;
    config->decoder_config.rope_theta = 10000.0f;
    config->decoder_config.max_seq_len = max_seq_len;
    config->decoder_config.vocab_size = 256;
}

static void init_layer_common(blt_local_layer_storage* l, float scale) {
    fill_constant(&l->norm1_weight, 1.0f);
    fill_small_uniform(&l->attn_qkv_w, scale);
    fill_small_uniform(&l->attn_proj_w, scale);
    fill_constant(&l->norm2_weight, 1.0f);
    fill_small_uniform(&l->ffn_up_w, scale);
    fill_small_uniform(&l->ffn_gate_w, scale);
    fill_small_uniform(&l->ffn_down_w, scale);
    fill_constant(&l->cross_norm_weight, 1.0f);
    fill_small_uniform(&l->cross_weight_q, scale);
    fill_small_uniform(&l->cross_weight_k, scale);
    fill_small_uniform(&l->cross_weight_v, scale);
    fill_small_uniform(&l->cross_weight_proj, scale);
}

static void random_init_model(blt_model* model, float scale) {
    fill_small_uniform(&model->encoder->byte_embedding_weight, scale);
    for (size_t i = 0; i < model->encoder->config.num_layers; i++) {
        init_layer_common(&model->encoder->layers[i], scale);
    }
    for (size_t i = 0; i < model->global->stack.num_layers; i++) {
        blt_tensor* w[] = {
            &model->global->stack.layer_storage[i].norm1_weight,
            &model->global->stack.layer_storage[i].norm2_weight,
        };
        for (size_t j = 0; j < 2; j++) fill_constant(w[j], 1.0f);
        fill_small_uniform(&model->global->stack.layer_storage[i].attn_qkv_w, scale);
        fill_small_uniform(&model->global->stack.layer_storage[i].attn_proj_w, scale);
        fill_small_uniform(&model->global->stack.layer_storage[i].ffn_up_w, scale);
        fill_small_uniform(&model->global->stack.layer_storage[i].ffn_gate_w, scale);
        fill_small_uniform(&model->global->stack.layer_storage[i].ffn_down_w, scale);
    }
    for (size_t i = 0; i < model->decoder->config.num_layers; i++) {
        init_layer_common(&model->decoder->layers[i], scale);
    }
    fill_small_uniform(&model->decoder->lm_head_weight, scale);
}

static size_t build_fixed_patches(size_t seq_len, size_t patch_len, blt_patch_info* patches_out) {
    size_t num_patches = 0;
    size_t start = 0;
    while (start < seq_len) {
        size_t len = (start + patch_len <= seq_len) ? patch_len : (seq_len - start);
        patches_out[num_patches].start_idx = start;
        patches_out[num_patches].length = len;
        patches_out[num_patches].peak_entropy = 0.0f;
        num_patches++;
        start += len;
    }
    return num_patches;
}

static blt_tensor make_bytes_tensor(blt_arena* arena, const uint8_t* bytes, size_t len) {
    size_t shape[1] = {len};
    blt_tensor t = blt_tensor_create(arena, shape, 1, BLT_DTYPE_UINT8);
    memcpy(t.data, bytes, len);
    return t;
}


//----------------------------------------------------------------------
// Test 1: stage-split equivalence.
//
// blt_model_forward must remain bit-identical to calling
// blt_model_encode followed by blt_model_decode -- this is the contract
// that lets inference controllers freeze latents between rounds without
// perturbing the canonical full-forward numerics.

int run_model_stage_split_equivalence(void) {
    srand(11);

    blt_arena* model_arena = blt_arena_create(1024 * 1024, BLT_BACKEND_CPU);
    blt_arena* scratch = blt_arena_create(8 * 1024 * 1024, BLT_BACKEND_CPU);
    TEST_ASSERT(model_arena && scratch);

    blt_model_config config;
    make_small_model_config(&config);
    blt_model* model = blt_model_create(model_arena, &config);
    TEST_ASSERT(model != NULL);
    random_init_model(model, 0.1f);

    // Scenario A: single document, odd trailing partial patch
    {
        const char* text = "int x=1;\nret";   // 12 bytes
        size_t seq_len = strlen(text);
        blt_tensor bytes_in = make_bytes_tensor(scratch, (const uint8_t*)text, seq_len);

        blt_patch_info patches[16];
        size_t num_patches = build_fixed_patches(seq_len, 5, patches);

        size_t logits_shape[2] = {seq_len, config.decoder_config.vocab_size};
        size_t scalar_shape[1] = {1};

        // full forward
        blt_tensor logits_f = blt_tensor_create(scratch, logits_shape, 2, BLT_DTYPE_FP32);
        blt_tensor loss_f = blt_tensor_create(scratch, scalar_shape, 1, BLT_DTYPE_FP32);
        blt_model_forward(model, &bytes_in, patches, num_patches, NULL, 0,
            &logits_f, &loss_f, scratch);

        // staged
        blt_model_enc_out enc;
        blt_model_encode(model, &bytes_in, patches, num_patches, NULL, 0, &enc, scratch);
        TEST_ASSERT(enc.patch_doc_boundaries == NULL);
        TEST_ASSERT(enc.patch_out.shape[0] == num_patches);
        TEST_ASSERT(enc.byte_hidden_out.shape[0] == seq_len);

        blt_tensor logits_d = blt_tensor_create(scratch, logits_shape, 2, BLT_DTYPE_FP32);
        blt_tensor loss_d = blt_tensor_create(scratch, scalar_shape, 1, BLT_DTYPE_FP32);
        blt_model_decode(model, &enc, patches, num_patches, &bytes_in, NULL, 0,
            NULL, &logits_d, &loss_d, scratch);

        tensors_equal_exact(&logits_f, &logits_d);
        tensors_equal_exact(&loss_f, &loss_d);
    }

    // Scenario B: two documents whose boundary aligns to a patch boundary --
    // exercises the byte->patch doc-boundary remap inside the encode stage
    {
        const char* text = "abcdEFGHijkl";   // 12 bytes
        size_t seq_len = strlen(text);
        blt_tensor bytes_in = make_bytes_tensor(scratch, (const uint8_t*)text, seq_len);

        blt_patch_info patches[16];
        size_t num_patches = build_fixed_patches(seq_len, 4, patches);
        size_t doc_boundaries[2] = {0, 8};   // patch indices 0 and 2

        size_t logits_shape[2] = {seq_len, config.decoder_config.vocab_size};
        size_t scalar_shape[1] = {1};

        blt_tensor logits_f = blt_tensor_create(scratch, logits_shape, 2, BLT_DTYPE_FP32);
        blt_tensor loss_f = blt_tensor_create(scratch, scalar_shape, 1, BLT_DTYPE_FP32);
        blt_model_forward(model, &bytes_in, patches, num_patches, doc_boundaries, 2,
            &logits_f, &loss_f, scratch);

        blt_model_enc_out enc;
        blt_model_encode(model, &bytes_in, patches, num_patches, doc_boundaries, 2, &enc, scratch);
        TEST_ASSERT(enc.patch_doc_boundaries != NULL);
        TEST_ASSERT(enc.patch_doc_boundaries[0] == 0 && enc.patch_doc_boundaries[1] == 2);

        blt_tensor logits_d = blt_tensor_create(scratch, logits_shape, 2, BLT_DTYPE_FP32);
        blt_tensor loss_d = blt_tensor_create(scratch, scalar_shape, 1, BLT_DTYPE_FP32);
        blt_model_decode(model, &enc, patches, num_patches, &bytes_in, doc_boundaries, 2,
            NULL, &logits_d, &loss_d, scratch);

        tensors_equal_exact(&logits_f, &logits_d);
        tensors_equal_exact(&loss_f, &loss_d);
    }

    blt_arena_destroy(scratch);
    blt_arena_destroy(model_arena);
    return 1;
}


//----------------------------------------------------------------------
// Test 2: logits-only decode (nullable loss / nullable bytes).
//
// The inference fast path skips the shifted cross-entropy entirely; the
// logits must be unchanged by that skip.

int run_model_decode_nullable_loss(void) {
    srand(12);

    blt_arena* model_arena = blt_arena_create(1024 * 1024, BLT_BACKEND_CPU);
    blt_arena* scratch = blt_arena_create(8 * 1024 * 1024, BLT_BACKEND_CPU);
    TEST_ASSERT(model_arena && scratch);

    blt_model_config config;
    make_small_model_config(&config);
    blt_model* model = blt_model_create(model_arena, &config);
    TEST_ASSERT(model != NULL);
    random_init_model(model, 0.1f);

    const char* text = "return 0;\n";
    size_t seq_len = strlen(text);
    blt_tensor bytes_in = make_bytes_tensor(scratch, (const uint8_t*)text, seq_len);

    blt_patch_info patches[16];
    size_t num_patches = build_fixed_patches(seq_len, 3, patches);

    size_t logits_shape[2] = {seq_len, config.decoder_config.vocab_size};
    size_t scalar_shape[1] = {1};

    blt_tensor logits_f = blt_tensor_create(scratch, logits_shape, 2, BLT_DTYPE_FP32);
    blt_tensor loss = blt_tensor_create(scratch, scalar_shape, 1, BLT_DTYPE_FP32);
    blt_model_forward(model, &bytes_in, patches, num_patches, NULL, 0, &logits_f, &loss, scratch);

    blt_model_enc_out enc;
    blt_model_encode(model, &bytes_in, patches, num_patches, NULL, 0, &enc, scratch);

    blt_tensor logits_only = blt_tensor_create(scratch, logits_shape, 2, BLT_DTYPE_FP32);
    blt_model_decode(model, &enc, patches, num_patches, NULL, NULL, 0,
        NULL, &logits_only, NULL, scratch);

    tensors_equal_exact(&logits_f, &logits_only);

    blt_arena_destroy(scratch);
    blt_arena_destroy(model_arena);
    return 1;
}


//----------------------------------------------------------------------
// Test 3: D_0 policies for rows without encoder h_final states.
//
// Core property (BLT-S drafting correctness rests on it): adding extra
// rows beyond num_hfinal_rows must NOT change the logits of the covered
// prefix rows (causality), regardless of policy or garbage in the extra
// input rows. LEARNED mode must additionally consult the d0_embed_weight
// table for extra rows and stay deterministic.

typedef struct {
    size_t embed_dim;
    size_t patch_dim;
} dec_cfg_dims;

static void make_decoder_config(blt_local_decoder_config* cfg, const dec_cfg_dims* dims) {
    memset(cfg, 0, sizeof(*cfg));
    cfg->embed_dim = dims->embed_dim;
    cfg->patch_dim = dims->patch_dim;   // 0 = no split
    cfg->num_layers = 1;
    cfg->hidden_dim = 32;
    cfg->num_heads = 2;
    cfg->cross_attn_heads = 2;
    cfg->local_window = 0;
    cfg->cross_attn_all_layers = true;
    cfg->rope_theta = 10000.0f;
    cfg->max_seq_len = 64;
    cfg->vocab_size = 256;
}

// Builds a decoder with deterministic small-random weights.
static blt_local_decoder* make_random_decoder(blt_arena* arena, const dec_cfg_dims* dims, float scale) {
    blt_local_decoder_config cfg;
    make_decoder_config(&cfg, dims);
    blt_local_decoder* dec = blt_local_decoder_create(arena, &cfg);
    TEST_ASSERT(dec != NULL);
    init_layer_common(&dec->layers[0], scale);
    fill_small_uniform(&dec->lm_head_weight, scale);
    // deterministic non-zero D_0 table (rand-independent)
    float* tbl = (float*)dec->d0_embed_weight.data;
    for (size_t i = 0; i < dec->d0_embed_weight.numel; i++) {
        tbl[i] = ((float)((i * 7) % 13) - 6.0f) * 0.05f;
    }
    return dec;
}

static int run_d0_mode_case(blt_d0_mode mode, size_t patch_dim) {
    srand(13);

    blt_arena* arena = blt_arena_create(8 * 1024 * 1024, BLT_BACKEND_CPU);
    TEST_ASSERT(arena != NULL);

    dec_cfg_dims dims = { .embed_dim = 16, .patch_dim = patch_dim };
    size_t E = dims.embed_dim;
    size_t pdim = (dims.patch_dim == 0) ? E : dims.patch_dim;

    blt_local_decoder* dec = make_random_decoder(arena, &dims, 0.1f);

    // Prefix: 8 h-final rows tiled by 2 patches of 4; plus 4 extra rows.
    enum { H = 8, EXTRA = 4 };
    const size_t S = H + EXTRA;

    blt_patch_info patches[8];
    size_t num_patches = build_fixed_patches(H, 4, patches);

    size_t patch_in_shape[2] = {num_patches, pdim};
    blt_tensor patch_in = blt_tensor_create(arena, patch_in_shape, 2, BLT_DTYPE_FP32);
    fill_small_uniform(&patch_in, 0.5f);

    size_t hidden_shape_prefix[2] = {H, E};
    blt_tensor h_prefix = blt_tensor_create(arena, hidden_shape_prefix, 2, BLT_DTYPE_FP32);
    fill_small_uniform(&h_prefix, 0.5f);

    // Reference logits: prefix alone, legacy semantics (no opts), same
    // frozen patch latents as the extended run below
    size_t ref_shape[2] = {H, 256};
    blt_tensor ref_logits = blt_tensor_create(arena, ref_shape, 2, BLT_DTYPE_FP32);
    blt_local_decoder_forward_ext(dec, &h_prefix, &patch_in, patches, num_patches,
        NULL, NULL, 0, NULL, &ref_logits, NULL, arena);

    // Full input: prefix rows verbatim + poisoned extra rows (must never
    // influence anything)
    size_t hidden_shape[2] = {S, E};
    blt_tensor h_full = blt_tensor_create(arena, hidden_shape, 2, BLT_DTYPE_FP32);
    memcpy(h_full.data, h_prefix.data, H * E * sizeof(float));
    {
        float* tail = (float*)h_full.data + H * E;
        for (size_t i = 0; i < EXTRA * E; i++) {
            tail[i] = 12345.6789f;   // poison
        }
    }

    uint32_t tokens[EXTRA] = {3, 42, 255, 256};
    blt_local_decoder_d0_opts opts = {
        .d0_mode = mode,
        .num_hfinal_rows = H,
        .d0_extra_tokens = tokens,
    };

    size_t out_shape[2] = {S, 256};
    blt_tensor out_logits = blt_tensor_create(arena, out_shape, 2, BLT_DTYPE_FP32);
    blt_local_decoder_forward_ext(dec, &h_full, &patch_in, patches, num_patches,
        NULL, NULL, 0, &opts, &out_logits, NULL, arena);

    // Prefix rows must be bit-identical to the prefix-only run
    logit_rows_equal_exact(&out_logits, &ref_logits, H, 256);

    // LEARNED: extra rows must depend on the table (differ from zeros-run)
    // and be deterministic across reruns.
    if (mode == BLT_D0_LEARNED) {
        // deterministic rerun
        blt_tensor rerun = blt_tensor_create(arena, out_shape, 2, BLT_DTYPE_FP32);
        blt_local_decoder_forward_ext(dec, &h_full, &patch_in, patches, num_patches,
            NULL, NULL, 0, &opts, &rerun, NULL, arena);
        tensors_equal_exact(&out_logits, &rerun);

        // contrast run: same everything, ZEROS policy -> different extra rows
        blt_local_decoder_d0_opts zopts = {
            .d0_mode = BLT_D0_ZEROS,
            .num_hfinal_rows = H,
            .d0_extra_tokens = NULL,
        };
        blt_tensor zeros_logits = blt_tensor_create(arena, out_shape, 2, BLT_DTYPE_FP32);
        blt_local_decoder_forward_ext(dec, &h_full, &patch_in, patches, num_patches,
            NULL, NULL, 0, &zopts, &zeros_logits, NULL, arena);

        const float* a = (const float*)out_logits.data;
        const float* b = (const float*)zeros_logits.data;
        size_t diffs = 0;
        for (size_t i = H * 256; i < S * 256; i++) {
            if (a[i] != b[i]) diffs++;
        }
        TEST_ASSERT(diffs > 0);
    }

    blt_arena_destroy(arena);
    return 1;
}

int run_decoder_d0_modes(void) {
    // no split
    TEST_ASSERT(run_d0_mode_case(BLT_D0_ZEROS, 0));
    TEST_ASSERT(run_d0_mode_case(BLT_D0_LEARNED, 0));
    // k-split (patch_dim = 2 * embed_dim)
    TEST_ASSERT(run_d0_mode_case(BLT_D0_ZEROS, 32));
    TEST_ASSERT(run_d0_mode_case(BLT_D0_LEARNED, 32));
    return 1;
}
