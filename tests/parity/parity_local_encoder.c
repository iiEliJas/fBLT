#include "test_helpers.h"
#include "test_suite.h"
#include "models/local_encoder.h"

static void build_reference_patches(blt_patch_info *patches, size_t *num_patches) {
    size_t starts[] = {0, 4, 9, 13, 18};
    size_t n = sizeof(starts) / sizeof(starts[0]);
    for (size_t i = 0; i < n; i++) {
        patches[i].start_idx = starts[i];
        patches[i].length = (i + 1 < n) ? (starts[i + 1] - starts[i]) : (24 - starts[i]);
        patches[i].peak_entropy = 0.0f;
    }
    *num_patches = n;
}

// Loader for UINT8 Tensors (Used for bytes array)
static int load_binary_tensor_uint8(const char *path, blt_arena *arena, blt_tensor *out_tensor) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "failed to open %s\n", path);
        return 0;
    }

    uint32_t ndim = 0;
    if (fread(&ndim, sizeof(ndim), 1, fp) != 1) {
        fclose(fp);
        return 0;
    }

    size_t shape[BLT_MAX_NDIM] = {0};
    for (uint32_t i = 0; i < ndim; ++i) {
        uint32_t dim = 0;
        if (fread(&dim, sizeof(dim), 1, fp) != 1) {
            fclose(fp);
            return 0;
        }
        shape[i] = dim;
    }

    size_t numel = 1;
    for (uint32_t i = 0; i < ndim; ++i) {
        numel *= shape[i];
    }

    *out_tensor = blt_tensor_create(arena, shape, ndim, BLT_DTYPE_UINT8);
    if (!out_tensor->data) {
        fclose(fp);
        return 0;
    }

    uint8_t *data = (uint8_t *)out_tensor->data;
    if (fread(data, 1, numel, fp) != numel) {
        fclose(fp);
        return 0;
    }

    fclose(fp);
    return 1;
}

static int load_reference_weights(blt_local_encoder *model, const char *dir, blt_arena *arena) {
    char path[512];
    const blt_local_encoder_config *cfg = &model->config;

    snprintf(path, sizeof(path), "%s/byte_embedding_weight.bin", dir);
    TEST_ASSERT(blt_test_load_binary_tensor(path, arena, &model->byte_embedding_weight));

    for (size_t t = 0; t < model->ngram_weights.num_tables; t++) {
        snprintf(path, sizeof(path), "%s/ngram_table_%zu.bin", dir, t);
        TEST_ASSERT(blt_test_load_binary_tensor(path, arena, &model->ngram_weights.tables[t]));
    }

    for (size_t l = 0; l < cfg->num_layers; l++) {
        blt_local_encoder_layer_storage *ls = &model->layers[l];

        snprintf(path, sizeof(path), "%s/layer_%zu_norm1_weight.bin", dir, l);
        TEST_ASSERT(blt_test_load_binary_tensor(path, arena, &ls->norm1_weight));

        snprintf(path, sizeof(path), "%s/layer_%zu_attn_qkv_w.bin", dir, l);
        TEST_ASSERT(blt_test_load_binary_tensor(path, arena, &ls->attn_qkv_w));

        snprintf(path, sizeof(path), "%s/layer_%zu_attn_proj_w.bin", dir, l);
        TEST_ASSERT(blt_test_load_binary_tensor(path, arena, &ls->attn_proj_w));

        snprintf(path, sizeof(path), "%s/layer_%zu_norm2_weight.bin", dir, l);
        TEST_ASSERT(blt_test_load_binary_tensor(path, arena, &ls->norm2_weight));

        snprintf(path, sizeof(path), "%s/layer_%zu_ffn_up_w.bin", dir, l);
        TEST_ASSERT(blt_test_load_binary_tensor(path, arena, &ls->ffn_up_w));

        snprintf(path, sizeof(path), "%s/layer_%zu_ffn_gate_w.bin", dir, l);
        TEST_ASSERT(blt_test_load_binary_tensor(path, arena, &ls->ffn_gate_w));

        snprintf(path, sizeof(path), "%s/layer_%zu_ffn_down_w.bin", dir, l);
        TEST_ASSERT(blt_test_load_binary_tensor(path, arena, &ls->ffn_down_w));

        snprintf(path, sizeof(path), "%s/layer_%zu_cross_norm_weight.bin", dir, l);
        TEST_ASSERT(blt_test_load_binary_tensor(path, arena, &ls->cross_norm_weight));

        snprintf(path, sizeof(path), "%s/layer_%zu_cross_weight_q.bin", dir, l);
        TEST_ASSERT(blt_test_load_binary_tensor(path, arena, &ls->cross_weight_q));

        snprintf(path, sizeof(path), "%s/layer_%zu_cross_weight_k.bin", dir, l);
        TEST_ASSERT(blt_test_load_binary_tensor(path, arena, &ls->cross_weight_k));

        snprintf(path, sizeof(path), "%s/layer_%zu_cross_weight_v.bin", dir, l);
        TEST_ASSERT(blt_test_load_binary_tensor(path, arena, &ls->cross_weight_v));

        snprintf(path, sizeof(path), "%s/layer_%zu_cross_weight_proj.bin", dir, l);
        TEST_ASSERT(blt_test_load_binary_tensor(path, arena, &ls->cross_weight_proj));
    }
    return 1;
}

static int run_local_encoder_parity_case(bool cross_attn_all_layers) {
    blt_arena *arena = blt_arena_create(8 * 1024 * 1024, BLT_BACKEND_CPU);
    if (!arena) return 0;

    const char *suffix = cross_attn_all_layers ? "all_layers" : "final_layer";
    char path_bytes[256], path_patch[256], path_hidden[256], weights_dir[256];
    snprintf(path_bytes, sizeof(path_bytes), "data/local_encoder_bytes_in.bin");
    snprintf(path_patch, sizeof(path_patch), "data/local_encoder_patch_out_%s.bin", suffix);
    snprintf(path_hidden, sizeof(path_hidden), "data/local_encoder_byte_hidden_out_%s.bin", suffix);
    snprintf(weights_dir, sizeof(weights_dir), "data/local_encoder_weights_%s", suffix);

    blt_tensor bytes_in = {0};
    blt_tensor expected_patch = {0};
    blt_tensor expected_hidden = {0};
    TEST_ASSERT(load_binary_tensor_uint8(path_bytes, arena, &bytes_in));
    TEST_ASSERT(load_binary_tensor(path_patch, arena, &expected_patch));
    TEST_ASSERT(load_binary_tensor(path_hidden, arena, &expected_hidden));

    blt_patch_info patches[8];
    size_t num_patches = 0;
    build_reference_patches(patches, &num_patches);

    blt_local_encoder_config cfg = {0};
    cfg.embed_dim = 32;
    cfg.num_layers = 3;
    cfg.hidden_dim = 64;
    cfg.num_heads = 4;
    cfg.cross_attn_heads = 4;
    cfg.local_window = 8;
    cfg.cross_attn_all_layers = cross_attn_all_layers;
    cfg.pool_type = BLT_POOL_MEAN;
    cfg.rope_theta = 500000.0f;
    cfg.max_seq_len = 64;
    cfg.ngram_config.ngram_sizes[0] = 3;
    cfg.ngram_config.ngram_sizes[1] = 4;
    cfg.ngram_config.ngram_sizes[2] = 5;
    cfg.ngram_config.num_ngram_sizes = 3;
    cfg.ngram_config.per_ngram_vocab = 512;
    cfg.ngram_config.hash_prime = 1000000007ULL;
    cfg.ngram_config.normalize = true;
    cfg.ngram_config.embed_dim = cfg.embed_dim;

    blt_local_encoder *model = blt_local_encoder_create(arena, &cfg);
    TEST_ASSERT(model != NULL);

    load_reference_weights(model, weights_dir, arena);

    blt_tensor patch_out = blt_tensor_create(arena, expected_patch.shape, expected_patch.ndim, BLT_DTYPE_FP32);
    blt_tensor byte_hidden_out = blt_tensor_create(arena, expected_hidden.shape, expected_hidden.ndim, BLT_DTYPE_FP32);

    blt_local_encoder_forward(model, &bytes_in, patches, num_patches, NULL, 0, &patch_out, &byte_hidden_out, arena);

    TEST_ASSERT_CLOSE(&patch_out, &expected_patch, 1e-4f);
    TEST_ASSERT_CLOSE(&byte_hidden_out, &expected_hidden, 1e-4f);

    blt_arena_destroy(arena);
    return 1;
}

int run_local_encoder_parity(void) {
    return run_local_encoder_parity_case(true) && run_local_encoder_parity_case(false);
    ;
}
