#include "test_helpers.h"
#include "test_suite.h"
#include "models/local_decoder.h"

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
    for (uint32_t i = 0; i < ndim; ++i) numel *= shape[i];

    *out_tensor = blt_tensor_create(arena, shape, ndim, BLT_DTYPE_UINT8);
    if (!out_tensor->data) {
        fclose(fp);
        return 0;
    }

    if (fread(out_tensor->data, 1, numel, fp) != numel) {
        fclose(fp);
        return 0;
    }
    fclose(fp);
    return 1;
}

static void build_decoder_parity_patches(blt_patch_info *patches, size_t *num_patches) {
    size_t starts[] = {0, 4, 9, 13, 18};
    size_t n = sizeof(starts) / sizeof(starts[0]);
    for (size_t i = 0; i < n; i++) {
        patches[i].start_idx = starts[i];
        patches[i].length = (i + 1 < n) ? (starts[i + 1] - starts[i]) : (24 - starts[i]);
        patches[i].peak_entropy = 0.0f;
    }
    *num_patches = n;
}

static int load_decoder_reference_weights(blt_local_decoder *model, const char *dir, blt_arena *arena) {
    char path[512];
    const blt_local_decoder_config *cfg = &model->config;

    for (size_t l = 0; l < cfg->num_layers; l++) {
        blt_local_decoder_layer_storage *ls = &model->layers[l];

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

    snprintf(path, sizeof(path), "%s/lm_head_weight.bin", dir);
    TEST_ASSERT(blt_test_load_binary_tensor(path, arena, &model->lm_head_weight));
    return 1;
}

static int run_local_decoder_parity_case(bool cross_attn_all_layers) {
    blt_arena *arena = blt_arena_create(16 * 1024 * 1024, BLT_BACKEND_CPU);
    if (!arena) return 0;

    const char *suffix = cross_attn_all_layers ? "all_layers" : "final_layer";
    char path_hidden_in[256], path_patch_in[256], path_bytes[256];
    char path_logits[256], path_loss[256];
    char path_grad_hidden[256], path_grad_patch[256], weights_dir[256];

    snprintf(path_hidden_in, sizeof(path_hidden_in), "%s/local_decoder_byte_hidden_in.bin", g_test_data_dir);
    snprintf(path_patch_in, sizeof(path_patch_in), "%s/local_decoder_patch_in.bin", g_test_data_dir);
    snprintf(path_bytes, sizeof(path_bytes), "%s/local_decoder_bytes_in.bin", g_test_data_dir);
    snprintf(path_logits, sizeof(path_logits), "%s/local_decoder_logits_out_%s.bin", g_test_data_dir, suffix);
    snprintf(path_loss, sizeof(path_loss), "%s/local_decoder_loss_out_%s.bin", g_test_data_dir, suffix);
    snprintf(path_grad_hidden, sizeof(path_grad_hidden), "%s/local_decoder_grad_byte_hidden_in_%s.bin", g_test_data_dir,
             suffix);
    snprintf(path_grad_patch, sizeof(path_grad_patch), "%s/local_decoder_grad_patch_in_%s.bin", g_test_data_dir,
             suffix);
    snprintf(weights_dir, sizeof(weights_dir), "%s/local_decoder_weights_%s", g_test_data_dir, suffix);

    blt_tensor byte_hidden_in = {0};
    blt_tensor patch_in = {0};
    blt_tensor bytes_in = {0};
    blt_tensor expected_logits = {0};
    blt_tensor expected_loss = {0};
    blt_tensor expected_grad_hidden = {0};
    blt_tensor expected_grad_patch = {0};

    TEST_ASSERT(load_binary_tensor(path_hidden_in, arena, &byte_hidden_in));
    TEST_ASSERT(load_binary_tensor(path_patch_in, arena, &patch_in));
    TEST_ASSERT(load_binary_tensor_uint8(path_bytes, arena, &bytes_in));
    TEST_ASSERT(load_binary_tensor(path_logits, arena, &expected_logits));
    TEST_ASSERT(load_binary_tensor(path_loss, arena, &expected_loss));
    TEST_ASSERT(load_binary_tensor(path_grad_hidden, arena, &expected_grad_hidden));
    TEST_ASSERT(load_binary_tensor(path_grad_patch, arena, &expected_grad_patch));

    blt_patch_info patches[8];
    size_t num_patches = 0;
    build_decoder_parity_patches(patches, &num_patches);

    blt_local_decoder_config cfg = {0};
    cfg.embed_dim = 32;
    cfg.num_layers = 3;
    cfg.hidden_dim = 64;
    cfg.num_heads = 4;
    cfg.cross_attn_heads = 4;
    cfg.local_window = 8;
    cfg.cross_attn_all_layers = cross_attn_all_layers;
    cfg.rope_theta = 500000.0f;
    cfg.max_seq_len = 64;
    cfg.vocab_size = 256;

    blt_local_decoder *model = blt_local_decoder_create(arena, &cfg);
    TEST_ASSERT(model != NULL);
    TEST_ASSERT(load_decoder_reference_weights(model, weights_dir, arena));

    blt_local_decoder_grad *grad = blt_local_decoder_grad_create(arena, model);
    TEST_ASSERT(grad != NULL);

    blt_tensor logits_out = blt_tensor_create(arena, expected_logits.shape, expected_logits.ndim, BLT_DTYPE_FP32);

    size_t loss_shape[1] = {1};
    blt_tensor loss_out = blt_tensor_create(arena, loss_shape, 1, BLT_DTYPE_FP32);

    blt_local_decoder_forward(model, &byte_hidden_in, &patch_in, patches, num_patches, &bytes_in, NULL, NULL, 0,
                              &logits_out, &loss_out, arena);

    TEST_ASSERT_CLOSE(&logits_out, &expected_logits, 1e-3f);
    TEST_ASSERT_CLOSE(&loss_out, &expected_loss, 1e-3f);

    blt_tensor grad_byte_hidden_in =
        blt_tensor_create(arena, byte_hidden_in.shape, byte_hidden_in.ndim, BLT_DTYPE_FP32);
    blt_tensor grad_patch_in = blt_tensor_create(arena, patch_in.shape, patch_in.ndim, BLT_DTYPE_FP32);

    blt_local_decoder_backward(model, &byte_hidden_in, &patch_in, patches, num_patches, &bytes_in, NULL, NULL, 0,
                               &grad_byte_hidden_in, &grad_patch_in, grad, arena);

    TEST_ASSERT_CLOSE(&grad_byte_hidden_in, &expected_grad_hidden, 1e-3f);
    TEST_ASSERT_CLOSE(&grad_patch_in, &expected_grad_patch, 1e-3f);

    blt_arena_destroy(arena);
    return 1;
}

int run_local_decoder_parity(void) { return run_local_decoder_parity_case(false); }