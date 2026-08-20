#include "test_helpers.h"
#include "test_suite.h"
#include "blt/models/global_transformer.h"



static int load_global_transformer_reference_weights(blt_global_transformer* model, const char* dir, blt_arena* arena) {
    char path[512];
    size_t num_layers = model->stack.num_layers;

    for (size_t l = 0; l < num_layers; l++) {
        blt_transformer_layer_storage* ls = &model->stack.layer_storage[l];

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
    }
    return 1;
}



int run_global_transformer_parity(void) {
    blt_arena* arena = blt_arena_create(16 * 1024 * 1024, BLT_BACKEND_CPU);
    if (!arena) return 0;

    blt_tensor patch_in = {0};
    blt_tensor expected_patch_out = {0};
    blt_tensor grad_patch_out_seed = {0};
    blt_tensor expected_grad_patch_in = {0};

    TEST_ASSERT(load_binary_tensor("data/global_transformer_patch_in.bin", arena, &patch_in));
    TEST_ASSERT(load_binary_tensor("data/global_transformer_patch_out.bin", arena, &expected_patch_out));
    TEST_ASSERT(load_binary_tensor("data/global_transformer_grad_patch_out.bin", arena, &grad_patch_out_seed));
    TEST_ASSERT(load_binary_tensor("data/global_transformer_grad_patch_in.bin", arena, &expected_grad_patch_in));

    blt_global_transformer_config cfg = {0};
    cfg.num_layers = 3;
    cfg.embed_dim = 32;
    cfg.hidden_dim = 64;
    cfg.num_heads = 4;
    cfg.max_seq_len = 64;
    cfg.rope_theta = 500000.0f;

    blt_global_transformer* model = blt_global_transformer_create(arena, &cfg);
    TEST_ASSERT(model != NULL);
    TEST_ASSERT(load_global_transformer_reference_weights(model, "data/global_transformer_weights", arena));

    blt_global_transformer_grad* grad = blt_global_transformer_grad_create(arena, model);
    TEST_ASSERT(grad != NULL);

    size_t doc_boundaries[] = { 0 };   // single document spanning every patch
    size_t num_docs = 1;

    blt_tensor patch_out = blt_tensor_create(arena, expected_patch_out.shape, expected_patch_out.ndim, BLT_DTYPE_FP32);
    blt_global_transformer_forward(model, &patch_in, doc_boundaries, num_docs, &patch_out, arena);
    TEST_ASSERT_CLOSE(&patch_out, &expected_patch_out, 1e-4f);

    blt_tensor grad_patch_in = blt_tensor_create(arena, patch_in.shape, patch_in.ndim, BLT_DTYPE_FP32);
    blt_global_transformer_backward(model, &patch_in, doc_boundaries, num_docs,
                                     &grad_patch_out_seed, &grad_patch_in, grad, arena);
    TEST_ASSERT_CLOSE(&grad_patch_in, &expected_grad_patch_in, 1e-4f);

    // per-layer weight gradients
    char path[512];
    for (size_t l = 0; l < cfg.num_layers; l++) {
        const blt_transformer_layer_grad* lg = &grad->stack_grad->layer_grads[l];
        blt_tensor expected = {0};

        snprintf(path, sizeof(path), "data/global_transformer_weights/grad_layer_%zu_norm1_weight.bin", l);
        TEST_ASSERT(load_binary_tensor(path, arena, &expected));
        TEST_ASSERT_CLOSE(&lg->norm1_weight, &expected, 1e-4f);

        snprintf(path, sizeof(path), "data/global_transformer_weights/grad_layer_%zu_attn_qkv_w.bin", l);
        TEST_ASSERT(load_binary_tensor(path, arena, &expected));
        TEST_ASSERT_CLOSE(&lg->attn_qkv_w, &expected, 1e-4f);

        snprintf(path, sizeof(path), "data/global_transformer_weights/grad_layer_%zu_attn_proj_w.bin", l);
        TEST_ASSERT(load_binary_tensor(path, arena, &expected));
        TEST_ASSERT_CLOSE(&lg->attn_proj_w, &expected, 1e-4f);

        snprintf(path, sizeof(path), "data/global_transformer_weights/grad_layer_%zu_norm2_weight.bin", l);
        TEST_ASSERT(load_binary_tensor(path, arena, &expected));
        TEST_ASSERT_CLOSE(&lg->norm2_weight, &expected, 1e-4f);

        snprintf(path, sizeof(path), "data/global_transformer_weights/grad_layer_%zu_ffn_up_w.bin", l);
        TEST_ASSERT(load_binary_tensor(path, arena, &expected));
        TEST_ASSERT_CLOSE(&lg->ffn_up_w, &expected, 1e-4f);

        snprintf(path, sizeof(path), "data/global_transformer_weights/grad_layer_%zu_ffn_gate_w.bin", l);
        TEST_ASSERT(load_binary_tensor(path, arena, &expected));
        TEST_ASSERT_CLOSE(&lg->ffn_gate_w, &expected, 1e-4f);

        snprintf(path, sizeof(path), "data/global_transformer_weights/grad_layer_%zu_ffn_down_w.bin", l);
        TEST_ASSERT(load_binary_tensor(path, arena, &expected));
        TEST_ASSERT_CLOSE(&lg->ffn_down_w, &expected, 1e-4f);
    }

    blt_arena_destroy(arena);
    return 1;
}