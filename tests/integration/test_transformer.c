#include "test_helpers.h"
#include "test_suite.h"

#include "blt/core/allocator.h"
#include "blt/core/tensor.h"
#include "blt/models/attention.h"
#include "blt/models/transformer.h"

#include <stdio.h>

// Absolute tolerance
#define PHASE2_ATOL 1e-4f

#define PHASE2_ARENA_BYTES (16 * 1024 * 1024)



// -----------------------------------------------------------------------
// Attention parity test

int run_phase2_attention_parity_test(void) {
    blt_arena* arena = blt_arena_create(PHASE2_ARENA_BYTES, BLT_BACKEND_CPU);
    if (!arena) {
        return 0;
    }

    blt_tensor input = {0};
    blt_tensor qkv_w = {0};
    blt_tensor proj_w = {0};
    blt_tensor expected_out = {0};

    int ok = 1;
    ok &= load_binary_tensor("data/phase2_attn_input.bin", arena, &input);
    ok &= load_binary_tensor("data/phase2_attn_qkv_w.bin", arena, &qkv_w);
    ok &= load_binary_tensor("data/phase2_attn_proj_w.bin", arena, &proj_w);
    ok &= load_binary_tensor("data/phase2_attn_expected_out.bin", arena, &expected_out);
    if (!ok) {
        blt_arena_destroy(arena);
        return 0;
    }

    TEST_ASSERT(input.ndim == 2);
    size_t seq_len = input.shape[0];
    size_t embed_dim = input.shape[1];

    size_t out_shape[2] = { seq_len, embed_dim };
    blt_tensor output = blt_tensor_create(arena, out_shape, 2, BLT_DTYPE_FP32);

    // same config as generated
    blt_attention_config config = {
        .embed_dim = embed_dim,
        .num_heads = 4,
        .head_dim = 0,   // inferred as embed_dim / num_heads
        .is_causal = true,
    };

    blt_multihead_attention(&input, &qkv_w, &proj_w, &output, &config, arena);

    TEST_ASSERT_CLOSE(&output, &expected_out, PHASE2_ATOL);

    blt_arena_destroy(arena);
    return 1;
}



// -----------------------------------------------------------------------
// Transformer block parity test

int run_phase2_transformer_block_parity_test(void) {
    blt_arena* arena = blt_arena_create(PHASE2_ARENA_BYTES, BLT_BACKEND_CPU);
    if (!arena) {
        return 0;
    }

    blt_tensor input = {0};
    blt_tensor norm1_weight = {0};
    blt_tensor norm1_bias = {0};
    blt_tensor attn_qkv_w = {0};
    blt_tensor attn_proj_w = {0};
    blt_tensor norm2_weight = {0};
    blt_tensor norm2_bias = {0};
    blt_tensor ffn_up_w = {0};
    blt_tensor ffn_down_w = {0};
    blt_tensor expected_out = {0};

    int ok = 1;
    ok &= load_binary_tensor("data/phase2_block_input.bin", arena, &input);
    ok &= load_binary_tensor("data/phase2_block_norm1_weight.bin", arena, &norm1_weight);
    ok &= load_binary_tensor("data/phase2_block_norm1_bias.bin", arena, &norm1_bias);
    ok &= load_binary_tensor("data/phase2_block_attn_qkv_w.bin", arena, &attn_qkv_w);
    ok &= load_binary_tensor("data/phase2_block_attn_proj_w.bin", arena, &attn_proj_w);
    ok &= load_binary_tensor("data/phase2_block_norm2_weight.bin", arena, &norm2_weight);
    ok &= load_binary_tensor("data/phase2_block_norm2_bias.bin", arena, &norm2_bias);
    ok &= load_binary_tensor("data/phase2_block_ffn_up_w.bin", arena, &ffn_up_w);
    ok &= load_binary_tensor("data/phase2_block_ffn_down_w.bin", arena, &ffn_down_w);
    ok &= load_binary_tensor("data/phase2_block_expected_out.bin", arena, &expected_out);
    if (!ok) {
        blt_arena_destroy(arena);
        return 0;
    }

    TEST_ASSERT(input.ndim == 2);
    size_t seq_len = input.shape[0];
    size_t embed_dim = input.shape[1];
    size_t hidden_dim = ffn_up_w.shape[1];

    size_t out_shape[2] = { seq_len, embed_dim };
    blt_tensor output = blt_tensor_create(arena, out_shape, 2, BLT_DTYPE_FP32);

    // This block uses standard LayerNorm + GELU config
    blt_transformer_config config = {
        .attn_config = {
            .embed_dim = embed_dim,
            .num_heads = 4,
            .head_dim = 0,
            .is_causal = true,
        },
        .hidden_dim = hidden_dim,
        .layer_norm_eps = 1e-5f,
        .norm_type = BLT_NORM_LAYERNORM,          // [Note for myself]: Swap norm_type/activation_type for future tests
        .activation_type = BLT_ACTIVATION_GELU,
    };

    blt_transformer_weights weights = {
        .norm1_weight = &norm1_weight,
        .norm1_bias = &norm1_bias,
        .attn_qkv_w = &attn_qkv_w,
        .attn_proj_w = &attn_proj_w,
        .norm2_weight = &norm2_weight,
        .norm2_bias = &norm2_bias,
        .ffn_up_w = &ffn_up_w,
        .ffn_gate_w = NULL,   // [Unused]: activation_type == BLT_ACTIVATION_GELU
        .ffn_down_w = &ffn_down_w,
    };

    size_t offset_before = arena->offset;

    blt_transformer_forward(&input, &weights, &output, &config, arena);

    size_t offset_after = arena->offset;

    // confirm the block allocated stayed inside the arenas bump offset
    printf("[phase2] transformer block arena footprint: %zu -> %zu bytes (capacity %zu)\n",
           offset_before, offset_after, arena->capacity);
    TEST_ASSERT(offset_after <= arena->capacity);
    TEST_ASSERT(offset_after >= offset_before);

    TEST_ASSERT_CLOSE(&output, &expected_out, PHASE2_ATOL);

    blt_arena_destroy(arena);
    return 1;
}



// -----------------------------------------------------------------------
// both tests

int run_phase2_tests(void) {
    int ok = 1;
    ok &= run_phase2_attention_parity_test();
    ok &= run_phase2_transformer_block_parity_test();
    return ok;
}