#include "test_helpers.h"
#include "test_suite.h"

#include "blt/core/allocator.h"
#include "blt/core/tensor.h"
#include "blt/models/attention.h"
#include "blt/models/transformer.h"

#include <stdio.h>

// Absolute tolerance
#define PARITY_ATOL 1e-4f

#define PARITY_ARENA_BYTES (16 * 1024 * 1024)



// -----------------------------------------------------------------------
// Attention parity test

int run_attention_parity_test(void) {
    blt_arena* arena = blt_arena_create(PARITY_ARENA_BYTES, BLT_BACKEND_CPU);
    if (!arena) {
        return 0;
    }

    blt_tensor input = {0};
    blt_tensor qkv_w = {0};
    blt_tensor proj_w = {0};
    blt_tensor expected_out = {0};

    int ok = 1;
    ok &= load_binary_tensor("data/attn_input.bin", arena, &input);
    ok &= load_binary_tensor("data/attn_qkv_w.bin", arena, &qkv_w);
    ok &= load_binary_tensor("data/attn_proj_w.bin", arena, &proj_w);
    ok &= load_binary_tensor("data/attn_expected_out.bin", arena, &expected_out);
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
    blt_attention_config config = {0};
    config.embed_dim = embed_dim;
    config.num_heads = 4;
    config.head_dim = 0;  // inferred as embed_dim / num_heads
    config.is_causal = true;

    blt_multihead_attention(&input, &qkv_w, &proj_w, &output, &config, arena);

    TEST_ASSERT_CLOSE(&output, &expected_out, PARITY_ATOL);

    blt_arena_destroy(arena);
    return 1;
}



int run_attention_backward_parity_test(void) {
    blt_arena* arena = blt_arena_create(PARITY_ARENA_BYTES, BLT_BACKEND_CPU);
    if (!arena) {
        return 0;
    }
 
    blt_tensor input = {0};
    blt_tensor qkv_w = {0};
    blt_tensor proj_w = {0};
    blt_tensor grad_out = {0};
    blt_tensor expected_grad_input = {0};
    blt_tensor expected_grad_qkv_w = {0};
    blt_tensor expected_grad_proj_w = {0};
 
    int ok = 1;
    ok &= load_binary_tensor("data/attn_input.bin", arena, &input);
    ok &= load_binary_tensor("data/attn_qkv_w.bin", arena, &qkv_w);
    ok &= load_binary_tensor("data/attn_proj_w.bin", arena, &proj_w);
    ok &= load_binary_tensor("data/attn_grad_out.bin", arena, &grad_out);
    ok &= load_binary_tensor("data/attn_expected_grad_input.bin", arena, &expected_grad_input);
    ok &= load_binary_tensor("data/attn_expected_grad_qkv_w.bin", arena, &expected_grad_qkv_w);
    ok &= load_binary_tensor("data/attn_expected_grad_proj_w.bin", arena, &expected_grad_proj_w);
    if (!ok) {
        blt_arena_destroy(arena);
        return 0;
    }
 
    TEST_ASSERT(input.ndim == 2);
    size_t seq_len = input.shape[0];
    size_t embed_dim = input.shape[1];
 
    size_t in_shape[2] = { seq_len, embed_dim };
    size_t qkv_shape[2] = { embed_dim, 3 * embed_dim };
    size_t proj_shape[2] = { embed_dim, embed_dim };
 
    blt_tensor grad_input = blt_tensor_create(arena, in_shape, 2, BLT_DTYPE_FP32);
    blt_tensor grad_weight_qkv = blt_tensor_create(arena, qkv_shape, 2, BLT_DTYPE_FP32);
    blt_tensor grad_weight_proj = blt_tensor_create(arena, proj_shape, 2, BLT_DTYPE_FP32);
 
    // same config as the forward parity test
    blt_attention_config config = {0};
    config.embed_dim = embed_dim;
    config.num_heads = 4;
    config.head_dim = 0;  // inferred as embed_dim / num_heads
    config.is_causal = true;
 
    blt_multihead_attention_backward(&input, &qkv_w, &proj_w, &grad_out,
                                      &grad_input, &grad_weight_qkv, &grad_weight_proj,
                                      &config, arena);
 
    TEST_ASSERT_CLOSE(&grad_input, &expected_grad_input, PARITY_ATOL);
    TEST_ASSERT_CLOSE(&grad_weight_qkv, &expected_grad_qkv_w, PARITY_ATOL);
    TEST_ASSERT_CLOSE(&grad_weight_proj, &expected_grad_proj_w, PARITY_ATOL);
 
    blt_arena_destroy(arena);
    return 1;
}


// -----------------------------------------------------------------------
// Transformer block parity test

int run_transformer_block_parity_test(void) {
    blt_arena* arena = blt_arena_create(PARITY_ARENA_BYTES, BLT_BACKEND_CPU);
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
    ok &= load_binary_tensor("data/transformer_input.bin", arena, &input);
    ok &= load_binary_tensor("data/transformer_norm1_weight.bin", arena, &norm1_weight);
    ok &= load_binary_tensor("data/transformer_norm1_bias.bin", arena, &norm1_bias);
    ok &= load_binary_tensor("data/transformer_attn_qkv_w.bin", arena, &attn_qkv_w);
    ok &= load_binary_tensor("data/transformer_attn_proj_w.bin", arena, &attn_proj_w);
    ok &= load_binary_tensor("data/transformer_norm2_weight.bin", arena, &norm2_weight);
    ok &= load_binary_tensor("data/transformer_norm2_bias.bin", arena, &norm2_bias);
    ok &= load_binary_tensor("data/transformer_ffn_up_w.bin", arena, &ffn_up_w);
    ok &= load_binary_tensor("data/transformer_ffn_down_w.bin", arena, &ffn_down_w);
    ok &= load_binary_tensor("data/transformer_expected_out.bin", arena, &expected_out);
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
            .use_rope = true,
            .rope_theta = 10000.0f,
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
    printf("    [transformer parity] block arena footprint: %zu -> %zu bytes (capacity %zu)\n",
           offset_before, offset_after, arena->capacity);
    TEST_ASSERT(offset_after <= arena->capacity);
    TEST_ASSERT(offset_after >= offset_before);

    TEST_ASSERT_CLOSE(&output, &expected_out, PARITY_ATOL);

    blt_arena_destroy(arena);
    return 1;
}



// -----------------------------------------------------------------------
// all tests

int run_transformer_parity_tests(void) {
    int ok = 1;
    ok &= run_attention_parity_test();
    ok &= run_attention_backward_parity_test();
    ok &= run_transformer_block_parity_test();
    return ok;
}