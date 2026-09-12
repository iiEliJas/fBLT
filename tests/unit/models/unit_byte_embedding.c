#include <stdio.h>
#include "blt/core/allocator.h"
#include "blt/models/byte_embedding.h"

#include "test_helpers.h"
#include "test_suite.h"

int run_byte_embedding_model_tests(void) {
    // 256 x embed_dim floats uses 4096 bytes for embed_dim=4
    blt_arena *arena = blt_arena_create(16384, BLT_BACKEND_CPU);
    if (!arena) {
        fprintf(stderr, "[FAIL] arena creation for byte embedding tests\n");
        return 0;
    }

    size_t embed_dim = 4;
    blt_byte_embedding emb = blt_byte_embedding_create(arena, 256, embed_dim);

    // Set rows for byte values 0, 1, and 65 ('A')
    float *weight_data = (float *)emb.weight.data;
    for (size_t d = 0; d < embed_dim; d++) {
        weight_data[0 * embed_dim + d] = 0.0f;              // byte 0
        weight_data[1 * embed_dim + d] = 1.0f + (float)d;   // byte 1
        weight_data[65 * embed_dim + d] = 10.0f + (float)d; // byte 'A'
    }

    size_t seq_len = 3;
    size_t bytes_shape[1] = {seq_len};
    blt_tensor bytes_in = blt_tensor_create(arena, bytes_shape, 1, BLT_DTYPE_UINT8);
    uint8_t *bytes_data = (uint8_t *)bytes_in.data;
    bytes_data[0] = 65;
    bytes_data[1] = 0;
    bytes_data[2] = 1;

    size_t out_shape[2] = {seq_len, embed_dim};
    blt_tensor out = blt_tensor_create(arena, out_shape, 2, BLT_DTYPE_FP32);

    blt_byte_embedding_forward(&emb, &bytes_in, &out);

    float *out_data = (float *)out.data;
    // Row 0 of output should be weight row for byte 65
    for (size_t d = 0; d < embed_dim; d++) {
        TEST_ASSERT(out_data[0 * embed_dim + d] == 10.0f + (float)d);
    }
    // Row 1 of output should be weight row for byte 0
    for (size_t d = 0; d < embed_dim; d++) {
        TEST_ASSERT(out_data[1 * embed_dim + d] == 0.0f);
    }
    // Row 2 of output should be weight row for byte 1
    for (size_t d = 0; d < embed_dim; d++) {
        TEST_ASSERT(out_data[2 * embed_dim + d] == 1.0f + (float)d);
    }

    //------------------------------------------------------------
    // Backward embedding test
    // scatter-add grad_out into grad_weight at the rows touched by bytes_in.
    // Use two positions that hit the same byte value to check accumulation.
    bytes_data[0] = 1;
    bytes_data[1] = 1;
    bytes_data[2] = 2;

    blt_tensor grad_out = blt_tensor_create(arena, out_shape, 2, BLT_DTYPE_FP32);
    float *grad_out_data = (float *)grad_out.data;
    for (size_t i = 0; i < seq_len * embed_dim; i++) {
        grad_out_data[i] = 1.0f;
    }

    size_t grad_weight_shape[2] = {256, embed_dim};
    blt_tensor grad_weight = blt_tensor_create(arena, grad_weight_shape, 2, BLT_DTYPE_FP32);

    blt_byte_embedding_backward(&emb, &bytes_in, &grad_out, &grad_weight);

    float *grad_weight_data = (float *)grad_weight.data;
    // byte 1 was hit twice so grad row should be 2
    for (size_t d = 0; d < embed_dim; d++) {
        TEST_ASSERT(grad_weight_data[1 * embed_dim + d] == 2.0f);
    }
    // byte 2 was hit once so grad row should be 1
    for (size_t d = 0; d < embed_dim; d++) {
        TEST_ASSERT(grad_weight_data[2 * embed_dim + d] == 1.0f);
    }
    // an untouched byte row should be 0
    for (size_t d = 0; d < embed_dim; d++) {
        TEST_ASSERT(grad_weight_data[3 * embed_dim + d] == 0.0f);
    }

    blt_arena_destroy(arena);
    return 1;
}