#include "test_helpers.h"
#include "test_suite.h"

#include "blt/models/hash_ngram.h"
#include "blt/core/allocator.h"

#include <stdint.h>
#include <string.h>



// -------------------------------------------------------------------
// Test 1: rolling hash returns sentinel until enough bytes are seen, 
// then produces a deterministic value
// -------------------------------------------------------------------
static int test_rolling_hash_basic(void) {
    blt_rolling_hash_state state;
    blt_rolling_hash_init(&state, 3, 1000000007ULL, 97);
 
    uint64_t h0 = blt_rolling_hash_update(&state, 'a');
    uint64_t h1 = blt_rolling_hash_update(&state, 'b');
    TEST_ASSERT(h0 == UINT64_MAX);
    TEST_ASSERT(h1 == UINT64_MAX);
 
    uint64_t h2 = blt_rolling_hash_update(&state, 'c');
    TEST_ASSERT(h2 != UINT64_MAX);
    TEST_ASSERT(h2 < 97);
 
    uint64_t h3 = blt_rolling_hash_update(&state, 'd');
    TEST_ASSERT(h3 != UINT64_MAX);
    TEST_ASSERT(h3 < 97);
 
    return 1;
}
 


// -------------------------------------------------------------------
// Test 2: same byte sequence produces the same hash trace
// -------------------------------------------------------------------

static int test_rolling_hash_deterministic(void) {
    const uint8_t bytes[6] = { 'f','u','n','c','t','n' };
 
    blt_rolling_hash_state s1, s2;
    blt_rolling_hash_init(&s1, 3, 1000000007ULL, 97);
    blt_rolling_hash_init(&s2, 3, 1000000007ULL, 97);
 
    for (size_t i = 0; i < 6; i++) {
        uint64_t r1 = blt_rolling_hash_update(&s1, bytes[i]);
        uint64_t r2 = blt_rolling_hash_update(&s2, bytes[i]);
        TEST_ASSERT(r1 == r2);
    }
 
    return 1;
}




// -------------------------------------------------------------------
// Test 3: Hash Determinism & Rolling Correctness
// -------------------------------------------------------------------

static int test_hash_rolling_correctness(void) {
    const size_t n = 4;
    const uint64_t prime = 31;
    const uint64_t modulus = 1000; // acting as vocab_size
    const uint8_t bytes[] = {10, 20, 30, 40, 50, 60, 70};
    const size_t seq_len = 7;

    blt_rolling_hash_state state;
    blt_rolling_hash_init(&state, n, prime, modulus);

    for (size_t i = 0; i < seq_len; i++) {
        uint64_t hash = blt_rolling_hash_update(&state, bytes[i]);

        if (i < n - 1) {
            TEST_ASSERT(hash == UINT64_MAX);
        } else {
            // Manually compute: sum_{j=0}^{n-1} b_{i-j} * prime^{n-1-j} (mod modulus)
            uint64_t manual_hash = 0;
            for (size_t j = 0; j < n; j++) {
                uint64_t p_pow = 1;
                for (size_t k = 0; k < n - 1 - j; k++) {
                    p_pow = (p_pow * prime) % modulus;
                }
                uint64_t term = ((uint64_t)bytes[i - j] * p_pow) % modulus;
                manual_hash = (manual_hash + term) % modulus;
            }
            TEST_ASSERT(hash == manual_hash);
        }
    }

    return 1;
}



// -------------------------------------------------------------------
// Test 4: Boundary Omission
// -------------------------------------------------------------------

static int test_boundary_omission() {
    blt_arena* arena = blt_arena_create(1024 * 1024, BLT_BACKEND_CPU);
    TEST_ASSERT(arena != NULL);

    blt_hash_ngram_config config;
    memset(&config, 0, sizeof(config));
    config.ngram_sizes[0] = 3;
    config.num_ngram_sizes = 1;
    config.per_ngram_vocab = 1000;
    config.hash_prime = 31;
    config.normalize = false; // Disable normalization to verify exact equality
    config.embed_dim = 4;

    const size_t seq_len = 10;
    uint8_t bytes[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

    blt_hash_ngram_weights weights = blt_hash_ngram_create(arena, &config);

    blt_tensor bytes_in = {0};
    blt_tensor byte_emb = {0};
    blt_tensor out = {0};

    size_t bytes_shape[1] = {seq_len};
    bytes_in = blt_tensor_create(arena, bytes_shape, 1, BLT_DTYPE_UINT8);
    memcpy(bytes_in.data, bytes, seq_len * sizeof(uint8_t));

    size_t emb_shape[2] = {seq_len, config.embed_dim};
    byte_emb = blt_tensor_create(arena, emb_shape, 2, BLT_DTYPE_FP32);
    out = blt_tensor_create(arena, emb_shape, 2, BLT_DTYPE_FP32);

    // Fill byte_emb with identifiable non-zero values
    float* emb_data = (float*)byte_emb.data;
    for (size_t i = 0; i < seq_len * config.embed_dim; i++) {
        emb_data[i] = (float)i + 1.0f;
    }

    blt_hash_ngram_forward(&weights, &config, &bytes_in, &byte_emb, &out);

    float* out_data = (float*)out.data;

    // Assert position 0 gets no contribution from the 3-gram table
    for (size_t e = 0; e < config.embed_dim; e++) {
        TEST_ASSERT(out_data[0 * config.embed_dim + e] == emb_data[0 * config.embed_dim + e]);
    }

    // Assert position 1 gets no contribution from the 3-gram table
    for (size_t e = 0; e < config.embed_dim; e++) {
        TEST_ASSERT(out_data[1 * config.embed_dim + e] == emb_data[1 * config.embed_dim + e]);
    }

    blt_arena_destroy(arena);
    return 1;
}


// -------------------------------------------------------------------
// Helper for small config. 2 n-gram sizes, tiny vocab/embed_dim
// -------------------------------------------------------------------

static blt_hash_ngram_config make_test_config(size_t embed_dim) {
    blt_hash_ngram_config config;
    memset(&config, 0, sizeof(config));
    config.ngram_sizes[0] = 3;
    config.ngram_sizes[1] = 4;
    config.num_ngram_sizes = 2;
    config.per_ngram_vocab = 97;
    config.hash_prime = 1000000007ULL;
    config.normalize = true;
    config.embed_dim = embed_dim;
    return config;
}



// -------------------------------------------------------------------
// Test 4: forward pass on a longer sequence adds nonzero contributions
// for positions past the smallest n-gram size
// -------------------------------------------------------------------

static int test_forward_long_sequence(void) {
    const size_t embed_dim = 4;

    blt_arena* arena = blt_arena_create(4 * 1024 * 1024, BLT_BACKEND_CPU);
    TEST_ASSERT(arena != NULL);
 
    blt_hash_ngram_config config = make_test_config(embed_dim);
    blt_hash_ngram_weights weights = blt_hash_ngram_create(arena, &config);
 
    size_t seq_len = 8;
    size_t bytes_shape[1] = { seq_len };
    blt_tensor bytes_in = blt_tensor_create(arena, bytes_shape, 1, BLT_DTYPE_UINT8);
    uint8_t* bytes_data = (uint8_t*)bytes_in.data;
    const char* text = "function";  // 8 bytes
    memcpy(bytes_data, text, seq_len);
 
    size_t emb_shape[2] = { seq_len, embed_dim };
    blt_tensor byte_emb = blt_tensor_create(arena, emb_shape, 2, BLT_DTYPE_FP32);
    float* emb_data = (float*)byte_emb.data;
    memset(emb_data, 0, seq_len * embed_dim * sizeof(float));
 
    blt_tensor out = blt_tensor_create(arena, emb_shape, 2, BLT_DTYPE_FP32);
 
    blt_hash_ngram_forward(&weights, &config, &bytes_in, &byte_emb, &out);
 
    TEST_ASSERT(out.shape[0] == seq_len);
    TEST_ASSERT(out.shape[1] == embed_dim);
 
    /* Position 0 gets no n-gram contribution (neither n=3 nor n=4 has
       enough preceding bytes yet), so with a zero byte embedding and
       normalization it must be exactly zero. */
    float* out_data = (float*)out.data;
    for (size_t e = 0; e < embed_dim; e++) {
        TEST_ASSERT(out_data[0 * embed_dim + e] == 0.0f);
    }
 
    /* Position 3 (0-indexed) has seen 4 bytes, so both n=3 and n=4
       n-grams are active; output should be nonzero somewhere in the row. */
    int nonzero_found = 0;
    for (size_t e = 0; e < embed_dim; e++) {
        if (out_data[3 * embed_dim + e] != 0.0f) {
            nonzero_found = 1;
            break;
        }
    }
    TEST_ASSERT(nonzero_found);
 
    blt_arena_destroy(arena);
    return 1;
}
 


// -------------------------------------------------------------------
// Test 5: backward pass produces correctly-shaped gradients
// checks grad_byte_emb is the (scaled) copy of grad_out
// -------------------------------------------------------------------

static int test_backward_basic(void) {
    const size_t embed_dim = 4;
    
    blt_arena* arena = blt_arena_create(4 * 1024 * 1024, BLT_BACKEND_CPU);
    TEST_ASSERT(arena != NULL);
 
    blt_hash_ngram_config config = make_test_config(embed_dim);

    size_t seq_len = 8;
    size_t bytes_shape[1] = { seq_len };
    blt_tensor bytes_in = blt_tensor_create(arena, bytes_shape, 1, BLT_DTYPE_UINT8);
    uint8_t* bytes_data = (uint8_t*)bytes_in.data;
    const char* text = "function";
    memcpy(bytes_data, text, seq_len);
 
    size_t emb_shape[2] = { seq_len, embed_dim };
    blt_tensor grad_out = blt_tensor_create(arena, emb_shape, 2, BLT_DTYPE_FP32);
    float* grad_out_data = (float*)grad_out.data;
    for (size_t i = 0; i < seq_len * embed_dim; i++) {
        grad_out_data[i] = 2.0f;
    }
 
    blt_tensor grad_byte_emb = blt_tensor_create(arena, emb_shape, 2, BLT_DTYPE_FP32);
 
    blt_tensor grad_tables[BLT_MAX_NGRAM_SIZES];
    for (size_t t = 0; t < config.num_ngram_sizes; t++) {
        size_t table_shape[2] = { config.per_ngram_vocab, embed_dim };
        grad_tables[t] = blt_tensor_create(arena, table_shape, 2, BLT_DTYPE_FP32);
        memset(grad_tables[t].data, 0, grad_tables[t].numel * sizeof(float));
    }
 
    blt_hash_ngram_backward(&config, &bytes_in, &grad_out, &grad_byte_emb, grad_tables);
 
    // grad_byte_emb should equal grad_out scaled by 1/(num_ngram_sizes+1)
    float scale = 1.0f / (float)(config.num_ngram_sizes + 1);
    float* grad_byte_emb_data = (float*)grad_byte_emb.data;
    for (size_t i = 0; i < seq_len * embed_dim; i++) {
        float expected = grad_out_data[i] * scale;
        float diff = grad_byte_emb_data[i] - expected;
        if (diff < 0) diff = -diff;
        TEST_ASSERT(diff < 1e-5f);
    }
 
    // At least one table should have received a nonzero scatter-add.
    int nonzero_found = 0;
    for (size_t t = 0; t < config.num_ngram_sizes; t++) {
        float* table_data = (float*)grad_tables[t].data;
        for (size_t i = 0; i < grad_tables[t].numel; i++) {
            if (table_data[i] != 0.0f) {
                nonzero_found = 1;
                break;
            }
        }
    }
    TEST_ASSERT(nonzero_found);
 
    blt_arena_destroy(arena);
    return 1;
}


// -------------------------------------------------------------------
// Main Test Runner
// -------------------------------------------------------------------

int run_hash_ngram_model_tests(void) {
    int ok = 1;
    ok &= test_rolling_hash_basic();
    ok &= test_rolling_hash_deterministic();
    ok &= test_hash_rolling_correctness();
    ok &= test_boundary_omission();
    ok &= test_forward_long_sequence();
    ok &= test_backward_basic();

    return ok;
}