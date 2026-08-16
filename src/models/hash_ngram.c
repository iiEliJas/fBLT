#include "blt/models/hash_ngram.h"
#include "blt/core/backend.h"

#include <stdlib.h>
#include <string.h>
#include <stdint.h>



// -------------------------------------------------------------------
// Rolling polynomial hash
// -------------------------------------------------------------------

// Initializes the rolling hash state.
// We precompute prime^n (mod modulus) to allow O(1) rolling updates later.
// We validate that prime * modulus does not overflow uint64, ensuring all 
// subsequent multiplications remain safe.
void blt_rolling_hash_init(blt_rolling_hash_state* state, size_t n, uint64_t prime, uint64_t modulus) {
    BLT_REQUIRE(state != NULL, "blt_rolling_hash_init: state is NULL");
    BLT_REQUIRE(n > 0 && n <= 8, "blt_rolling_hash_init: n must be in [1, 8]");
    BLT_REQUIRE(modulus > 1, "blt_rolling_hash_init: modulus must be > 1");
    
    BLT_REQUIRE(prime > 0 && prime < (UINT64_MAX / modulus),
                "blt_rolling_hash_init: prime * modulus would overflow uint64");

    state->prime = prime;
    state->modulus = modulus;
    state->n = n;

    // Compute prime^n mod modulus.
    // Initialize pow_val to 1 mod modulus
    // This correctly handles modulus = 1 and serves as mult identity
    uint64_t pow_val = 1ULL % modulus;
    for (size_t i = 0; i < n; i++) {
        pow_val = (pow_val * prime) % modulus;
    }
    state->prime_pow_n = pow_val;

    memset(state->window, 0, sizeof(state->window));
    state->window_start = 0;
    state->positions_seen = 0;
    state->current_hash = 0;
}



// Feeds a new byte to the rolling hash
// For an n-gram ending at position i, the polynomial hash is:
//   H(i) = (b_i * p^(n-1) + b_(i-1) * p^(n-2) + ... + b_(i-n+1) * p^0) mod m
// When a new byte b_new arrives, we can compute the new hash from the old one:
//   H_new = (H_old * p + b_new - b_outgoing * p^n) mod m
// where b_outgoing is the byte that just left the n-gram window.
// Because (A - B) mod m can be negative in C, we add m before taking modulo:
//   H_new = (H_old * p + b_new + m - (b_outgoing * p^n) mod m) mod m
uint64_t blt_rolling_hash_update(blt_rolling_hash_state* state, uint8_t new_byte) {
    BLT_REQUIRE(state != NULL, "blt_rolling_hash_update: state is NULL");

    // Capture the byte about to be evicted before overwriting.
    uint8_t outgoing = state->window[state->window_start];

    // Write new byte into the circular buffer and advance.
    state->window[state->window_start] = new_byte;
    state->window_start = (state->window_start + 1) % state->n;
    state->positions_seen++;

    // Not enough bytes yet for a full n-gram.
    if (state->positions_seen < state->n) {
        return UINT64_MAX;
    }

    // First valid hash: compute directly from window contents.
    // Math: H = (b_0 * p^(n-1) + b_1 * p^(n-2) + ... + b_(n-1) * p^0) mod m
    if (state->positions_seen == state->n) {
        uint64_t hash = 0;
        size_t idx = state->window_start;
        for (size_t i = 0; i < state->n; i++) {
            hash = (hash * state->prime + state->window[idx]) % state->modulus;
            idx = (idx + 1) % state->n;
        }
        state->current_hash = hash;
        return state->current_hash;
    }

    // Rolling update.
    uint64_t updated = (state->current_hash * state->prime + new_byte) % state->modulus;
    uint64_t outgoing_contrib = ((uint64_t)outgoing * state->prime_pow_n) % state->modulus;
    
    // Add modulus to prevent negative underflow before final modulo.
    updated = (updated + state->modulus - outgoing_contrib) % state->modulus;
    state->current_hash = updated;

    return state->current_hash;
}



// -------------------------------------------------------------------
// Weight creation
// -------------------------------------------------------------------

// Allocates embedding tables and initializes them with small uniform random values
// Uses Uniform(-0.02, 0.02) instead of zero
blt_hash_ngram_weights blt_hash_ngram_create(blt_arena* arena, const blt_hash_ngram_config* config) {
    BLT_REQUIRE(arena != NULL, "blt_hash_ngram_create: arena is NULL");
    BLT_REQUIRE(config != NULL, "blt_hash_ngram_create: config is NULL");
    BLT_REQUIRE(config->num_ngram_sizes > 0 &&
                config->num_ngram_sizes <= BLT_MAX_NGRAM_SIZES,
                "blt_hash_ngram_create: num_ngram_sizes out of range");
    BLT_REQUIRE(config->per_ngram_vocab > 0,
                "blt_hash_ngram_create: per_ngram_vocab must be > 0");
    BLT_REQUIRE(config->embed_dim > 0,
                "blt_hash_ngram_create: embed_dim must be > 0");

    for (size_t i = 0; i < config->num_ngram_sizes; i++) {
        BLT_REQUIRE(config->ngram_sizes[i] > 0 &&
                    config->ngram_sizes[i] <= 8,
                    "blt_hash_ngram_create: ngram_sizes entries must be in [1, 8]");
    }

    blt_hash_ngram_weights weights;
    memset(&weights, 0, sizeof(weights));
    weights.num_tables = config->num_ngram_sizes;

    const size_t shape[2] = { config->per_ngram_vocab, config->embed_dim };
    const float bound = 0.02f;

    for (size_t t = 0; t < config->num_ngram_sizes; t++) {
        weights.tables[t] = blt_tensor_create(arena, shape, 2, BLT_DTYPE_FP32);

        float* data = (float*)weights.tables[t].data;
        size_t numel = weights.tables[t].numel;
        for (size_t i = 0; i < numel; i++) {
            float rand_float = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
            data[i] = rand_float * bound;
        }
    }

    return weights;
}



// -------------------------------------------------------------------
// Forward
// -------------------------------------------------------------------

// Forward pass for the hash n-gram module.
// For each position i, the output is:
//   out_i = byte_emb_i + sum_{n} Table_n[hash(b_{i-n+1...i})]
// If normalize is true, the entire sum is divided by (K + 1), where K is num_ngram_sizes.
//   out_i = (1 / (K + 1)) * (byte_emb_i + sum_{n} Table_n[hash(...)])
// Positions i < n-1 receive no contribution from the size-n table.
void blt_hash_ngram_forward(
    const blt_hash_ngram_weights* weights,
    const blt_hash_ngram_config* config,
    const blt_tensor* bytes_in,
    const blt_tensor* byte_emb,
    blt_tensor* out
) {

    BLT_REQUIRE(weights != NULL, "blt_hash_ngram_forward: weights is NULL");
    BLT_REQUIRE(config != NULL, "blt_hash_ngram_forward: config is NULL");
    BLT_REQUIRE(bytes_in != NULL, "blt_hash_ngram_forward: bytes_in is NULL");
    BLT_REQUIRE(byte_emb != NULL, "blt_hash_ngram_forward: byte_emb is NULL");
    BLT_REQUIRE(out != NULL, "blt_hash_ngram_forward: out is NULL");

    BLT_REQUIRE(bytes_in->ndim == 1 && bytes_in->dtype == BLT_DTYPE_UINT8,
                "blt_hash_ngram_forward: bytes_in must be 1D UINT8");

    const size_t seq_len = bytes_in->shape[0];
    const size_t embed_dim = config->embed_dim;
    const size_t table_dims[2] = { config->per_ngram_vocab, embed_dim };

    {
        const size_t dims[2] = { seq_len, embed_dim };
        blt_check_nd_fp32(byte_emb, 2, dims, "blt_hash_ngram_forward: byte_emb");
        blt_check_nd_fp32(out, 2, dims, "blt_hash_ngram_forward: out");
    }

    const uint8_t* bytes = (const uint8_t*)bytes_in->data;
    const float* emb_in = (const float*)byte_emb->data;
    float* out_data = (float*)out->data;

    // Step 1: Copy base byte embeddings into out
    if (out_data != emb_in) {
        memcpy(out_data, emb_in, seq_len * embed_dim * sizeof(float));
    }

    // Step 2: Add n-gram contributions
    for (size_t n_idx = 0; n_idx < config->num_ngram_sizes; n_idx++) {
        size_t n = config->ngram_sizes[n_idx];
        if (n == 0 || n > seq_len) {
            continue;
        }

        blt_check_nd_fp32(&weights->tables[n_idx], 2, table_dims,
                          "blt_hash_ngram_forward: weights->tables");

        blt_rolling_hash_state state;
        blt_rolling_hash_init(&state, n, config->hash_prime, config->per_ngram_vocab);

        const float* table = (const float*)weights->tables[n_idx].data;

        for (size_t i = 0; i < seq_len; i++) {
            uint64_t hash = blt_rolling_hash_update(&state, bytes[i]);
            if (hash == UINT64_MAX) {
                continue;
            }
            const float* emb_ptr = table + (size_t)hash * embed_dim;
            float* out_row = out_data + i * embed_dim;
            for (size_t e = 0; e < embed_dim; e++) {
                out_row[e] += emb_ptr[e];
            }
        }
    }

    // Step 3: Apply normalization scale
    if (config->normalize) {
        float scale = 1.0f / (float)(config->num_ngram_sizes + 1);
        size_t total = seq_len * embed_dim;
        for (size_t i = 0; i < total; i++) {
            out_data[i] *= scale;
        }
    }
}



// -------------------------------------------------------------------
// Backward
// -------------------------------------------------------------------

// Backward pass for gradients
// Given dL/d_out, we compute gradients
// 1. dL/d_byte_emb = (1 / (K + 1)) * dL/d_out
// 2. dL/d_Table_n[hash] += (1 / (K + 1)) * dL/d_out_i
// Because the same hash bucket can be hit by multiple positions, we must scatter add the gradients
void blt_hash_ngram_backward(
    const blt_hash_ngram_config* config,
    const blt_tensor* bytes_in,
    const blt_tensor* grad_out,
    blt_tensor* grad_byte_emb,
    blt_tensor* grad_tables
) {

    BLT_REQUIRE(config != NULL, "blt_hash_ngram_backward: config is NULL");
    BLT_REQUIRE(bytes_in != NULL, "blt_hash_ngram_backward: bytes_in is NULL");
    BLT_REQUIRE(grad_out != NULL, "blt_hash_ngram_backward: grad_out is NULL");
    BLT_REQUIRE(grad_byte_emb != NULL, "blt_hash_ngram_backward: grad_byte_emb is NULL");
    BLT_REQUIRE(grad_tables != NULL, "blt_hash_ngram_backward: grad_tables is NULL");

    BLT_REQUIRE(bytes_in->ndim == 1 && bytes_in->dtype == BLT_DTYPE_UINT8,
                "blt_hash_ngram_backward: bytes_in must be 1D UINT8");

    const size_t seq_len = bytes_in->shape[0];
    const size_t embed_dim = config->embed_dim;
    const size_t table_dims[2] = { config->per_ngram_vocab, embed_dim };

    {
        const size_t dims[2] = { seq_len, embed_dim };
        blt_check_nd_fp32(grad_out, 2, dims, "blt_hash_ngram_backward: grad_out");
        blt_check_nd_fp32(grad_byte_emb, 2, dims, "blt_hash_ngram_backward: grad_byte_emb");
    }

    const uint8_t* bytes = (const uint8_t*)bytes_in->data;
    const float* grad_out_data = (const float*)grad_out->data;
    float* grad_byte_emb_data = (float*)grad_byte_emb->data;

    float scale = config->normalize
        ? 1.0f / (float)(config->num_ngram_sizes + 1)
        : 1.0f;

    // Scatter add scaled grad_out into grad_tables
    for (size_t n_idx = 0; n_idx < config->num_ngram_sizes; n_idx++) {
        size_t n = config->ngram_sizes[n_idx];
        if (n == 0 || n > seq_len) {
            continue;
        }

        blt_check_nd_fp32(&grad_tables[n_idx], 2, table_dims,
                          "blt_hash_ngram_backward: grad_tables");

        blt_rolling_hash_state state;
        blt_rolling_hash_init(&state, n, config->hash_prime, config->per_ngram_vocab);

        float* grad_table = (float*)grad_tables[n_idx].data;

        for (size_t i = 0; i < seq_len; i++) {
            uint64_t hash = blt_rolling_hash_update(&state, bytes[i]);
            if (hash == UINT64_MAX) {
                continue;
            }
            const float* grad_row = grad_out_data + i * embed_dim;
            float* grad_table_row = grad_table + (size_t)hash * embed_dim;
            for (size_t e = 0; e < embed_dim; e++) {
                grad_table_row[e] += scale * grad_row[e];
            }
        }
    }

    // grad_byte_emb = scale * grad_out
    size_t total = seq_len * embed_dim;
    for (size_t i = 0; i < total; i++) {
        grad_byte_emb_data[i] = scale * grad_out_data[i];
    }
}