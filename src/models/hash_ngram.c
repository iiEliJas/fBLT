#include "blt/models/hash_ngram.h"
#include "blt/core/backend.h"
#include "blt/ops/gather_scatter.h"
#include "blt/ops/vecmath.h"
#include "blt/ops/elementwise.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>


// Rolling polynomial hash
//
// We precompute prime^n (mod modulus) to allow O(1) rolling updates later.
// We validate that prime * modulus does not overflow uint64, ensuring all
// subsequent multiplications remain safe.
void blt_rolling_hash_init(blt_rolling_hash_state* state, size_t n, uint64_t prime, uint64_t modulus) {
    BLT_REQUIRE(state != NULL, "blt_rolling_hash_init: state is NULL");
    BLT_REQUIRE(n > 0 && n <= 8, "blt_rolling_hash_init: n must be in [1, 8]");
    BLT_REQUIRE(modulus > 1, "blt_rolling_hash_init: modulus must be > 1");

    // Reserve headroom for the +byte term in the rolling update:
    // max intermediate is (modulus-1)*prime + 255, which must stay in uint64.
    BLT_REQUIRE(prime > 0 && prime <= ((UINT64_MAX - 255) / modulus),
                "blt_rolling_hash_init: prime * modulus would overflow uint64");

    state->prime = prime;
    state->modulus = modulus;
    state->n = n;

    // Compute prime^n mod modulus.
    // Initialize pow_val to 1 mod modulus -- correctly handles modulus = 1
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


uint64_t blt_rolling_hash_update(blt_rolling_hash_state* state, uint8_t new_byte) {
    BLT_REQUIRE(state != NULL, "blt_rolling_hash_update: state is NULL");

    // Capture the byte about to be evicted before overwriting.
    uint8_t outgoing = state->window[state->window_start];

    state->window[state->window_start] = new_byte;
    state->window_start = (state->window_start + 1) % state->n;
    state->positions_seen++;

    if (state->positions_seen < state->n) {
        return UINT64_MAX;
    }

    // First valid hash: compute directly from window contents.
    // H = (b_0 * p^(n-1) + b_1 * p^(n-2) + ... + b_(n-1) * p^0) mod m
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

    // Rolling update: shift left, add new byte, subtract outgoing contribution.
    uint64_t updated = (state->current_hash * state->prime + new_byte) % state->modulus;
    uint64_t outgoing_contrib = ((uint64_t)outgoing * state->prime_pow_n) % state->modulus;

    // Add modulus to prevent negative underflow before final modulo.
    updated = (updated + state->modulus - outgoing_contrib) % state->modulus;
    state->current_hash = updated;

    return state->current_hash;
}


// Byte-id tensors may live on either backend; hashing consumes the ids on
// the host, so device-resident inputs are staged through a host copy.
static const uint8_t* bytes_host(const blt_tensor* bytes_in, uint8_t** staging) {
    if (bytes_in->backend == BLT_BACKEND_CPU) {
        *staging = NULL;
        return (const uint8_t*)bytes_in->data;
    }
    *staging = (uint8_t*)malloc(bytes_in->numel);
    BLT_REQUIRE(*staging != NULL, "bytes_host: staging alloc failed");
    blt_tensor_download(bytes_in, *staging, bytes_in->numel);
    return *staging;
}


// Allocates embedding tables and initializes them with small uniform random values.
// Uses Uniform(-0.02, 0.02) instead of zero.
blt_hash_ngram_weights blt_hash_ngram_create(blt_arena* arena, const blt_hash_ngram_config* config) {
    BLT_REQUIRE(arena != NULL, "blt_hash_ngram_create: arena is NULL");
    BLT_REQUIRE(config != NULL, "blt_hash_ngram_create: config is NULL");
    BLT_REQUIRE(config->num_ngram_sizes <= BLT_MAX_NGRAM_SIZES,
                "blt_hash_ngram_create: num_ngram_sizes out of range");
    BLT_REQUIRE(config->num_ngram_sizes == 0 || config->per_ngram_vocab > 0,
                "blt_hash_ngram_create: per_ngram_vocab must be > 0 when tables are used");
    BLT_REQUIRE(config->embed_dim > 0,
                "blt_hash_ngram_create: embed_dim must be > 0");

    // num_ngram_sizes == 0 is valid: the module is disabled and contributes
    // nothing (forward/backward loop over zero tables).
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

        // Init runs on a host staging buffer: the arena may be device
        // memory, which cannot be written through its host pointer.
        float* stage = (float*)malloc(weights.tables[t].numel * sizeof(float));
        BLT_REQUIRE(stage != NULL, "blt_hash_ngram_create: staging alloc failed");
        for (size_t i = 0; i < weights.tables[t].numel; i++) {
            float rand_float = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
            stage[i] = rand_float * bound;
        }
        blt_tensor_upload(&weights.tables[t], stage,
                          weights.tables[t].numel * sizeof(float));
        free(stage);
    }

    return weights;
}


// Forward: out_i = byte_emb_i + sum_{n} Table_n[hash(b_{i-n+1...i})]
// If normalize is true, the entire sum is divided by (K + 1), where K is
// num_ngram_sizes:
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

    uint8_t* bytes_stage = NULL;
    const uint8_t* bytes = bytes_host(bytes_in, &bytes_stage);

    // Copy base byte embeddings into out.
    const size_t total = seq_len * embed_dim;
    blt_strided_copy(byte_emb->backend,
                     (float*)out->data, total,
                     (const float*)byte_emb->data, total, 1, total);

    // Add n-gram contributions. Hashing is integer-only host work;
    // bucket indices feed a dispatched accumulate op so tables may live on
    // either backend.
    uint32_t* idx = NULL;
    if (config->num_ngram_sizes > 0) {
        idx = (uint32_t*)malloc(seq_len * sizeof(uint32_t));
        BLT_REQUIRE(idx != NULL, "blt_hash_ngram_forward: failed to allocate index buffer");
    }
    for (size_t n_idx = 0; n_idx < config->num_ngram_sizes; n_idx++) {
        size_t n = config->ngram_sizes[n_idx];
        if (n == 0 || n > seq_len) {
            continue;
        }

        blt_check_nd_fp32(&weights->tables[n_idx], 2, table_dims,
                          "blt_hash_ngram_forward: weights->tables");

        blt_rolling_hash_state state;
        blt_rolling_hash_init(&state, n, config->hash_prime, config->per_ngram_vocab);

        for (size_t i = 0; i < seq_len; i++) {
            uint64_t hash = blt_rolling_hash_update(&state, bytes[i]);
            idx[i] = (hash == UINT64_MAX) ? BLT_IDX_SENTINEL : (uint32_t)hash;
        }
        blt_indexed_row_accumulate(&weights->tables[n_idx], idx, out);
    }
    free(idx);
    free(bytes_stage);

    // Apply normalization scale.
    if (config->normalize) {
        float scale = 1.0f / (float)(config->num_ngram_sizes + 1);
        blt_scale(out, scale);
    }
}


// Backward: dL/d_byte_emb = (1 / (K + 1)) * dL/d_out
//            dL/d_Table_n[hash] += (1 / (K + 1)) * dL/d_out_i
// Same hash bucket can be hit by multiple positions, so scatter-add gradients.
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

    uint8_t* bytes_stage = NULL;
    const uint8_t* bytes = bytes_host(bytes_in, &bytes_stage);

    float scale = config->normalize ? 1.0f / (float)(config->num_ngram_sizes + 1) : 1.0f;

    // Scatter-add scaled grad_out into grad_tables via dispatched ops;
    // hashing stays on the host (integer-only).
    uint32_t* idx = NULL;
    if (config->num_ngram_sizes > 0) {
        idx = (uint32_t*)malloc(seq_len * sizeof(uint32_t));
        BLT_REQUIRE(idx != NULL, "blt_hash_ngram_backward: failed to allocate index buffer");
    }
    for (size_t n_idx = 0; n_idx < config->num_ngram_sizes; n_idx++) {
        size_t n = config->ngram_sizes[n_idx];
        if (n == 0 || n > seq_len) {
            continue;
        }

        blt_check_nd_fp32(&grad_tables[n_idx], 2, table_dims,
                          "blt_hash_ngram_backward: grad_tables");

        blt_rolling_hash_state state;
        blt_rolling_hash_init(&state, n, config->hash_prime, config->per_ngram_vocab);

        for (size_t i = 0; i < seq_len; i++) {
            uint64_t hash = blt_rolling_hash_update(&state, bytes[i]);
            idx[i] = (hash == UINT64_MAX) ? BLT_IDX_SENTINEL : (uint32_t)hash;
        }
        blt_indexed_row_scatter_add_normalized(grad_tables[n_idx].backend, &grad_tables[n_idx], idx, grad_out, scale);
    }
    free(idx);
    free(bytes_stage);

    blt_scaled_copy(grad_byte_emb, grad_out, scale);
}
