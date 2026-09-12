#ifndef BLT_HASH_NGRAM_H
#define BLT_HASH_NGRAM_H


#ifdef __cplusplus
extern "C" {
#endif
#include "blt/core/tensor.h"
#include "blt/core/allocator.h"
#include <stdint.h>
#include <stdbool.h>

#define BLT_MAX_NGRAM_SIZES 6

// Which n-gram sizes to use, hash table dimensions, and output normalization.
typedef struct {
    size_t ngram_sizes[BLT_MAX_NGRAM_SIZES]; // Active n-gram sizes (e.g., {3, 4, 5, 6, 7, 8})
    size_t num_ngram_sizes;                 // Number of active sizes in ngram_sizes
    size_t per_ngram_vocab;                 // Size of each hash embedding table (modulus for the hash)
    uint64_t hash_prime;                    // Base prime for the rolling polynomial hash
    bool normalize;                         // If true, divide output by (num_ngram_sizes + 1)
    size_t embed_dim;                       // Dimensionality of the embeddings
} blt_hash_ngram_config;

typedef struct {
    blt_tensor tables[BLT_MAX_NGRAM_SIZES]; // each [per_ngram_vocab, embed_dim] FP32
    size_t num_tables;
} blt_hash_ngram_weights;

// Rolling polynomial hash, O(1) per position.
typedef struct {
    uint64_t current_hash;
    uint64_t prime;
    uint64_t modulus;
    uint64_t prime_pow_n;   // prime^n % modulus, precomputed
    uint8_t window[8];      // Circular buffer of the last n bytes
    size_t window_start;
    size_t n;               // N-gram size
    size_t positions_seen;  // How many bytes have been fed so far
} blt_rolling_hash_state;

// Initialize rolling hash state.
// n: n-gram size, prime: hash base, modulus: typically embedding table size.
void blt_rolling_hash_init(blt_rolling_hash_state* state, size_t n, uint64_t prime, uint64_t modulus);

// Feeds a new byte into the rolling hash. Returns the hash of the current
// n-gram if positions_seen >= n, UINT64_MAX otherwise.
uint64_t blt_rolling_hash_update(blt_rolling_hash_state* state, uint8_t new_byte);

// Allocates n-gram weights in the arena with small uniform random values.
blt_hash_ngram_weights blt_hash_ngram_create(blt_arena* arena, const blt_hash_ngram_config* config);

// bytes_in: 1D UINT8 [seq_len], byte_emb: 2D FP32 [seq_len, embed_dim].
// out: [seq_len, embed_dim] — byte_emb + n-gram embeddings.
void blt_hash_ngram_forward(
    const blt_hash_ngram_weights* weights,
    const blt_hash_ngram_config* config,
    const blt_tensor* bytes_in,
    const blt_tensor* byte_emb,
    blt_tensor* out
);

// bytes_in: 1D UINT8 [seq_len], grad_out: 2D FP32 [seq_len, embed_dim].
// grad_byte_emb: [seq_len, embed_dim]. grad_tables: [per_ngram_vocab, embed_dim]
// array — MUST be zero-initialized by the caller.
void blt_hash_ngram_backward(
    const blt_hash_ngram_config* config,
    const blt_tensor* bytes_in,
    const blt_tensor* grad_out,
    blt_tensor* grad_byte_emb,
    blt_tensor* grad_tables
);

#ifdef __cplusplus
}
#endif
#endif
