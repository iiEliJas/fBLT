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

// Configuration for the hash n-gram embedding module.
// Controls which n-gram sizes are used, the size of the hash tables,
// and whether the output is normalized.
typedef struct {
    size_t ngram_sizes[BLT_MAX_NGRAM_SIZES]; // Active n-gram sizes (e.g., {3, 4, 5, 6, 7, 8})
    size_t num_ngram_sizes;                 // Number of active sizes in ngram_sizes
    size_t per_ngram_vocab;                 // Size of each hash embedding table (modulus for the hash)
    uint64_t hash_prime;                    // Base prime for the rolling polynomial hash
    bool normalize;                         // If true, divide output by (num_ngram_sizes + 1)
    size_t embed_dim;                       // Dimensionality of the embeddings
} blt_hash_ngram_config;

// Container for the learnable hash n-gram embedding tables.
typedef struct {
    blt_tensor tables[BLT_MAX_NGRAM_SIZES]; // Array of tensors, each [per_ngram_vocab, embed_dim] FP32
    size_t num_tables;                      // Number of active tables (matches num_ngram_sizes)
} blt_hash_ngram_weights;

// Stateful rolling hash to compute polynomial hashes in O(1) per position.
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

// Init the rolling hash state
// n: size of the n-gram.
// prime: base for the polynomial hash.
// modulus: typically the size of the embedding table.
void blt_rolling_hash_init(blt_rolling_hash_state* state, size_t n, uint64_t prime, uint64_t modulus);

// Feeds a new byte into the rolling hash.
// Returns the hash of the current n-gram if positions_seen >= n.
// Returns UINT64_MAX if not enough bytes have been seen yet.
uint64_t blt_rolling_hash_update(blt_rolling_hash_state* state, uint8_t new_byte);

// Allocates and initializes the n-gram weights in the arena.
// Tables are initialized with small uniform random values
blt_hash_ngram_weights blt_hash_ngram_create(blt_arena* arena, const blt_hash_ngram_config* config);

// Forward pass for hash n-gram embeddings.
// Inputs:
//   bytes_in: 1D UINT8 tensor of raw byte values [seq_len]
//   byte_emb: 2D FP32 tensor of base byte embeddings [seq_len, embed_dim]
// Output:
//   out: 2D FP32 tensor [seq_len, embed_dim]. Contains byte_emb + n-gram embeddings.
void blt_hash_ngram_forward(
    const blt_hash_ngram_weights* weights,
    const blt_hash_ngram_config* config,
    const blt_tensor* bytes_in,
    const blt_tensor* byte_emb,
    blt_tensor* out
);

// Backward pass for hash n-gram embeddings.
// Inputs:
//   bytes_in: 1D UINT8 tensor of raw byte values [seq_len]
//   grad_out: 2D FP32 gradient tensor [seq_len, embed_dim]
// Outputs:
//   grad_byte_emb: 2D FP32 gradient w.r.t base byte embeddings [seq_len, embed_dim]
//   grad_tables: Array of 2D FP32 gradients w.r.t hash tables [per_ngram_vocab, embed_dim].
//               MUST be zero-initialized by the caller before calling this function.
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
