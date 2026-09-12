#ifndef BLT_FLOPS_CALCULATOR_H
#define BLT_FLOPS_CALCULATOR_H

#include <stddef.h>


// F1: Per-operation FLOPs primitives (Table 11)

// Attention(l, h_k, n_heads, m) = 4 * l * h_k * n_heads * (m+1)/2
double blt_flops_attention(size_t l, size_t h_k, size_t n_heads, size_t m);

// QKVO(l, h, r) = (r*2 + 2) * 2 * l * h^2
// r is the ratio of queries to key/values (can be fractional, e.g. n_p/k),
// so it is taken as a double rather than size_t.
double blt_flops_qkvo(size_t l, size_t h, double r);

// Feed-forward(l, h, d_ff) = 2 * l * 2 * h * (d_ff * h)
// d_ff is the feed-forward dimension MULTIPLIER (paper uses d_ff = 4),
// i.e. the absolute FFN width is d_ff * h.
double blt_flops_feedforward(size_t l, size_t h, size_t d_ff);

// De-Embedding(h, V) = 2 * h * |V|
double blt_flops_deembedding(size_t h, size_t vocab_size);

// Cross-Attention(l, h_k, n_heads, patch_size, r)
//   = Attention(l, h_k, n_heads, patch_size) + QKVO(l, h_k * n_heads, r)
double blt_flops_cross_attention(size_t l, size_t h_k, size_t n_heads,
                                  size_t patch_size, double r);


// F2: Per-module transformer FLOPs (Eq. 19-22)

// Transformer-FLOPs(l, h, m, n_heads, h_k, d_ff, V)
//   = Feed-forward(l, h, d_ff)
//   + QKVO(l, h, r=1)
//   + Attention(l, h_k, n_heads, m)
//   + De-Embedding(h, V)
//
// Pass vocab_size = 0 to skip de-embedding (encoder/global model have none;
// only the local decoder has V = 256).
double blt_flops_transformer_block(
    size_t l, size_t h, size_t m, size_t n_heads, size_t h_k,
    size_t d_ff, size_t vocab_size
);


// F3: Full BLT FLOPs-per-byte (Eq. 13-17)

typedef struct {
    // Global transformer
    size_t h_G, l_G, n_ctx;      // n_ctx = byte context length
    size_t global_heads;
    size_t global_head_dim;
    size_t d_ff_G;
    // Local encoder
    size_t h_E, l_E, w_E;        // w_E = local window
    size_t enc_heads, enc_head_dim;
    size_t d_ff_E;
    // Local decoder
    size_t h_D, l_D, w_D;
    size_t dec_heads, dec_head_dim;
    size_t d_ff_D;
    // Shared
    size_t n_p;                  // average patch size (bytes/patch)
    size_t k;                    // encoder/decoder cross-attn head-split factor
    size_t vocab_size;           // 256, decoder only
} blt_flops_config;

double blt_flops_per_byte(const blt_flops_config* cfg);

#endif