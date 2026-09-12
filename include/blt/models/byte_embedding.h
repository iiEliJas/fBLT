#ifndef BLT_MODEL_BYTE_EMBEDDING_H
#define BLT_MODEL_BYTE_EMBEDDING_H

#ifdef __cplusplus
extern "C" {
#endif
#include "blt/core/tensor.h"
#include "blt/core/allocator.h"

typedef struct {
    size_t vocab_size; // 256 or 257 (with MASK token)
    blt_tensor weight; // [vocab_size, embed_dim] FP32
    size_t embed_dim;
} blt_byte_embedding;

// Allocates and zero-initializes the embedding table. Caller fills weight->data
blt_byte_embedding blt_byte_embedding_create(blt_arena* arena, size_t vocab_size, size_t embed_dim);

// bytes_in: 1D UINT8 tensor [seq_len] (raw byte values 0-255)
// out: 2D FP32 tensor [seq_len, embed_dim]
void blt_byte_embedding_forward(const blt_byte_embedding* emb, const blt_tensor* bytes_in,
                                 blt_tensor* out);

// grad_out: [seq_len, embed_dim], scatter-adds into grad_weight [256, embed_dim]
// (must be zeroed by caller before accumulating across a batch)
void blt_byte_embedding_backward(const blt_byte_embedding* emb, const blt_tensor* bytes_in,
                                  const blt_tensor* grad_out, blt_tensor* grad_weight);

#ifdef __cplusplus
}
#endif
#endif
