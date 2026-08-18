#ifndef BLT_MASK_BUILDER_H
#define BLT_MASK_BUILDER_H

#include "blt/core/tensor.h"
#include "blt/core/allocator.h"

#include <stdbool.h>

typedef struct {
    size_t seq_len_q;   // Query sequence length
    size_t seq_len_kv;  // Key/Value sequence length
    
    // For Sliding Window + Document Boundaries (Self-Attention)
    size_t sliding_window;  // 0 for full causal attention
    const size_t* doc_boundaries;   // Array of index where new docs start
    size_t num_docs;
    
    // For block-diagonal / grouped attention (Cross-Attention & Fast-BLT blocks)
    const size_t* query_group_ids;  // group ID for each query position
    const size_t* kv_group_ids;     // group ID for each kv position
    bool bidirectional_within_group;    // if true, attend to all positions in group; if false, causal within group
    
    bool is_causal; // apply causal mask
} blt_mask_config;

// Creates a 2D FP32 mask tensor of shape [seq_len_q, seq_len_kv]
// 0 for allowed attention and -INFINITY for masked attention
void blt_build_attention_mask(const blt_mask_config* config, blt_tensor* out_mask, blt_arena* arena);

#endif