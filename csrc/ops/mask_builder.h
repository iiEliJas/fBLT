#ifndef BLT_MASK_BUILDER_H
#define BLT_MASK_BUILDER_H

#ifdef __cplusplus
extern "C" {
#endif
#include "core/tensor.h"
#include "core/allocator.h"

#include <stdbool.h>

typedef struct {
    size_t seq_len_q;
    size_t seq_len_kv;

    size_t sliding_window; // 0 for full causal attention
    // Builder convention: doc_boundaries[d] is the first position of document
    // d+1 for d in [0, num_docs-1); document 0 starts implicitly at 0.
    // The array therefore has num_docs-1 valid entries.
    const size_t *doc_boundaries;
    size_t num_docs;

    const size_t *query_group_ids;
    const size_t *kv_group_ids;
    bool bidirectional_within_group; // if true, attend to all positions in group; if false, causal within group

    bool is_causal;
    size_t causal_offset; // is_causal only: query row i may attend keys
                          // j <= i + causal_offset (chunked/incremental
                          // decode over a cached prefix sets this to the
                          // prefix length; 0 == plain causal)
} blt_mask_config;

// 0 for allowed attention and -INFINITY for masked attention
void blt_build_attention_mask(const blt_mask_config *config, blt_tensor *out_mask, blt_arena *arena);

// BLT-D block-diffusion self-attention masks (Fast-BLT section 3.1.1 / 3.2.2)
//
// Layout: the decoder sequence is [clean prefix ; block section], with the
// clean prefix occupying rows/cols [0, num_clean) and the block section
// rows/cols [num_clean, S).

typedef enum {
    BLT_BDM_TRAIN = 0, // Fast-BLT 3.2.2 prose rule: clean rows causal; block row i
                       // sees all clean bytes plus every block with block-index <=
                       // i's block-index (bidirectional within own block); pinned
                       // by the fixture test. Figure 5's matrix is strictly
                       // causal -- prose rule adopted.
    BLT_BDM_INFER      // Fast-BLT section 3.1.1: clean rows causal; every block row
                       // sees all clean positions and the WHOLE block section
                       // bidirectionally (single live block during generation).
} blt_block_diffusion_mode;

typedef struct {
    blt_block_diffusion_mode mode;
    size_t seq_len;    // total decoder sequence length S (square mask)
    size_t num_clean;  // N: clean prefix length (rows [0, N))
    size_t block_size; // B; TRAIN only — blocks tile [N, N + B*(M-1)) exactly.
                       // INFER ignores it (everything past N is one region).
} blt_block_diffusion_config;

// Creates a square FP32 additive mask [seq_len, seq_len]:
// 0 for allowed attention and -INFINITY for masked attention.
// Semantics fixed by Fast-BLT 3.2.2 prose (TRAIN) / section 3.1.1 + Figure 3
// (INFER); fixture tests pin both to hardcoded matrices.
void blt_build_block_diffusion_mask(const blt_block_diffusion_config *config, blt_tensor *out_mask, blt_arena *arena);

#ifdef __cplusplus
}
#endif
#endif
